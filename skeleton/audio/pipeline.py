"""Unified end-to-end audio pipeline coordinating STT, dialogue, TTS, and mouth sync."""

import io
import time
import wave
from typing import AsyncGenerator, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from skeleton.audio.config import AudioConfig
from skeleton.audio.mouth_sync import MouthSyncFrame, MouthSyncProcessor
from skeleton.audio.recorder import AudioRecorder, BaseAudioRecorder
from skeleton.audio.sounds import PreRecordedSound, SoundBank, SoundCategory
from skeleton.audio.stt import BaseSTTClient, WhisperSTTClient
from skeleton.audio.tts import BaseTTSClient, OpenAITTSClient
from skeleton.dialogue.models import VisionContext
from skeleton.dialogue.orchestrator import DialogueOrchestrator


class AudioDialogueResponse(BaseModel):
    """Complete multimodal output package for a single conversational turn."""

    user_transcript: str = Field(..., description="Recognized user speech input.")
    skeleton_response: str = Field(..., description="Generated snarky skeleton text response.")
    audio_bytes: bytes = Field(..., description="Synthesized deep male WAV audio.")
    mouth_frames: List[MouthSyncFrame] = Field(..., description="30Hz synchronized jaw servo trajectory.")
    total_latency_ms: float = Field(..., description="Total roundtrip turnaround time in milliseconds.")
    time_to_first_audio_ms: Optional[float] = Field(
        default=None,
        description="Time in milliseconds to synthesize and produce the first streaming voice chunk.",
    )


class AudioChunkResponse(BaseModel):
    """A streaming voice and mouth synchronization chunk for real-time playback."""

    chunk_index: int = Field(..., description="0-indexed sequence of this speech chunk.")
    is_final: bool = Field(..., description="True if this is the terminating chunk of the response.")
    text: str = Field(..., description="Sanitized dialogue text fragment.")
    audio_bytes: bytes = Field(..., description="Synthesized WAV audio chunk.")
    mouth_frames: List[MouthSyncFrame] = Field(..., description="30Hz jaw-sync frame trajectory for this chunk.")
    latency_ms: float = Field(..., description="Elapsed milliseconds from turn start until chunk readiness.")
    cumulative_duration_s: float = Field(..., description="Total audio seconds elapsed before this chunk.")



class SkeletonAudioPipeline:
    """Orchestrates the complete audio loop:

    Audio In -> Whisper STT -> Dialogue Engine (Gemini) -> OpenAI TTS (Onyx) -> Mouth Sync.
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        stt_client: Optional[BaseSTTClient] = None,
        tts_client: Optional[BaseTTSClient] = None,
        orchestrator: Optional[DialogueOrchestrator] = None,
        mouth_sync: Optional[MouthSyncProcessor] = None,
        recorder: Optional[BaseAudioRecorder] = None,
    ):
        self.config = config or AudioConfig()
        self.stt_client = stt_client or WhisperSTTClient(self.config)
        self.tts_client = tts_client or OpenAITTSClient(self.config)
        self.orchestrator = orchestrator or DialogueOrchestrator()
        self.mouth_sync = mouth_sync or MouthSyncProcessor(self.config)
        self.recorder = recorder or AudioRecorder(sample_rate=self.config.sample_rate)
        self.sound_bank = SoundBank(
            sounds_dir=self.config.sounds_dir,
            config=self.config,
            mouth_sync=self.mouth_sync,
        )

    async def record_and_process(
        self,
        duration_seconds: Optional[float] = None,
        device_index: Optional[int] = None,
        vision_context: Optional[VisionContext] = None,
        use_vad: Optional[bool] = None,
    ) -> AudioDialogueResponse:
        """Records microphone speech asynchronously and runs the full conversational audio loop.

        Args:
            duration_seconds: Duration of voice capture in seconds (0.5 - 30.0). If use_vad is True
                and duration_seconds is not explicitly passed, VAD dynamic endpointing is used.
            device_index: Optional physical input device index. None selects the system default.
            vision_context: Optional computer vision environmental awareness.
            use_vad: Whether to use VAD endpointing. Defaults to config.vad_enabled.

        Returns:
            AudioDialogueResponse with user transcript, skeleton response, synthesized audio, and mouth frames.
        """
        is_vad = self.config.vad_enabled if use_vad is None else use_vad

        if is_vad and duration_seconds is None:
            raw_wav = await self.recorder.record_with_vad_async(
                device_index=device_index,
                silence_duration_s=self.config.vad_silence_duration_s,
                max_recording_s=self.config.vad_max_recording_s,
                aggressiveness=self.config.vad_aggressiveness,
                frame_duration_ms=self.config.vad_frame_duration_ms,
            )
        else:
            raw_wav = await self.recorder.record_async(
                duration_seconds=duration_seconds or 3.5,
                device_index=device_index,
            )
        return await self.process_audio_turn(
            audio_bytes=raw_wav,
            filename="user_recording.wav",
            vision_context=vision_context,
        )

    async def process_audio_turn(
        self,
        audio_bytes: bytes,
        filename: str = "speech.wav",
        vision_context: Optional[VisionContext] = None,
    ) -> AudioDialogueResponse:
        """Processes raw microphone input into full spoken audio and mechanical jaw tracking."""
        t0 = time.perf_counter()

        # 1. Transcribe audio input using Whisper
        transcript = await self.stt_client.transcribe(audio_bytes, filename=filename)

        # 2. Process text through dialogue pipeline
        response = await self.process_text_turn(
            transcript, vision_context=vision_context, t0=t0
        )
        return response

    async def process_text_stream(
        self,
        user_transcript: str,
        vision_context: Optional[VisionContext] = None,
        t0: Optional[float] = None,
    ) -> AsyncGenerator[AudioChunkResponse, None]:
        """Streams synthesized voice chunks and real-time jaw-sync frames as dialogue is generated.

        This drastically reduces Time-To-First-Audio (TTFA) to the latency of the first sentence
        chunk, allowing audio playback and skull jaw movement to begin immediately.
        """
        if t0 is None:
            t0 = time.perf_counter()

        cumulative_duration_s = 0.0
        cumulative_time_offset_ms = 0.0

        async for chunk in self.orchestrator.process_utterance(user_transcript, vision_context):
            chunk_text = chunk.text.strip()
            if not chunk_text:
                continue

            # Synthesize voice chunk immediately
            chunk_wav = await self.tts_client.synthesize(
                chunk_text,
                voice=self.config.tts_voice,
                speed=self.config.tts_speed,
                response_format=self.config.tts_format,
            )

            # Compute mouth frames offset by previously emitted audio duration
            chunk_frames = self.mouth_sync.extract_jaw_trajectory(
                chunk_wav,
                time_offset_ms=cumulative_time_offset_ms,
            )

            # Measure duration of this WAV chunk to update timeline offset
            try:
                with wave.open(io.BytesIO(chunk_wav), "rb") as wf:
                    frames_count = wf.getnframes()
                    rate = wf.getframerate()
                    chunk_duration_s = frames_count / float(rate) if rate > 0 else 0.0
            except Exception:
                chunk_duration_s = len(chunk_frames) / float(self.config.mouth_fps)

            now_ms = (time.perf_counter() - t0) * 1000.0

            yield AudioChunkResponse(
                chunk_index=chunk.sequence_index,
                is_final=chunk.is_final,
                text=chunk_text,
                audio_bytes=chunk_wav,
                mouth_frames=chunk_frames,
                latency_ms=round(now_ms, 2),
                cumulative_duration_s=round(cumulative_duration_s, 3),
            )

            cumulative_duration_s += chunk_duration_s
            cumulative_time_offset_ms += chunk_duration_s * 1000.0

    async def process_text_turn(
        self,
        user_transcript: str,
        vision_context: Optional[VisionContext] = None,
        t0: Optional[float] = None,
    ) -> AudioDialogueResponse:
        """Processes text input into spoken skeleton audio and mouth-sync frames.

        Uses streaming chunk synthesis underneath so Time-To-First-Audio is tracked.
        """
        if t0 is None:
            t0 = time.perf_counter()

        text_chunks: List[str] = []
        raw_pcm_chunks: List[bytes] = []
        all_mouth_frames: List[MouthSyncFrame] = []
        time_to_first_audio_ms: Optional[float] = None

        sample_rate = self.config.sample_rate
        sample_width = 2
        num_channels = 1

        async for chunk_resp in self.process_text_stream(user_transcript, vision_context=vision_context, t0=t0):
            if time_to_first_audio_ms is None:
                time_to_first_audio_ms = chunk_resp.latency_ms

            text_chunks.append(chunk_resp.text)
            all_mouth_frames.extend(chunk_resp.mouth_frames)

            # Extract PCM audio bytes from WAV
            try:
                with wave.open(io.BytesIO(chunk_resp.audio_bytes), "rb") as wf:
                    num_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    sample_rate = wf.getframerate()
                    raw_pcm_chunks.append(wf.readframes(wf.getnframes()))
            except Exception:
                pass

        full_skeleton_text = " ".join(text_chunks).strip()
        if not full_skeleton_text:
            full_skeleton_text = "I have nothing to say to that."
            fallback_wav = await self.tts_client.synthesize(
                full_skeleton_text,
                voice=self.config.tts_voice,
                speed=self.config.tts_speed,
                response_format=self.config.tts_format,
            )
            all_mouth_frames = self.mouth_sync.extract_jaw_trajectory(fallback_wav)
            combined_wav = fallback_wav
        else:
            # Assemble seamless combined WAV file
            combined_buf = io.BytesIO()
            with wave.open(combined_buf, "wb") as wf:
                wf.setnchannels(num_channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(b"".join(raw_pcm_chunks))
            combined_wav = combined_buf.getvalue()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return AudioDialogueResponse(
            user_transcript=user_transcript,
            skeleton_response=full_skeleton_text,
            audio_bytes=combined_wav,
            mouth_frames=all_mouth_frames,
            total_latency_ms=round(elapsed_ms, 2),
            time_to_first_audio_ms=time_to_first_audio_ms,
        )

    def play_prerecorded_sound(
        self,
        sound_id: Optional[str] = None,
        category: Optional[Union[SoundCategory, str]] = None,
    ) -> Optional[AudioDialogueResponse]:
        """Instantly retrieves a pre-recorded sound response with pre-computed mouth frames.

        Eliminates LLM and TTS processing latency (yielding sub-millisecond dispatch time).
        """
        t0 = time.perf_counter()
        sound: Optional[PreRecordedSound] = None

        if sound_id:
            sound = self.sound_bank.get(sound_id)
        elif category:
            sound = self.sound_bank.get_random(category)
        else:
            sound = self.sound_bank.get_random()

        if sound is None:
            return None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return AudioDialogueResponse(
            user_transcript="[pre-recorded trigger]",
            skeleton_response=sound.text,
            audio_bytes=sound.audio_bytes,
            mouth_frames=sound.mouth_frames,
            total_latency_ms=round(elapsed_ms, 3),
        )

    async def close(self) -> None:
        """Closes all active network connections."""
        await self.stt_client.close()
        await self.tts_client.close()
        await self.orchestrator.close()
