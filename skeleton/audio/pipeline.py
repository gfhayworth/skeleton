"""Unified end-to-end audio pipeline coordinating STT, dialogue, TTS, and mouth sync."""

import time
from typing import List, Optional, Union
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

    async def process_text_turn(
        self,
        user_transcript: str,
        vision_context: Optional[VisionContext] = None,
        t0: Optional[float] = None,
    ) -> AudioDialogueResponse:
        """Processes text input directly into spoken skeleton audio and mouth-sync frames."""
        if t0 is None:
            t0 = time.perf_counter()

        # 1. Generate dialogue from LLM
        chunks: List[str] = []
        async for chunk in self.orchestrator.process_utterance(user_transcript, vision_context):
            chunks.append(chunk.text)

        full_skeleton_text = " ".join(chunks).strip()
        if not full_skeleton_text:
            full_skeleton_text = "I have nothing to say to that."

        # 2. Synthesize deep male voice (onyx @ 0.90x in WAV)
        wav_audio = await self.tts_client.synthesize(
            full_skeleton_text,
            voice=self.config.tts_voice,
            speed=self.config.tts_speed,
            response_format=self.config.tts_format,
        )

        # 3. Extract 30Hz jaw-opening trajectory
        mouth_frames = self.mouth_sync.extract_jaw_trajectory(wav_audio)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return AudioDialogueResponse(
            user_transcript=user_transcript,
            skeleton_response=full_skeleton_text,
            audio_bytes=wav_audio,
            mouth_frames=mouth_frames,
            total_latency_ms=round(elapsed_ms, 2),
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
