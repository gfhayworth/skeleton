"""Unified end-to-end audio pipeline coordinating STT, dialogue, TTS, and mouth sync."""

import time
from typing import List, Optional
from pydantic import BaseModel, Field
from skeleton.audio.config import AudioConfig
from skeleton.audio.mouth_sync import MouthSyncFrame, MouthSyncProcessor
from skeleton.audio.recorder import AudioRecorder, BaseAudioRecorder
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

    async def record_and_process(
        self,
        duration_seconds: float = 3.5,
        device_index: Optional[int] = None,
        vision_context: Optional[VisionContext] = None,
    ) -> AudioDialogueResponse:
        """Records microphone speech asynchronously and runs the full conversational audio loop.

        Args:
            duration_seconds: Duration of voice capture in seconds (0.5 - 30.0).
            device_index: Optional physical input device index. None selects the system default.
            vision_context: Optional computer vision environmental awareness.

        Returns:
            AudioDialogueResponse with user transcript, skeleton response, synthesized audio, and mouth frames.
        """
        raw_wav = await self.recorder.record_async(
            duration_seconds=duration_seconds,
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

    async def close(self) -> None:
        """Closes all active network connections."""
        await self.stt_client.close()
        await self.tts_client.close()
        await self.orchestrator.close()
