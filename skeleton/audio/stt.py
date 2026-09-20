"""Speech-to-Text client wrapping OpenAI Whisper API with buffer safety checks."""

from typing import Optional
import httpx
from skeleton.audio.config import AudioConfig


class BaseSTTClient:
    """Abstract base class for speech-to-text clients."""

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = "en",
    ) -> str:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class WhisperSTTClient(BaseSTTClient):
    """Production STT client utilizing the OpenAI Whisper-1 endpoint."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        api_key_val = (
            self.config.api_key.get_secret_value()
            if self.config.api_key
            else "mock-key"
        )
        self.headers = {
            "Authorization": f"Bearer {api_key_val}",
        }
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/"),
            headers=self.headers,
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0),
        )

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = "en",
    ) -> str:
        """Transcribes raw audio bytes into text using Whisper-1.

        Args:
            audio_bytes: Raw audio content (WAV, MP3, WEBM, M4A).
            filename: Virtual filename with extension for MIME detection.
            language: Expected language hint (default 'en').

        Returns:
            Transcribed text string.

        Raises:
            ValueError: If audio buffer is outside safety bounds (100B to 10MB).
            RuntimeError: If Whisper transcription fails or returns an error.
        """
        size = len(audio_bytes)
        if size < self.config.min_audio_bytes:
            raise ValueError(
                f"Audio payload too small ({size} bytes). Minimum required: {self.config.min_audio_bytes} bytes."
            )
        if size > self.config.max_audio_bytes:
            raise ValueError(
                f"Audio payload exceeded limit ({size} bytes > {self.config.max_audio_bytes} bytes)."
            )

        # Detect content type from filename
        content_type = "audio/wav"
        if filename.endswith(".mp3"):
            content_type = "audio/mpeg"
        elif filename.endswith(".webm"):
            content_type = "audio/webm"
        elif filename.endswith(".m4a"):
            content_type = "audio/mp4"

        files = {
            "file": (filename, audio_bytes, content_type),
        }
        data = {
            "model": self.config.stt_model,
        }
        if language:
            data["language"] = language

        response = await self._client.post(
            "/audio/transcriptions",
            files=files,
            data=data,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("text", "").strip()

    async def close(self) -> None:
        """Closes the underlying HTTP client session."""
        await self._client.aclose()


class MockSTTClient(BaseSTTClient):
    """Deterministic offline mock STT client for tests and offline demonstrations."""

    def __init__(self, simulated_transcript: Optional[str] = None):
        self.simulated_transcript = (
            simulated_transcript or "Hello skeleton, why are you staring at me?"
        )

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = "en",
    ) -> str:
        if len(audio_bytes) < 100:
            raise ValueError("Audio payload too small.")
        return self.simulated_transcript
