"""Text-to-Speech client synthesizing deep male skeleton voice via OpenAI tts-1."""

import io
import math
import struct
import wave
from typing import Optional
import httpx
from skeleton.audio.config import AudioConfig


class BaseTTSClient:
    """Abstract base class for speech synthesis clients."""

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        response_format: Optional[str] = None,
    ) -> bytes:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class OpenAITTSClient(BaseTTSClient):
    """Production TTS client generating deep, sinister skeleton speech with voice 'onyx'."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        api_key_val = (
            self.config.api_key.get_secret_value()
            if self.config.api_key
            else "mock-key"
        )
        self.headers = {
            "Authorization": f"Bearer {api_key_val}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/"),
            headers=self.headers,
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0),
        )

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        response_format: Optional[str] = None,
    ) -> bytes:
        """Synthesizes text into audio bytes.

        Defaults to the deep male 'onyx' voice at 0.90x speed in standard WAV format.
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize empty text.")

        payload = {
            "model": self.config.tts_model,
            "input": clean_text,
            "voice": voice or self.config.tts_voice,
            "speed": speed if speed is not None else self.config.tts_speed,
            "response_format": response_format or self.config.tts_format,
        }

        response = await self._client.post("/audio/speech", json=payload)
        response.raise_for_status()
        return response.content

    async def close(self) -> None:
        """Closes the underlying HTTP client session."""
        await self._client.aclose()


class MockTTSClient(BaseTTSClient):
    """Deterministic offline mock TTS client generating valid synthetic WAV audio.

    Uses only the Python standard library (wave and struct) to avoid external dependencies.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        response_format: Optional[str] = None,
    ) -> bytes:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize empty text.")

        # Duration proportional to word count (~0.25s per word, minimum 0.5s)
        duration_s = max(0.5, len(clean_text.split()) * 0.25)
        num_samples = int(self.sample_rate * duration_s)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)

            # Generate modulated 150 Hz tone (deep pitch) simulating voice speech pulses
            frames = bytearray()
            freq = 150.0  # Deep skeletal vocal frequency in Hz
            for i in range(num_samples):
                t = i / self.sample_rate
                # Amplitude modulated by low-frequency envelope to emulate syllable bursts
                syllable_mod = (math.sin(2 * math.pi * 3.5 * t) + 1.0) / 2.0
                sample = int(12000 * syllable_mod * math.sin(2 * math.pi * freq * t))
                frames.extend(struct.pack("<h", sample))

            wav_file.writeframes(frames)

        return buffer.getvalue()
