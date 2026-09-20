"""Configuration settings for the Audio Subsystem (STT, TTS, and Mouth Sync)."""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioConfig(BaseSettings):
    """Configuration parameters for audio receiver, speech synthesizer, and mouth sync."""

    model_config = SettingsConfigDict(
        env_prefix="SKELETON_AUDIO_",
        env_file=".env",
        extra="ignore",
    )

    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for OpenAI audio endpoints (Whisper and TTS).",
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for OpenAI audio APIs.",
    )

    # STT (Whisper) Settings
    stt_model: str = Field(
        default="whisper-1",
        description="OpenAI Whisper model identifier for speech transcription.",
    )
    max_audio_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Maximum allowed audio buffer size for transcription.",
    )
    min_audio_bytes: int = Field(
        default=100,
        description="Minimum audio buffer size to avoid empty/corrupt uploads.",
    )

    # Audio Recording Settings
    sample_rate: int = Field(
        default=16000,
        description="Sampling rate for microphone recording in Hz (16 kHz optimal for Whisper).",
    )

    # TTS Settings (Skeleton Voice Persona: Deep Male)
    tts_model: str = Field(
        default="tts-1",
        description="OpenAI TTS model identifier (tts-1 is optimized for low latency).",
    )
    tts_voice: str = Field(
        default="onyx",
        description="Voice persona identifier ('onyx' is the deep, gravelly male voice).",
    )
    tts_speed: float = Field(
        default=0.90,
        ge=0.25,
        le=4.0,
        description="Speech playback rate (0.90x slows the cadence for an imposing, sinister tone).",
    )
    tts_format: str = Field(
        default="wav",
        description="Audio format for TTS output ('wav' enables standard library PCM decoding).",
    )

    # Mechanical Mouth Sync & Servo Protection
    max_jaw_angle_deg: float = Field(
        default=35.0,
        description="Maximum jaw-opening angle in degrees for the physical skull servo.",
    )
    noise_floor_rms: float = Field(
        default=0.04,
        description="Deadband threshold; audio RMS below this value clamps jaw angle to 0 degrees.",
    )
    mouth_fps: int = Field(
        default=30,
        description="Sampling rate for jaw-tracking frames in Hz (frames per second).",
    )

    def model_post_init(self, __context) -> None:
        """Loads OPENAI_API_KEY from .env if api_key is not explicitly supplied."""
        load_dotenv(".env")
        if not self.api_key or self.api_key.get_secret_value() in ("", "mock-or-env-key"):
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                self.api_key = SecretStr(openai_key)
