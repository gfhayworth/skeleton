"""Configuration settings for the dialogue subsystem."""

from typing import List, Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DialogueConfig(BaseSettings):
    """Configuration options for low-latency dialogue and LLM streaming."""

    model_config = SettingsConfigDict(
        env_prefix="SKELETON_LLM_",
        env_file=".env",
        extra="ignore",
    )

    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for the OpenAI-compatible streaming LLM provider.",
    )
    base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL for the streaming LLM API endpoint.",
    )
    model: str = Field(
        default="llama-3.1-8b-instant",
        description="Model name to invoke for ultra-low latency.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for witty responses.",
    )
    max_response_tokens: int = Field(
        default=60,
        description="Maximum tokens generated (caps spoken output to ~30 words).",
    )

    # Latency and timeout parameters
    pre_emission_timeout_s: float = Field(
        default=1.0,
        description="Maximum seconds to wait for Time-To-First-Token before triggering canned fallback.",
    )
    stream_stall_timeout_s: float = Field(
        default=0.4,
        description="Maximum seconds between consecutive tokens before cleanly truncating stream.",
    )

    # Prompt and memory bounds
    max_history_turns: int = Field(
        default=4,
        description="Maximum conversation turns retained in active sliding context.",
    )
    max_context_tokens: int = Field(
        default=256,
        description="Strict token budget ceiling for context history.",
    )

    # Network connection pooling
    keepalive_expiry_s: float = Field(
        default=60.0,
        description="Seconds to keep idle HTTP/2 connections alive in pool.",
    )
    max_keepalive_connections: int = Field(
        default=5,
        description="Maximum pooled persistent connections to avoid TLS handshakes.",
    )

    # Pre-cached snarky fallback lines for animatronic persona
    canned_fallbacks: List[str] = Field(
        default_factory=lambda: [
            "Did you really just leave me hanging? My skull hurts thinking about that.",
            "I would roll my eyes at that, but I seem to have misplaced them.",
            "Fascinating. Tell me more, or don't, I am only bones anyway.",
            "I'm waiting with bated breath, metaphorically speaking of course.",
            "Hold on, my funny bone needs a quick recalibration.",
        ]
    )
