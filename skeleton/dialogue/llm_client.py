"""Low-latency streaming LLM client with connection pooling and dual-phase timeouts."""

import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Optional
import httpx
from skeleton.dialogue.config import DialogueConfig


class BaseLLMClient:
    """Abstract base class for streaming LLM clients."""

    async def stream_tokens(
        self,
        messages: List[dict],
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class LowLatencyLLMClient(BaseLLMClient):
    """Production streaming client using persistent HTTP/2 connection pooling

    over OpenAI-compatible endpoints (Groq, Cerebras, Together, local vLLM).
    """

    def __init__(self, config: Optional[DialogueConfig] = None):
        self.config = config or DialogueConfig()
        api_key_val = (
            self.config.api_key.get_secret_value()
            if self.config.api_key
            else "mock-or-env-key"
        )
        self.headers = {
            "Authorization": f"Bearer {api_key_val}",
            "Content-Type": "application/json",
        }
        # Configure persistent connection pool to eliminate TLS cold starts
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/"),
            headers=self.headers,
            limits=httpx.Limits(
                max_keepalive_connections=self.config.max_keepalive_connections,
                keepalive_expiry=self.config.keepalive_expiry_s,
            ),
            timeout=httpx.Timeout(
                connect=5.0,
                read=max(5.0, self.config.pre_emission_timeout_s),
                write=5.0,
                pool=3.0,
            ),
        )

    async def stream_tokens(
        self,
        messages: List[dict],
    ) -> AsyncGenerator[str, None]:
        """Streams raw token deltas from the provider with dual-phase timeout protection.

        Phase 1 (pre-emission): Triggers canned fallback if first token exceeds deadline.
        Phase 2 (inter-token): Cleanly aborts stream if connection stalls mid-stream.
        """
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_response_tokens,
            "stream": True,
        }

        # Disable thinking latency for Gemini models to ensure instant conversational streaming
        if "gemini" in self.config.model.lower() or "googleapis.com" in self.config.base_url:
            payload["reasoning_effort"] = "none"

        tokens_emitted = 0

        try:
            # Phase 1: Pre-emission timeout guard for Time-To-First-Token
            async with asyncio.timeout(self.config.pre_emission_timeout_s):
                req = self._client.build_request("POST", "/chat/completions", json=payload)
                response = await self._client.send(req, stream=True)
                response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            tokens_emitted += 1
                            yield content
                except json.JSONDecodeError:
                    continue

        except (asyncio.TimeoutError, httpx.TimeoutException, httpx.RequestError) as exc:
            if tokens_emitted == 0:
                # Phase 1 fallback: No tokens emitted yet, produce instant canned snark
                fallback = random.choice(self.config.canned_fallbacks)
                for word in fallback.split():
                    yield word + " "
            else:
                # Phase 2: Tokens already emitted to downstream TTS/mouth.
                # Cleanly truncate without double-speaking an unrelated canned line.
                pass
        finally:
            if "response" in locals():
                await response.aclose()

    async def close(self) -> None:
        """Closes the underlying persistent HTTP client pool."""
        await self._client.aclose()


class MockLLMClient(BaseLLMClient):
    """Deterministic offline mock client for tests, benchmarks, and interactive notebooks.

    Allows zero-cost demonstration and precise latency simulation.
    """

    def __init__(
        self,
        simulated_response: Optional[str] = None,
        ttft_delay_s: float = 0.05,
        inter_token_delay_s: float = 0.02,
        trigger_pre_emission_timeout: bool = False,
        trigger_stream_stall: bool = False,
        stall_after_tokens: int = 4,
        config: Optional[DialogueConfig] = None,
    ):
        self.simulated_response = simulated_response or (
            "Well, look who finally decided to show up. Nice shirt, did it come with a pulse?"
        )
        self.ttft_delay_s = ttft_delay_s
        self.inter_token_delay_s = inter_token_delay_s
        self.trigger_pre_emission_timeout = trigger_pre_emission_timeout
        self.trigger_stream_stall = trigger_stream_stall
        self.stall_after_tokens = stall_after_tokens
        self.config = config or DialogueConfig()

    async def stream_tokens(
        self,
        messages: List[dict],
    ) -> AsyncGenerator[str, None]:
        if self.trigger_pre_emission_timeout:
            # Simulate timeout before first token
            await asyncio.sleep(self.config.pre_emission_timeout_s + 0.1)
            fallback = random.choice(self.config.canned_fallbacks)
            for word in fallback.split():
                yield word + " "
            return

        # Simulate TTFT delay
        if self.ttft_delay_s > 0:
            await asyncio.sleep(self.ttft_delay_s)

        tokens = self.simulated_response.split(" ")
        emitted = 0

        for i, token in enumerate(tokens):
            if self.trigger_stream_stall and emitted >= self.stall_after_tokens:
                # Simulate stall mid-stream
                await asyncio.sleep(self.config.stream_stall_timeout_s + 0.1)
                return

            suffix = " " if i < len(tokens) - 1 else ""
            yield token + suffix
            emitted += 1

            if self.inter_token_delay_s > 0:
                await asyncio.sleep(self.inter_token_delay_s)
