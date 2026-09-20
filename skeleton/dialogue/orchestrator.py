"""Dialogue orchestrator managing state, vision context synchronization, and barge-in cancellation."""

import asyncio
import time
from typing import AsyncGenerator, List, Optional
from skeleton.dialogue.config import DialogueConfig
from skeleton.dialogue.llm_client import BaseLLMClient, LowLatencyLLMClient
from skeleton.dialogue.models import ChatMessage, DialogueState, SpeechChunk, VisionContext
from skeleton.dialogue.prompt_builder import PromptBuilder
from skeleton.dialogue.sentence_chunker import SentenceChunker


class VisionStateStore:
    """Thread-safe async container for vision sensory state updates

    from the background OpenCV camera tracking loop.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_state: Optional[VisionContext] = None

    async def update(self, context: VisionContext) -> None:
        """Atomically updates the latest vision frame data."""
        async with self._lock:
            self._current_state = context

    async def get_snapshot(self) -> Optional[VisionContext]:
        """Atomically captures an immutable snapshot of the latest vision context."""
        async with self._lock:
            return self._current_state


class DialogueOrchestrator:
    """Coordinates incoming speech transcripts, multimodal vision context,

    streaming LLM inference, and barge-in interruptions.
    """

    def __init__(
        self,
        config: Optional[DialogueConfig] = None,
        llm_client: Optional[BaseLLMClient] = None,
        chunker: Optional[SentenceChunker] = None,
    ):
        self.config = config or DialogueConfig()
        self.llm_client = llm_client or LowLatencyLLMClient(self.config)
        self.chunker = chunker or SentenceChunker()
        self.prompt_builder = PromptBuilder(self.config)
        self.vision_store = VisionStateStore()

        self.state: DialogueState = DialogueState.IDLE
        self.history: List[ChatMessage] = []
        self._active_task: Optional[asyncio.Task] = None

    def cancel_active_turn(self) -> None:
        """Barge-In: Cancels active dialogue generation immediately when

        the user begins speaking again or an interruption event occurs.
        """
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            self._active_task = None
        self.state = DialogueState.IDLE

    async def update_vision(self, context: VisionContext) -> None:
        """Called by background vision worker to update tracking/object status."""
        await self.vision_store.update(context)

    async def process_utterance(
        self,
        user_transcript: str,
        vision_context: Optional[VisionContext] = None,
    ) -> AsyncGenerator[SpeechChunk, None]:
        """Main entry point for incoming audio receiver transcripts.

        Cancels any prior active generation (barge-in support) and streams
        sanitized speech chunks in real time.
        """
        # 1. Handle Barge-In: interrupt in-flight generation if skeleton is speaking
        self.cancel_active_turn()

        self.state = DialogueState.PROCESSING
        start_time = time.time()

        # 2. Acquire vision snapshot if not explicitly provided
        if vision_context is None:
            vision_context = await self.vision_store.get_snapshot()

        # 3. Assemble hardened prompt
        messages = self.prompt_builder.build_messages(
            user_transcript=user_transcript,
            vision_context=vision_context,
            history=self.history,
        )

        full_response_parts: List[str] = []

        try:
            # 4. Stream tokens through the acoustic clause chunker
            token_stream = self.llm_client.stream_tokens(messages)
            self.state = DialogueState.SPEAKING

            async for chunk in self.chunker.stream_chunks(token_stream, start_time=start_time):
                full_response_parts.append(chunk.text)
                yield chunk

            # 5. Update conversational memory
            full_response = " ".join(full_response_parts).strip()
            if full_response:
                self.history.append(
                    ChatMessage(role="user", content=user_transcript)
                )
                self.history.append(
                    ChatMessage(role="assistant", content=full_response)
                )
                # Keep history within configured limits
                self.history = self.prompt_builder.prune_history(self.history)

        except asyncio.CancelledError:
            # Cleanly handle barge-in cancellation
            self.state = DialogueState.IDLE
            raise
        finally:
            if self.state != DialogueState.IDLE:
                self.state = DialogueState.IDLE

    async def close(self) -> None:
        """Releases underlying network resources."""
        self.cancel_active_turn()
        await self.llm_client.close()
