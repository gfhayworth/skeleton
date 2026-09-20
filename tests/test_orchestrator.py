"""Tests for dialogue orchestrator, state management, and barge-in cancellation."""

import asyncio
import pytest
from skeleton.dialogue.config import DialogueConfig
from skeleton.dialogue.llm_client import MockLLMClient
from skeleton.dialogue.models import DialogueState, VisionContext
from skeleton.dialogue.orchestrator import DialogueOrchestrator, VisionStateStore
from skeleton.dialogue.sentence_chunker import SentenceChunker


@pytest.mark.asyncio
async def test_vision_state_store_thread_safety():
    store = VisionStateStore()
    v1 = VisionContext(pan_angle_deg=10.0, detected_objects=["hat"])
    await store.update(v1)

    snapshot = await store.get_snapshot()
    assert snapshot is not None
    assert snapshot.pan_angle_deg == 10.0
    assert snapshot.detected_objects == ["hat"]


@pytest.mark.asyncio
async def test_orchestrator_end_to_end_chunks():
    mock_client = MockLLMClient(
        simulated_response="Look at that hat. Did a pirate drop it?",
        ttft_delay_s=0.01,
        inter_token_delay_s=0.005,
    )
    orchestrator = DialogueOrchestrator(
        llm_client=mock_client,
        chunker=SentenceChunker(),
    )

    vision = VisionContext(detected_objects=["pirate hat"])
    chunks = []
    async for chunk in orchestrator.process_utterance("What do you see?", vision):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].text == "Look at that hat."
    assert chunks[1].text == "Did a pirate drop it?"
    assert orchestrator.state == DialogueState.IDLE

    # History should be updated
    assert len(orchestrator.history) == 2
    assert orchestrator.history[0].role == "user"
    assert orchestrator.history[1].role == "assistant"
    assert "Look at that hat." in orchestrator.history[1].content


@pytest.mark.asyncio
async def test_orchestrator_barge_in_cancellation():
    # Long simulated response with deliberate delays
    mock_client = MockLLMClient(
        simulated_response="Sentence one is speaking right now. Sentence two will be interrupted.",
        ttft_delay_s=0.02,
        inter_token_delay_s=0.05,
    )
    orchestrator = DialogueOrchestrator(
        llm_client=mock_client,
        chunker=SentenceChunker(),
    )

    received_chunks = []

    async def consume_first_utterance():
        async for chunk in orchestrator.process_utterance("First message"):
            received_chunks.append(chunk)
            # Simulate human interrupting immediately after first chunk arrives
            if len(received_chunks) == 1:
                break

    await consume_first_utterance()
    assert len(received_chunks) == 1
    assert received_chunks[0].text == "Sentence one is speaking right now."

    # Now human interrupts with a new utterance
    mock_client.simulated_response = "Fine, interrupt me then."
    chunks_second = []
    async for chunk in orchestrator.process_utterance("Wait skeleton, shut up"):
        chunks_second.append(chunk)

    assert len(chunks_second) >= 1
    assert "Fine, interrupt me then." in [c.text for c in chunks_second]
