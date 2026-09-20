"""Tests for streaming LLM client and dual-phase timeouts."""

import pytest
from skeleton.dialogue.config import DialogueConfig
from skeleton.dialogue.llm_client import MockLLMClient


@pytest.mark.asyncio
async def test_mock_streaming_tokens():
    mock = MockLLMClient(
        simulated_response="I have no stomach for your nonsense.",
        ttft_delay_s=0.01,
        inter_token_delay_s=0.005,
    )
    tokens = [t async for t in mock.stream_tokens([])]
    joined = "".join(tokens).strip()
    assert joined == "I have no stomach for your nonsense."


@pytest.mark.asyncio
async def test_phase1_pre_emission_timeout_triggers_fallback():
    config = DialogueConfig(
        pre_emission_timeout_s=0.05,
        canned_fallbacks=["My funny bone is completely disconnected."],
    )
    mock = MockLLMClient(
        trigger_pre_emission_timeout=True,
        config=config,
    )

    tokens = [t async for t in mock.stream_tokens([])]
    emitted_text = "".join(tokens).strip()
    assert emitted_text == "My funny bone is completely disconnected."


@pytest.mark.asyncio
async def test_phase2_stream_stall_clean_abort():
    config = DialogueConfig(
        stream_stall_timeout_s=0.05,
        canned_fallbacks=["Should not speak this fallback."],
    )
    mock = MockLLMClient(
        simulated_response="Word1 Word2 Word3 Word4 Word5 Word6 Word7",
        trigger_stream_stall=True,
        stall_after_tokens=3,
        config=config,
    )

    tokens = [t async for t in mock.stream_tokens([])]
    emitted_text = "".join(tokens).strip()

    # Should have emitted the first 3 tokens, then cleanly stopped without double-speaking fallback
    assert emitted_text == "Word1 Word2 Word3"
    assert "Should not speak" not in emitted_text
