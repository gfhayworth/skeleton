"""Tests for prompt builder and context fencing."""

import pytest
from skeleton.dialogue.config import DialogueConfig
from skeleton.dialogue.models import ChatMessage, VisionContext
from skeleton.dialogue.prompt_builder import PromptBuilder


def test_system_persona_content():
    builder = PromptBuilder()
    sys_msg = builder.build_system_message()
    assert sys_msg["role"] == "system"
    assert "cynical skeleton" in sys_msg["content"]
    assert "CRITICAL OPERATIONAL RULES" in sys_msg["content"]
    assert "Spoken Audio Only" in sys_msg["content"]


def test_xml_payload_fencing():
    builder = PromptBuilder()
    vision = VisionContext(
        subject_detected=True,
        pan_angle_deg=15.0,
        detected_objects=["coffee cup"],
    )
    payload = builder.format_user_payload("Hello there!", vision)

    assert "<context>" in payload
    assert "</context>" in payload
    assert "<vision>" in payload
    assert "coffee cup" in payload
    assert "<transcript>Hello there!</transcript>" in payload


def test_history_pruning_by_turn():
    config = DialogueConfig(max_history_turns=2)
    builder = PromptBuilder(config)

    history = [
        ChatMessage(role="user", content="Turn 1"),
        ChatMessage(role="assistant", content="Response 1"),
        ChatMessage(role="user", content="Turn 2"),
        ChatMessage(role="assistant", content="Response 2"),
        ChatMessage(role="user", content="Turn 3"),
        ChatMessage(role="assistant", content="Response 3"),
    ]

    pruned = builder.prune_history(history)
    assert len(pruned) == 2
    assert pruned[0].content == "Turn 3"
    assert pruned[1].content == "Response 3"


def test_history_pruning_by_tokens():
    # max_context_tokens = 10 tokens (~40 chars)
    config = DialogueConfig(max_history_turns=10, max_context_tokens=10)
    builder = PromptBuilder(config)

    history = [
        ChatMessage(role="user", content="Short 1"),
        ChatMessage(role="assistant", content="A" * 100),  # Exceeds 40 chars
        ChatMessage(role="user", content="Short 2"),
    ]

    pruned = builder.prune_history(history)
    # The massive 100-char message must be pruned out
    total_chars = sum(len(m.content) for m in pruned)
    assert total_chars <= 40
