"""Tests for speech and mouth text sanitizer."""

import pytest
from skeleton.dialogue.sanitizer import TextSanitizer


def test_strip_stage_directions():
    raw = "*laughs dryly* Well, well, well. [whispering softly] Look at you."
    cleaned = TextSanitizer.sanitize(raw)
    assert cleaned == "Well, well, well. Look at you."


def test_strip_markdown():
    raw = "I **really** like your *shirt*, or `lack` thereof."
    cleaned = TextSanitizer.sanitize(raw)
    assert cleaned == "I really like your shirt, or lack thereof."


def test_strip_emojis():
    raw = "Nice hat! 💀🤣 Did you get it on sale?"
    cleaned = TextSanitizer.sanitize(raw)
    assert cleaned == "Nice hat! Did you get it on sale?"


def test_normalize_whitespace():
    raw = "   Too   many    spaces   and\nnewlines.   "
    cleaned = TextSanitizer.sanitize(raw)
    assert cleaned == "Too many spaces and newlines."


def test_empty_or_none():
    assert TextSanitizer.sanitize("") == ""
    assert TextSanitizer.sanitize(None) == ""
