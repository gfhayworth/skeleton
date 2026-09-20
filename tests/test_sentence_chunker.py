"""Tests for streaming sentence and phrase chunker."""

import pytest
from skeleton.dialogue.sentence_chunker import SentenceChunker


async def async_token_generator(tokens):
    for tok in tokens:
        yield tok


@pytest.mark.asyncio
async def test_chunker_sentence_splitting():
    chunker = SentenceChunker(min_clause_words=5)
    tokens = ["Hello ", "there. ", "I ", "am ", "a ", "skeleton! ", "Are ", "you ", "lost?"]
    chunks = [c async for c in chunker.stream_chunks(async_token_generator(tokens))]

    assert len(chunks) == 3
    assert chunks[0].text == "Hello there."
    assert chunks[1].text == "I am a skeleton!"
    assert chunks[2].text == "Are you lost?"


@pytest.mark.asyncio
async def test_chunker_abbreviation_protection():
    chunker = SentenceChunker(min_clause_words=5)
    tokens = ["Call ", "me ", "Dr. ", "Bones ", "if ", "you ", "please."]
    chunks = [c async for c in chunker.stream_chunks(async_token_generator(tokens))]

    # Should not split at "Dr."
    assert len(chunks) == 1
    assert chunks[0].text == "Call me Dr. Bones if you please."


@pytest.mark.asyncio
async def test_chunker_decimal_protection():
    chunker = SentenceChunker(min_clause_words=5)
    tokens = ["That ", "costs ", "3.14 ", "dollars ", "today."]
    chunks = [c async for c in chunker.stream_chunks(async_token_generator(tokens))]

    # Should not split at 3.
    assert len(chunks) == 1
    assert chunks[0].text == "That costs 3.14 dollars today."


@pytest.mark.asyncio
async def test_chunker_clause_word_threshold():
    chunker = SentenceChunker(min_clause_words=5)

    # Short clause (< 5 words before comma) should NOT split
    short_clause_tokens = ["Well, ", "I ", "see."]
    short_chunks = [c async for c in chunker.stream_chunks(async_token_generator(short_clause_tokens))]
    assert len(short_chunks) == 1
    assert short_chunks[0].text == "Well, I see."

    # Long clause (>= 5 words before comma) SHOULD split for early TTS emission
    long_clause_tokens = ["If you think that is clever, ", "you have another thing coming."]
    long_chunks = [c async for c in chunker.stream_chunks(async_token_generator(long_clause_tokens))]
    assert len(long_chunks) == 2
    assert long_chunks[0].text == "If you think that is clever,"
    assert long_chunks[1].text == "you have another thing coming."


@pytest.mark.asyncio
async def test_chunker_trailing_buffer_completion():
    chunker = SentenceChunker()
    tokens = ["No ", "final ", "punctuation ", "here"]
    chunks = [c async for c in chunker.stream_chunks(async_token_generator(tokens))]

    assert len(chunks) == 1
    assert chunks[0].text == "No final punctuation here."
    assert chunks[0].is_final is True
