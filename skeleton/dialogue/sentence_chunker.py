"""Streaming token chunker that parses LLM token deltas into coherent speech phrases."""

import re
import time
from typing import AsyncGenerator, Optional
from skeleton.dialogue.models import SpeechChunk
from skeleton.dialogue.sanitizer import TextSanitizer


class SentenceChunker:
    """Buffers incoming LLM streaming tokens, applies phonetic sanitization,

    and yields complete phrases/sentences for downstream TTS and mouth movement.
    """

    # Common abbreviations to protect from premature sentence splitting
    _ABBREVIATIONS = (
        "mr", "mrs", "ms", "dr", "prof", "vs", "etc", "e.g", "i.e", "sr", "jr"
    )

    # Terminal punctuation pattern: . ! ? followed by space or newline
    # Guarded by negative lookbehind for digits (decimals like 3.14)
    _TERMINAL_PUNCTUATION = re.compile(r"(?<!\d)[.!?](?=\s|$)")

    # Clause boundary punctuation: , ; : — -
    _CLAUSE_PUNCTUATION = re.compile(r"[,;:—](?=\s|$)")

    def __init__(self, min_clause_words: int = 5):
        """Args:

        min_clause_words: Minimum words required in the buffer before a clause
        punctuation (comma, semicolon) will trigger a split. This preserves natural
        TTS prosody.
        """
        self.min_clause_words = min_clause_words

    def _is_abbreviation(self, text_before_period: str) -> bool:
        """Checks if the word preceding a period is a known abbreviation."""
        tokens = text_before_period.strip().split()
        if not tokens:
            return False
        last_word = re.sub(r"[^\w]", "", tokens[-1]).lower()
        return last_word in self._ABBREVIATIONS

    def _find_split_index(self, buffer: str) -> Optional[int]:
        """Finds the character index to split the buffer, or None if no valid boundary is met."""
        # 1. Check for terminal punctuation (. ! ?)
        for match in self._TERMINAL_PUNCTUATION.finditer(buffer):
            end_idx = match.end()
            text_up_to_split = buffer[:end_idx]

            # Check if this period was part of an abbreviation (e.g. Dr. Bones)
            if match.group().startswith(".") and self._is_abbreviation(buffer[: match.start()]):
                continue

            return end_idx

        # 2. Check for clause boundaries (, ; : —) only if min word threshold is met
        words = buffer.split()
        if len(words) >= self.min_clause_words:
            for match in self._CLAUSE_PUNCTUATION.finditer(buffer):
                return match.end()

        return None

    async def stream_chunks(
        self,
        token_stream: AsyncGenerator[str, None],
        start_time: Optional[float] = None,
    ) -> AsyncGenerator[SpeechChunk, None]:
        """Consumes a stream of raw token deltas and yields clean SpeechChunk objects."""
        if start_time is None:
            start_time = time.time()

        buffer = ""
        sequence_index = 0

        async for token in token_stream:
            buffer += token

            while True:
                split_idx = self._find_split_index(buffer)
                if split_idx is None:
                    break

                raw_chunk = buffer[:split_idx]
                buffer = buffer[split_idx:].lstrip()

                clean_chunk = TextSanitizer.sanitize(raw_chunk)
                if clean_chunk:
                    elapsed_ms = (time.time() - start_time) * 1000.0
                    yield SpeechChunk(
                        text=clean_chunk,
                        sequence_index=sequence_index,
                        is_final=False,
                        elapsed_ms=round(elapsed_ms, 2),
                    )
                    sequence_index += 1

        # Process trailing text in buffer at end of stream
        clean_trailing = TextSanitizer.sanitize(buffer)
        if clean_trailing:
            # Ensure final sentence ends with proper punctuation
            if clean_trailing[-1] not in ".!?":
                clean_trailing += "."

            elapsed_ms = (time.time() - start_time) * 1000.0
            yield SpeechChunk(
                text=clean_trailing,
                sequence_index=sequence_index,
                is_final=True,
                elapsed_ms=round(elapsed_ms, 2),
            )
        elif sequence_index > 0:
            # If trailing buffer is empty but we emitted chunks, the last chunk is conceptually final
            pass
