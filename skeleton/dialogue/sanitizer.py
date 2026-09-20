"""Phonetic and text sanitization for speech synthesis and mouth actuation."""

import re


class TextSanitizer:
    """Cleans raw LLM outputs to remove stage directions, markdown, asterisks,

    and emojis before text is sent to TTS or mouth actuators.
    """

    # Matches stage directions like [whispering], (clears throat), [laughs]
    _BRACKETED_STAGE_DIRECTION = re.compile(
        r"(\[.*?\])|(\(.*?\))", flags=re.DOTALL
    )

    # Matches asterisk stage directions like *laughs*, *sighs*, *chuckles dryly*, *cackles*
    _ASTERISK_STAGE_DIRECTION = re.compile(
        r"\*\s*(?:[^*]*?\b(?:cackle|laugh|chuckle|giggle|sigh|groan|snicker|whisper|cough|gasp|shrug|pause|wheeze|smirk|grin|wink|roll|look|clear|glares?|stares?|snorts?)[^*]*?)\*",
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Matches markdown formatting: bold/italic markers, headers, backticks, tildes
    _MARKDOWN_PATTERN = re.compile(r"[\*_~`#]")

    # Matches excessive whitespace (convert newlines/tabs to space first)
    _WHITESPACE_PATTERN = re.compile(r"\s+")

    # Matches emojis and non-standard symbols outside ASCII / Latin printable text
    _EMOJI_AND_SYMBOL_PATTERN = re.compile(
        r"[^\x20-\x7E\xA0-\xFF]", flags=re.UNICODE
    )

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Sanitizes text to ensure safe pronunciation by TTS engines and

        prevent mechanical lip-sync anomalies.
        """
        if not text:
            return ""

        # 1. First normalize newlines/tabs into spaces so whitespace is preserved
        cleaned = cls._WHITESPACE_PATTERN.sub(" ", text)

        # 2. Remove bracketed and parenthetical stage directions
        cleaned = cls._BRACKETED_STAGE_DIRECTION.sub("", cleaned)

        # 3. Remove asterisk action stage directions (e.g. *laughs dryly*)
        cleaned = cls._ASTERISK_STAGE_DIRECTION.sub("", cleaned)

        # 4. Strip leftover markdown artifacts (*emphasis*, `code`, **bold**)
        cleaned = cls._MARKDOWN_PATTERN.sub("", cleaned)

        # 5. Strip emojis and non-standard symbols
        cleaned = cls._EMOJI_AND_SYMBOL_PATTERN.sub("", cleaned)

        # 6. Final whitespace normalization and trimming
        cleaned = cls._WHITESPACE_PATTERN.sub(" ", cleaned).strip()

        return cleaned
