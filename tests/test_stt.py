"""Tests for Speech-to-Text client and boundary checks."""

import pytest
from skeleton.audio.config import AudioConfig
from skeleton.audio.stt import MockSTTClient, WhisperSTTClient


@pytest.mark.asyncio
async def test_whisper_buffer_bounds_validation():
    config = AudioConfig(min_audio_bytes=100, max_audio_bytes=1000)
    client = WhisperSTTClient(config)

    # 1. Payload too small (< 100 bytes)
    with pytest.raises(ValueError, match="Audio payload too small"):
        await client.transcribe(b"too short")

    # 2. Payload too large (> 1000 bytes)
    with pytest.raises(ValueError, match="Audio payload exceeded limit"):
        await client.transcribe(b"x" * 1500)

    await client.close()


@pytest.mark.asyncio
async def test_mock_stt_client():
    mock = MockSTTClient("Where is your funny bone?")
    result = await mock.transcribe(b"x" * 150)
    assert result == "Where is your funny bone?"

    with pytest.raises(ValueError, match="Audio payload too small"):
        await mock.transcribe(b"tiny")
