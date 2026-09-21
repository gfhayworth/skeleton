"""Unit tests for pre-recorded sounds, SoundBank, and sidecar metadata cache."""

import json
import pytest
from pathlib import Path
from skeleton.audio.config import AudioConfig
from skeleton.audio.mouth_sync import MouthSyncProcessor
from skeleton.audio.sounds import (
    DEFAULT_SOUNDS_DIR,
    PreRecordedSound,
    SoundBank,
    SoundCategory,
)
from skeleton.audio.pipeline import SkeletonAudioPipeline
from skeleton.audio.recorder import MockAudioRecorder
from skeleton.audio.stt import MockSTTClient
from skeleton.audio.tts import MockTTSClient
from skeleton.dialogue.llm_client import MockLLMClient
from skeleton.dialogue.orchestrator import DialogueOrchestrator


def test_sound_bank_register_and_retrieve(tmp_path: Path):
    config = AudioConfig()
    mouth_sync = MouthSyncProcessor(config)
    bank = SoundBank(sounds_dir=tmp_path, config=config, mouth_sync=mouth_sync, autoload=False)

    # Generate dummy WAV bytes (mock TTS creates 16kHz sine)
    tts = MockTTSClient()
    import asyncio
    audio_bytes = asyncio.run(tts.synthesize("Ha ha ha!"))

    sound = bank.register_sound(
        sound_id="test_laugh",
        category=SoundCategory.LAUGH,
        text="Ha ha ha!",
        audio_bytes=audio_bytes,
    )

    assert sound.sound_id == "test_laugh"
    assert sound.category == SoundCategory.LAUGH
    assert len(sound.mouth_frames) > 0
    assert sound.duration_s > 0

    # Retrieve by ID
    assert bank.get("test_laugh") is sound
    assert bank.get("nonexistent") is None

    # Retrieve by Category
    laughs = bank.get_by_category(SoundCategory.LAUGH)
    assert len(laughs) == 1
    assert laughs[0].sound_id == "test_laugh"

    fillers = bank.get_by_category(SoundCategory.FILLER)
    assert len(fillers) == 0

    # Save to disk and re-load
    bank.save_sound(sound)
    assert (tmp_path / "test_laugh.wav").exists()
    assert (tmp_path / "test_laugh.json").exists()

    bank2 = SoundBank(sounds_dir=tmp_path, config=config, mouth_sync=mouth_sync, autoload=True)
    assert "test_laugh" in bank2.sounds
    loaded = bank2.get("test_laugh")
    assert loaded is not None
    assert loaded.text == "Ha ha ha!"
    assert loaded.category == SoundCategory.LAUGH
    assert len(loaded.mouth_frames) == len(sound.mouth_frames)


def test_sound_bank_loads_baked_assets():
    """Verify that the assets/sounds baked library loads correctly with valid metadata."""
    if not DEFAULT_SOUNDS_DIR.exists():
        pytest.skip("Default sounds directory not present in CI environment.")

    bank = SoundBank(sounds_dir=DEFAULT_SOUNDS_DIR, autoload=True)
    assert len(bank.sounds) >= 11

    # Check key categories
    assert len(bank.get_by_category(SoundCategory.FILLER)) >= 3
    assert len(bank.get_by_category(SoundCategory.LAUGH)) >= 2
    assert len(bank.get_by_category(SoundCategory.GREETING)) >= 2
    assert len(bank.get_by_category(SoundCategory.WAKE)) >= 2
    assert len(bank.get_by_category(SoundCategory.CONFUSED)) >= 2

    # Verify mouth frames are populated and valid
    for sound in bank.sounds.values():
        assert len(sound.mouth_frames) > 0
        assert sound.duration_s > 0
        assert len(sound.audio_bytes) > 1000
        # Check that jaw angles stay within bounds
        for frame in sound.mouth_frames:
            assert 0.0 <= frame.jaw_angle_deg <= 35.0
            assert 0.0 <= frame.normalized_open <= 1.0


def test_pipeline_prerecorded_sound_instant_playback():
    """Test instantaneous pre-recorded playback through SkeletonAudioPipeline without network overhead."""
    mock_stt = MockSTTClient("hello")
    mock_llm = MockLLMClient("hi")
    orchestrator = DialogueOrchestrator(llm_client=mock_llm)
    mock_tts = MockTTSClient()
    mock_recorder = MockAudioRecorder()

    pipeline = SkeletonAudioPipeline(
        stt_client=mock_stt,
        tts_client=mock_tts,
        orchestrator=orchestrator,
        recorder=mock_recorder,
    )

    resp = pipeline.play_prerecorded_sound(category=SoundCategory.LAUGH)
    assert resp is not None
    assert len(resp.audio_bytes) > 0
    assert len(resp.mouth_frames) > 0
    # Latency should be sub-millisecond (cache lookup)
    assert resp.total_latency_ms < 20.0
