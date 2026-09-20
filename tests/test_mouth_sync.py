"""Tests for mechanical mouth sync energy extraction and servo noise gating."""

import io
import math
import struct
import wave
import pytest
from skeleton.audio.config import AudioConfig
from skeleton.audio.mouth_sync import MouthSyncProcessor


def create_test_wav(duration_s=1.0, amplitude=15000, sample_rate=24000) -> bytes:
    """Helper creating test WAV audio."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        num_samples = int(sample_rate * duration_s)
        frames = bytearray()
        for i in range(num_samples):
            val = int(amplitude * math.sin(2 * math.pi * 200 * (i / sample_rate)))
            frames.extend(struct.pack("<h", val))
        w.writeframes(frames)
    return buffer.getvalue()


def test_mouth_sync_loud_audio():
    config = AudioConfig(mouth_fps=30, max_jaw_angle_deg=35.0, noise_floor_rms=0.04)
    processor = MouthSyncProcessor(config)

    wav_data = create_test_wav(duration_s=0.5, amplitude=16000)
    frames = processor.extract_jaw_trajectory(wav_data)

    assert len(frames) >= 14
    # Loud audio should open jaw
    assert any(f.jaw_angle_deg > 5.0 for f in frames)
    assert all(0.0 <= f.jaw_angle_deg <= 35.0 for f in frames)
    assert all(0.0 <= f.normalized_open <= 1.0 for f in frames)


def test_mouth_sync_deadband_noise_gate():
    config = AudioConfig(mouth_fps=30, max_jaw_angle_deg=35.0, noise_floor_rms=0.04)
    processor = MouthSyncProcessor(config)

    # Low amplitude noise (amplitude 300 / 32768 = ~0.009 RMS, well below noise floor 0.04)
    silent_wav = create_test_wav(duration_s=0.2, amplitude=300)
    frames = processor.extract_jaw_trajectory(silent_wav)

    # Noise gate must snap jaw angle strictly to 0.0
    assert all(f.jaw_angle_deg == 0.0 for f in frames)
    assert all(f.normalized_open == 0.0 for f in frames)


def test_mouth_sync_time_offset_continuity():
    processor = MouthSyncProcessor()
    wav_data = create_test_wav(duration_s=0.1)

    # Clause 1 with offset 0.0
    frames1 = processor.extract_jaw_trajectory(wav_data, time_offset_ms=0.0)
    # Clause 2 with offset 1000.0
    frames2 = processor.extract_jaw_trajectory(wav_data, time_offset_ms=1000.0)

    assert frames1[0].timestamp_ms == 0.0
    assert frames2[0].timestamp_ms == 1000.0
    assert frames2[-1].timestamp_ms > 1000.0
