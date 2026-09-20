"""Unit and integration tests for the AudioRecorder and record_and_process pipeline."""

import io
import wave
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from skeleton.audio.mouth_sync import MouthSyncProcessor
from skeleton.audio.pipeline import SkeletonAudioPipeline
from skeleton.audio.recorder import (
    AudioDevice,
    AudioDeviceError,
    AudioRecorder,
    MockAudioRecorder,
)
from skeleton.audio.stt import MockSTTClient
from skeleton.audio.tts import MockTTSClient
from skeleton.dialogue.llm_client import MockLLMClient
from skeleton.dialogue.orchestrator import DialogueOrchestrator


def test_mock_audio_recorder_list_devices():
    recorder = MockAudioRecorder()
    devices = recorder.list_input_devices()
    assert len(devices) == 1
    dev = devices[0]
    assert isinstance(dev, AudioDevice)
    assert dev.index == 0
    assert dev.name == "Mock Microphone"
    assert dev.max_input_channels == 1
    assert dev.default_samplerate == 16000.0


def test_mock_audio_recorder_record_wav_validity():
    recorder = MockAudioRecorder(sample_rate=16000)
    duration = 1.0
    wav_bytes = recorder.record(duration_seconds=duration)

    assert len(wav_bytes) > 0
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 16000


def test_mock_audio_recorder_duration_bounds():
    recorder = MockAudioRecorder()
    with pytest.raises(ValueError, match="too short"):
        recorder.record(duration_seconds=0.2)

    with pytest.raises(ValueError, match="too long"):
        recorder.record(duration_seconds=31.0)


@pytest.mark.asyncio
async def test_mock_audio_recorder_async():
    recorder = MockAudioRecorder(sample_rate=16000)
    wav_bytes = await recorder.record_async(duration_seconds=0.5)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnframes() == 8000


def test_audio_recorder_bounds():
    recorder = AudioRecorder(sample_rate=16000)
    with pytest.raises(ValueError, match=">= 0.5s"):
        recorder.record(duration_seconds=0.1)

    with pytest.raises(ValueError, match="<= 30.0s"):
        recorder.record(duration_seconds=35.0)


def test_audio_recorder_list_devices_mocked():
    fake_devices = [
        {"name": "Realtek Speakers", "max_input_channels": 0, "default_samplerate": 48000.0},
        {"name": "USB Condenser Mic", "max_input_channels": 1, "default_samplerate": 44100.0},
        {"name": "Logitech HD Webcam Mic", "max_input_channels": 2, "default_samplerate": 16000.0},
    ]

    with patch("sounddevice.query_devices", return_value=fake_devices):
        recorder = AudioRecorder()
        devices = recorder.list_input_devices()

        # Should filter out output-only device (index 0)
        assert len(devices) == 2
        assert devices[0].index == 1
        assert devices[0].name == "USB Condenser Mic"
        assert devices[0].max_input_channels == 1
        assert devices[1].index == 2
        assert devices[1].name == "Logitech HD Webcam Mic"
        assert devices[1].max_input_channels == 2


def test_audio_recorder_list_devices_error():
    with patch("sounddevice.query_devices", side_effect=RuntimeError("PortAudio uninitialized")):
        recorder = AudioRecorder()
        with pytest.raises(AudioDeviceError, match="Failed to query audio input devices"):
            recorder.list_input_devices()


def test_audio_recorder_record_mocked_sounddevice():
    recorder = AudioRecorder(sample_rate=16000)
    duration = 1.0
    num_frames = 16000
    dummy_audio_int16 = np.zeros((num_frames, 1), dtype=np.int16)

    with patch("sounddevice.rec", return_value=dummy_audio_int16) as mock_rec, \
         patch("sounddevice.wait") as mock_wait:
        wav_bytes = recorder.record(duration_seconds=duration, device_index=2)

        mock_rec.assert_called_once_with(
            num_frames,
            samplerate=16000,
            channels=1,
            dtype="int16",
            device=2,
        )
        mock_wait.assert_called_once()

        # Validate resulting WAV
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000


def test_audio_recorder_record_error_handling():
    recorder = AudioRecorder(sample_rate=16000)
    with patch("sounddevice.rec", side_effect=Exception("Device disconnected")):
        with pytest.raises(AudioDeviceError, match="Microphone recording failed"):
            recorder.record(duration_seconds=1.0, device_index=1)


@pytest.mark.asyncio
async def test_pipeline_record_and_process():
    mock_recorder = MockAudioRecorder(sample_rate=16000)
    mock_stt = MockSTTClient("Can you tell me where you hid the skull?")
    mock_llm = MockLLMClient("It's right where my head used to be, genius.")
    orchestrator = DialogueOrchestrator(llm_client=mock_llm)
    mock_tts = MockTTSClient()
    mouth_sync = MouthSyncProcessor()

    pipeline = SkeletonAudioPipeline(
        stt_client=mock_stt,
        tts_client=mock_tts,
        orchestrator=orchestrator,
        mouth_sync=mouth_sync,
        recorder=mock_recorder,
    )

    response = await pipeline.record_and_process(duration_seconds=1.0)

    assert response.user_transcript == "Can you tell me where you hid the skull?"
    assert "It's right where my head used to be" in response.skeleton_response
    assert len(response.audio_bytes) > 500
    assert len(response.mouth_frames) > 0
    assert response.total_latency_ms > 0

    await pipeline.close()
