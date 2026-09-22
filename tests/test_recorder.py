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


def test_mock_audio_recorder_record_with_vad():
    recorder = MockAudioRecorder(sample_rate=16000)
    speech_started = False

    def on_start():
        nonlocal speech_started
        speech_started = True

    wav_bytes = recorder.record_with_vad(
        silence_duration_s=0.45,
        max_recording_s=5.0,
        on_speech_start=on_start,
    )

    assert speech_started is True
    assert len(wav_bytes) > 0
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000


@pytest.mark.asyncio
async def test_mock_audio_recorder_record_with_vad_async():
    recorder = MockAudioRecorder(sample_rate=16000)
    wav_bytes = await recorder.record_with_vad_async(silence_duration_s=0.45)
    assert len(wav_bytes) > 0


def test_audio_recorder_record_with_vad_invalid_frame_duration():
    recorder = AudioRecorder(sample_rate=16000)
    with pytest.raises(ValueError, match="frame_duration_ms must be 10, 20, or 30"):
        recorder.record_with_vad(frame_duration_ms=25)


def test_audio_recorder_record_with_vad_streaming_mock():
    recorder = AudioRecorder(sample_rate=16000)

    # Frame is 30ms -> 480 samples -> 960 bytes
    sample_rate = 16000
    samples_per_frame = int(sample_rate * 30 / 1000)
    t = np.linspace(0, 0.03, samples_per_frame, endpoint=False)

    # Speech frame: 200 Hz tone
    speech_frame = (np.sin(2 * np.pi * 200 * t) * 15000).astype(np.int16).tobytes()
    # Silence frame: zeros
    silence_frame = np.zeros(samples_per_frame, dtype=np.int16).tobytes()

    # Sequence: 2 silence frames (pre-speech), 4 speech frames, 4 silence frames (trailing silence)
    frames_sequence = [silence_frame, silence_frame] + [speech_frame] * 4 + [silence_frame] * 5

    frame_idx = 0

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def read(self, num_samples):
            nonlocal frame_idx
            if frame_idx < len(frames_sequence):
                frame = frames_sequence[frame_idx]
                frame_idx += 1
                return frame, False
            return silence_frame, False

    speech_started_flag = False

    def on_speech():
        nonlocal speech_started_flag
        speech_started_flag = True

    with patch("sounddevice.RawInputStream", return_value=FakeStream()):
        wav_bytes = recorder.record_with_vad(
            silence_duration_s=0.1,  # 0.1s / 0.03s ≈ 3-4 frames triggers stop
            max_recording_s=5.0,
            frame_duration_ms=30,
            aggressiveness=2,
            on_speech_start=on_speech,
        )

    assert speech_started_flag is True
    assert len(wav_bytes) > 0
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000


def test_audio_recorder_record_with_vad_defaults():
    import inspect
    from skeleton.audio.config import AudioConfig

    cfg = AudioConfig()
    assert cfg.vad_silence_duration_s == 0.25
    assert cfg.vad_aggressiveness == 3

    sig = inspect.signature(AudioRecorder.record_with_vad)
    assert sig.parameters["silence_duration_s"].default == 0.25
    assert sig.parameters["aggressiveness"].default == 3

    mock_sig = inspect.signature(MockAudioRecorder.record_with_vad)
    assert mock_sig.parameters["silence_duration_s"].default == 0.25
    assert mock_sig.parameters["aggressiveness"].default == 3

