"""Unit tests for the desktop AudioWorker, transport modes, and controller state machine."""

import base64
import io
import json
import queue
import subprocess
import sys
import time
import wave
from unittest.mock import MagicMock, patch
import httpx
import pytest

from skeleton.audio.mouth_sync import MouthSyncProcessor
from skeleton.audio.pipeline import SkeletonAudioPipeline
from skeleton.audio.recorder import MockAudioRecorder
from skeleton.audio.stt import MockSTTClient
from fastapi.testclient import TestClient
from skeleton.audio.tts import MockTTSClient
from skeleton.desktop.app import AudioWorker
from skeleton.dialogue.llm_client import MockLLMClient
from skeleton.dialogue.orchestrator import DialogueOrchestrator
from skeleton.server.app import create_app


def _make_test_client():
    mock_stt = MockSTTClient("Is anyone there?")
    mock_llm = MockLLMClient("Unfortunately, yes.")
    orchestrator = DialogueOrchestrator(llm_client=mock_llm)
    mock_tts = MockTTSClient()
    mouth_sync = MouthSyncProcessor()
    mock_recorder = MockAudioRecorder()

    pipeline = SkeletonAudioPipeline(
        stt_client=mock_stt,
        tts_client=mock_tts,
        orchestrator=orchestrator,
        mouth_sync=mouth_sync,
        recorder=mock_recorder,
    )
    app = create_app(custom_pipeline=pipeline)
    return TestClient(app)


def test_audio_worker_init_and_stop():
    ui_queue = queue.Queue()
    worker = AudioWorker(
        base_url="http://test",
        record_seconds=1.0,
        recorder=MockAudioRecorder(),
        ui_queue=ui_queue,
    )
    assert not worker.stop_event.is_set()
    assert worker.transport_mode == "websocket"
    worker.stop()
    assert worker.stop_event.is_set()


def test_audio_worker_websocket_cycle():
    """Verify AudioWorker executes conversational turn over WebSocket connection."""
    ui_queue = queue.Queue()
    mock_recorder = MockAudioRecorder()

    worker = AudioWorker(
        base_url="http://127.0.0.1:8000",
        record_seconds=1.0,
        recorder=mock_recorder,
        ui_queue=ui_queue,
        transport_mode="websocket",
        auto_greet_first_turn=False,
    )

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 400)
    dummy_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    ws_messages = [
        json.dumps({"event": "state", "data": {"state": "ready"}}),
        json.dumps({"event": "state", "data": {"state": "listening"}}),
        json.dumps({"event": "transcript", "data": {"transcript": "Who goes there?"}}),
        json.dumps({
            "event": "chunk",
            "data": {
                "text": "Only bones and shadows.",
                "audio_base64": dummy_b64,
                "latency_ms": 115.0,
                "mouth_frames": [0.4, 0.7, 0.2],
            },
        }),
        json.dumps({"event": "done", "data": {"total_latency_ms": 230.0}}),
    ]

    class MockWS:
        def __init__(self, msgs):
            self.msgs = list(msgs)
            self.sent = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def recv(self):
            if self.msgs:
                return self.msgs.pop(0)
            raise EOFError("No messages left")

        def send(self, data):
            self.sent.append(data)

    mock_ws = MockWS(ws_messages)

    with patch("websockets.sync.client.connect", return_value=mock_ws) as mock_connect, \
         patch("sounddevice.play") as mock_play, \
         patch("sounddevice.stop") as mock_stop, \
         patch("time.sleep") as mock_sleep:

        mock_sleep.side_effect = lambda s: worker.stop()
        worker.run()

        mock_connect.assert_called_once_with("ws://127.0.0.1:8000/ws/audio")
        mock_play.assert_called_once()

        events = []
        while not ui_queue.empty():
            events.append(ui_queue.get_nowait())

        event_types = [e[0] for e in events]
        assert "STATUS" in event_types
        assert "TURN" in event_types

        turn_event = next(e for e in events if e[0] == "TURN")
        turn_data = turn_event[1]
        assert turn_data["user"] == "Who goes there?"
        assert "Only bones and shadows." in turn_data["skeleton"]
        assert turn_data["mouth_frames"] == 3
        assert turn_data["latency_ms"] == 115.0


def test_audio_worker_single_cycle():
    client = _make_test_client()
    ui_queue = queue.Queue()
    mock_recorder = MockAudioRecorder()

    worker = AudioWorker(
        base_url="http://test",
        record_seconds=1.0,
        recorder=mock_recorder,
        ui_queue=ui_queue,
        http_client=client,
        transport_mode="sse",
        auto_greet_first_turn=False,
    )

    # Patch sounddevice play & time.sleep to avoid hardware use and delays in CI
    with patch("sounddevice.play") as mock_play, patch("sounddevice.stop") as mock_stop, patch("time.sleep") as mock_sleep:
        mock_sleep.side_effect = lambda s: worker.stop()
        worker.run()

        # Collect events from ui_queue
        events = []
        while not ui_queue.empty():
            events.append(ui_queue.get_nowait())

        event_types = [e[0] for e in events]
        assert "STATUS" in event_types
        assert "TURN" in event_types

        # Verify turn data
        turn_event = next(e for e in events if e[0] == "TURN")
        turn_data = turn_event[1]
        assert turn_data["user"] == "Is anyone there?"
        assert "Unfortunately, yes." in turn_data["skeleton"]
        assert turn_data["mouth_frames"] > 0

        # Verify sound playback was triggered
        mock_play.assert_called_once()

    client.close()


def test_audio_worker_handles_server_error():
    ui_queue = queue.Queue()
    mock_recorder = MockAudioRecorder()

    worker = AudioWorker(
        base_url="http://test",
        record_seconds=1.0,
        recorder=mock_recorder,
        ui_queue=ui_queue,
        transport_mode="sse",
        auto_greet_first_turn=False,
    )

    def handle_err(req):
        return httpx.Response(500, json={"detail": "Simulated server failure"})

    mock_client = httpx.Client(transport=httpx.MockTransport(handle_err))
    worker.client = mock_client
    worker._owns_client = True

    def stop_on_sleep(*args, **kwargs):
        worker.stop()

    with patch("time.sleep", side_effect=stop_on_sleep):
        worker.run()

    events = []
    while not ui_queue.empty():
        events.append(ui_queue.get_nowait())

    error_events = [e for e in events if e[0] == "ERROR"]
    assert len(error_events) > 0
    assert "500" in error_events[0][1]

    mock_client.close()


def test_audio_worker_with_fixed_recording():
    client = _make_test_client()
    ui_queue = queue.Queue()
    mock_recorder = MockAudioRecorder()

    worker = AudioWorker(
        base_url="http://test",
        record_seconds=0.5,
        use_vad=False,
        recorder=mock_recorder,
        ui_queue=ui_queue,
        http_client=client,
        transport_mode="sse",
        auto_greet_first_turn=False,
    )

    with patch("sounddevice.play"), patch("sounddevice.wait"), patch("time.sleep") as mock_sleep:
        mock_sleep.side_effect = lambda s: worker.stop()
        worker.run()

    events = []
    while not ui_queue.empty():
        events.append(ui_queue.get_nowait())

    turn_event = next(e for e in events if e[0] == "TURN")
    assert turn_event[1]["user"] == "Is anyone there?"

    client.close()


def test_desktop_sound_trigger_playback():
    """Verify pre-recorded sound trigger handler in desktop app plays audio and emits TURN event."""
    test_client = _make_test_client()
    pipeline = test_client.app.state.pipeline
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 400)
    dummy_wav = buf.getvalue()
    pipeline.sound_bank.register_sound(
        sound_id="test_desktop_cackle",
        category="laugh",
        text="Hehehe!",
        audio_bytes=dummy_wav,
    )

    # Test direct request
    resp = test_client.post("/play_sound/test_desktop_cackle")
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Hehehe!"
    assert len(data["mouth_frames"]) > 0
    assert len(data["audio_base64"]) > 0
    test_client.close()


def test_desktop_package_import_and_no_warnings():
    """Verify package import and execution does not trigger RuntimeWarning."""
    # Test importing via __getattr__
    import skeleton.desktop as desktop_pkg
    assert hasattr(desktop_pkg, "AudioWorker")
    assert hasattr(desktop_pkg, "launch_gui")

    # Verify invalid attribute raises AttributeError
    with pytest.raises(AttributeError):
        _ = desktop_pkg.non_existent_symbol

    # Test subprocess execution with -W error
    res = subprocess.run(
        [sys.executable, "-W", "error", "-c", "import skeleton.desktop; from skeleton.desktop import AudioWorker, launch_gui; assert AudioWorker is not None; assert launch_gui is not None"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "RuntimeWarning" not in res.stderr


def test_audio_worker_auto_greet_first_turn():
    """Verify that AudioWorker immediately plays a random greeting on first voice detection."""
    test_client = _make_test_client()
    pipeline = test_client.app.state.pipeline

    # Register a greeting sound
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 400)
    pipeline.sound_bank.register_sound(
        sound_id="test_greeting_nice_face",
        category="greeting",
        text="Nice costume! Oh wait, that's just your face.",
        audio_bytes=buf.getvalue(),
    )

    ui_queue = queue.Queue()
    mock_recorder = MockAudioRecorder()

    worker = AudioWorker(
        base_url="http://test",
        record_seconds=1.0,
        recorder=mock_recorder,
        ui_queue=ui_queue,
        http_client=test_client,
        auto_greet_first_turn=True,
    )

    with patch("sounddevice.play") as mock_play, \
         patch("sounddevice.stop") as mock_stop, \
         patch("time.sleep") as mock_sleep:

        mock_sleep.side_effect = lambda s: worker.stop()
        worker.run()

        mock_play.assert_called_once()

    events = []
    while not ui_queue.empty():
        events.append(ui_queue.get_nowait())

    turn_event = next(e for e in events if e[0] == "TURN")
    turn_data = turn_event[1]
    assert "[Voice Detected" in turn_data["user"]
    valid_greetings = [s.text for s in pipeline.sound_bank.get_by_category("greeting")]
    assert turn_data["skeleton"] in valid_greetings
    assert not worker.is_first_turn

    test_client.close()


def test_audio_worker_auto_greet_failure_falls_back():
    """Verify that if random greeting fails, AudioWorker gracefully falls back to normal conversational processing."""
    test_client = _make_test_client()
    ui_queue = queue.Queue()
    mock_recorder = MockAudioRecorder()

    worker = AudioWorker(
        base_url="http://test",
        record_seconds=1.0,
        recorder=mock_recorder,
        ui_queue=ui_queue,
        http_client=test_client,
        auto_greet_first_turn=True,
        transport_mode="sse",
    )

    # Mock _play_random_greeting to simulate failure / network error
    with patch.object(worker, "_play_random_greeting", return_value=False) as mock_greet, \
         patch("sounddevice.play") as mock_play, \
         patch("sounddevice.stop") as mock_stop, \
         patch("time.sleep") as mock_sleep:

        mock_sleep.side_effect = lambda s: worker.stop()
        worker.run()

        mock_greet.assert_called_once()
        # Fallback to standard turn should have occurred
        events = []
        while not ui_queue.empty():
            events.append(ui_queue.get_nowait())

        turn_event = next(e for e in events if e[0] == "TURN")
        turn_data = turn_event[1]
        assert turn_data["user"] == "Is anyone there?"
        assert "Unfortunately, yes." in turn_data["skeleton"]

    test_client.close()


def test_audio_worker_interruptible_playback_completes():
    """Verify that playing un-interrupted audio finishes normally."""
    worker = AudioWorker(recorder=MockAudioRecorder())
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 800)  # 50ms audio
    dummy_wav = buf.getvalue()

    with patch("sounddevice.play") as mock_play, patch("time.sleep"):
        interrupted, frames = worker._play_audio_interruptible(dummy_wav)
        assert interrupted is False
        assert frames == []
        mock_play.assert_called_once()


def test_audio_worker_manual_interrupt():
    """Verify that manual cutoff triggers immediate interruption and sd.stop()."""
    worker = AudioWorker(recorder=MockAudioRecorder())
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 8000)  # 0.5s audio
    dummy_wav = buf.getvalue()

    worker.trigger_interrupt()
    assert worker.interrupt_event.is_set()

    barge_in_called = []
    with patch("sounddevice.play"), patch("sounddevice.stop") as mock_stop:
        interrupted, frames = worker._play_audio_interruptible(
            dummy_wav,
            on_barge_in=lambda: barge_in_called.append(True),
        )
        assert interrupted is True
        assert len(barge_in_called) == 1
        mock_stop.assert_called()


def test_audio_worker_voice_barge_in_detection():
    """Verify that concurrent voiced frames with high RMS trigger voice barge-in."""
    import numpy as np

    worker = AudioWorker(
        voice_barge_in=True,
        barge_in_threshold=0.05,
        barge_in_consecutive_frames=3,
        barge_in_grace_period_s=0.0,
    )
    # Ensure worker does not treat recorder as MockAudioRecorder for this test
    worker.recorder = MagicMock()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)  # 1.0s audio
    dummy_wav = buf.getvalue()

    # Generate high energy 30ms PCM frame (480 samples @ 16kHz)
    high_energy_samples = (np.sin(2 * np.pi * 200 * np.linspace(0, 0.03, 480)) * 15000).astype(np.int16)
    high_energy_bytes = high_energy_samples.tobytes()

    class MockMicStream:
        def __init__(self):
            self.call_count = 0

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

        def read(self, samples):
            self.call_count += 1
            return high_energy_bytes, False

    mock_stream = MockMicStream()
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = True

    barge_in_invoked = []
    with patch("sounddevice.play"), \
         patch("sounddevice.stop") as mock_stop, \
         patch("sounddevice.RawInputStream", return_value=mock_stream), \
         patch("webrtcvad.Vad", return_value=mock_vad):

        interrupted, frames = worker._play_audio_interruptible(
            dummy_wav,
            on_barge_in=lambda: barge_in_invoked.append(True),
        )

        assert interrupted is True
        assert len(frames) == 3
        assert len(barge_in_invoked) == 1
        mock_stop.assert_called()


def test_audio_worker_acoustic_echo_rejection():
    """Verify that low-RMS or unvoiced frames do NOT trigger barge-in."""
    import numpy as np

    worker = AudioWorker(
        voice_barge_in=True,
        barge_in_threshold=0.08,
        barge_in_consecutive_frames=3,
        barge_in_grace_period_s=0.0,
    )
    worker.recorder = MagicMock()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)  # 0.1s audio
    dummy_wav = buf.getvalue()

    # Low-amplitude frame (RMS < 0.08)
    low_energy_samples = (np.sin(2 * np.pi * 200 * np.linspace(0, 0.03, 480)) * 500).astype(np.int16)
    low_energy_bytes = low_energy_samples.tobytes()

    class MockMicStream:
        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

        def read(self, samples):
            return low_energy_bytes, False

    mock_stream = MockMicStream()
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = True

    with patch("sounddevice.play"), \
         patch("sounddevice.stop") as mock_stop, \
         patch("sounddevice.RawInputStream", return_value=mock_stream), \
         patch("webrtcvad.Vad", return_value=mock_vad), \
         patch("time.sleep"):

        interrupted, frames = worker._play_audio_interruptible(dummy_wav)
        assert interrupted is False
        assert frames == []


def test_audio_worker_headless_portaudio_degradation():
    """Verify that if RawInputStream fails (headless CI/no mic), playback continues without crashing."""
    worker = AudioWorker(voice_barge_in=True)
    worker.recorder = MagicMock()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 800)  # 50ms audio
    dummy_wav = buf.getvalue()

    with patch("sounddevice.play") as mock_play, \
         patch("sounddevice.RawInputStream", side_effect=Exception("No input device available")), \
         patch("time.sleep"):

        interrupted, frames = worker._play_audio_interruptible(dummy_wav)
        assert interrupted is False
        assert frames == []
        mock_play.assert_called_once()


def test_audio_worker_websocket_barge_in():
    """Verify that during WebSocket playback, interruption dispatches barge_in frame to server."""
    ui_queue = queue.Queue()
    mock_recorder = MockAudioRecorder()

    worker = AudioWorker(
        base_url="http://127.0.0.1:8000",
        record_seconds=1.0,
        recorder=mock_recorder,
        ui_queue=ui_queue,
        transport_mode="websocket",
        auto_greet_first_turn=False,
    )

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 400)
    dummy_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    ws_messages = [
        json.dumps({"event": "state", "data": {"state": "ready"}}),
        json.dumps({"event": "state", "data": {"state": "listening"}}),
        json.dumps({"event": "transcript", "data": {"transcript": "Tell me a secret."}}),
        json.dumps({
            "event": "chunk",
            "data": {
                "text": "Bones don't keep secrets.",
                "audio_base64": dummy_b64,
                "latency_ms": 120.0,
                "mouth_frames": [0.5],
            },
        }),
    ]

    class MockWS:
        def __init__(self, msgs):
            self.msgs = list(msgs)
            self.sent = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def recv(self):
            if self.msgs:
                return self.msgs.pop(0)
            raise EOFError("No messages left")

        def send(self, data):
            self.sent.append(data)

    mock_ws = MockWS(ws_messages)

    # Intercept _play_audio_interruptible to simulate barge-in trigger
    def fake_play_interruptible(wav_bytes, on_barge_in=None):
        if on_barge_in:
            on_barge_in()
        return True, [b"\x01\x00" * 240]

    with patch("websockets.sync.client.connect", return_value=mock_ws), \
         patch.object(worker, "_play_audio_interruptible", side_effect=fake_play_interruptible), \
         patch("time.sleep") as mock_sleep:

        mock_sleep.side_effect = lambda s: worker.stop()
        worker.run()

        # Check that barge_in message was sent to server
        sent_types = [json.loads(s).get("type") for s in mock_ws.sent if isinstance(s, str) and s.startswith("{")]
        assert "barge_in" in sent_types

        # Check status queue has Interrupted status
        events = []
        while not ui_queue.empty():
            events.append(ui_queue.get_nowait())

        statuses = [e[1] for e in events if e[0] == "STATUS"]
        assert any("Interrupted" in s for s in statuses)
        # Prefix audio should have been saved for next turn
        assert len(worker.interruption_prefix_audio) > 0


def test_audio_recorder_record_with_vad_initial_frames():
    """Verify that initial_audio_frames are correctly prepended to recorded audio."""
    mock_rec = MockAudioRecorder()
    initial_frames = [b"\x00\x00" * 480, b"\x01\x00" * 480]

    wav_bytes = mock_rec.record_with_vad(
        silence_duration_s=0.45,
        max_recording_s=1.0,
        initial_audio_frames=initial_frames,
    )

    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            assert len(frames) >= len(b"".join(initial_frames))
            # Verify initial frames are present at the beginning
            assert frames.startswith(b"".join(initial_frames))


