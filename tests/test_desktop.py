"""Unit tests for the desktop AudioWorker and controller state machine."""

import io
import queue
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
    worker.stop()
    assert worker.stop_event.is_set()


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
    )

    # Patch sounddevice play & wait and time.sleep to avoid hardware use and delays in CI
    with patch("sounddevice.play") as mock_play, patch("sounddevice.wait") as mock_wait, patch("time.sleep"):
        # After audio playback finishes, stop worker so it exits cleanly
        def wait_and_stop():
            worker.stop()

        mock_wait.side_effect = wait_and_stop

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

        # Verify sound playback was triggered and waited
        mock_play.assert_called_once()
        mock_wait.assert_called_once()

    client.close()


def test_audio_worker_handles_server_error():
    ui_queue = queue.Queue()
    mock_recorder = MockAudioRecorder()

    worker = AudioWorker(
        base_url="http://test",
        record_seconds=1.0,
        recorder=mock_recorder,
        ui_queue=ui_queue,
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
