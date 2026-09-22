"""Tests for the WebSocket duplex audio streaming and control interface."""

import io
import json
import math
import struct
import wave
import pytest
from fastapi.testclient import TestClient

from skeleton.audio.mouth_sync import MouthSyncProcessor
from skeleton.audio.pipeline import SkeletonAudioPipeline
from skeleton.audio.recorder import MockAudioRecorder
from skeleton.audio.stt import MockSTTClient
from skeleton.audio.tts import MockTTSClient
from skeleton.dialogue.llm_client import MockLLMClient
from skeleton.dialogue.orchestrator import DialogueOrchestrator
from skeleton.server.app import create_app


def _make_dummy_wav(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        num_samples = int(duration_s * sample_rate)
        samples = bytearray()
        for i in range(num_samples):
            val = int(10000 * math.sin(2 * math.pi * 440 * (i / sample_rate)))
            samples.extend(struct.pack("<h", val))
        wf.writeframes(samples)
    return buffer.getvalue()


def _build_test_app():
    mock_stt = MockSTTClient("Hello there skeleton!")
    mock_llm = MockLLMClient("What do you want, meatbag?")
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
    return create_app(custom_pipeline=pipeline)


def test_websocket_connect_and_ping():
    app = _build_test_app()
    client = TestClient(app)

    with client.websocket_connect("/ws/audio") as ws:
        # Initial greeting state
        init_event = ws.receive_json()
        assert init_event["event"] == "state"
        assert init_event["data"]["state"] == "ready"

        # Send ping
        ws.send_json({"type": "ping"})
        pong_event = ws.receive_json()
        assert pong_event["event"] == "pong"
        assert "timestamp" in pong_event["data"]


def test_websocket_audio_streaming_turn():
    app = _build_test_app()
    client = TestClient(app)
    dummy_wav = _make_dummy_wav()

    with client.websocket_connect("/ws/audio") as ws:
        # 1. State ready
        init_event = ws.receive_json()
        assert init_event["data"]["state"] == "ready"

        # 2. Start turn
        ws.send_json({"type": "start_turn", "format": "wav"})
        state_event = ws.receive_json()
        assert state_event["event"] == "state"
        assert state_event["data"]["state"] == "listening"

        # 3. Stream binary audio chunks
        chunk_size = 512
        for offset in range(0, len(dummy_wav), chunk_size):
            ws.send_bytes(dummy_wav[offset : offset + chunk_size])

        # 4. End turn
        ws.send_json({"type": "end_turn"})

        # Collect events until done
        events = []
        while True:
            evt = ws.receive_json()
            events.append(evt)
            if evt["event"] == "done":
                break

        event_names = [e["event"] for e in events]
        assert "transcript" in event_names
        assert "chunk" in event_names
        assert "done" in event_names

        transcript_evt = next(e for e in events if e["event"] == "transcript")
        assert transcript_evt["data"]["transcript"] == "Hello there skeleton!"

        chunk_evt = next(e for e in events if e["event"] == "chunk")
        assert "What do you want, meatbag?" in chunk_evt["data"]["text"]
        assert len(chunk_evt["data"]["audio_base64"]) > 100
        assert len(chunk_evt["data"]["mouth_frames"]) > 0


def test_websocket_barge_in():
    app = _build_test_app()
    client = TestClient(app)

    with client.websocket_connect("/ws/audio") as ws:
        _ = ws.receive_json()  # ready

        # Send barge-in command
        ws.send_json({"type": "barge_in"})

        barge_ack = ws.receive_json()
        assert barge_ack["event"] == "barge_in_ack"
        assert "cancelled" in barge_ack["data"]

        state_evt = ws.receive_json()
        assert state_evt["event"] == "state"
        assert state_evt["data"]["state"] == "interrupted"


def test_websocket_empty_audio_turn_rejected():
    app = _build_test_app()
    client = TestClient(app)

    with client.websocket_connect("/ws/audio") as ws:
        _ = ws.receive_json()  # ready

        ws.send_json({"type": "start_turn"})
        _ = ws.receive_json()  # listening

        # Send end_turn without sending audio
        ws.send_json({"type": "end_turn"})

        err_evt = ws.receive_json()
        assert err_evt["event"] == "error"
        assert "too small" in err_evt["data"]["error"]

        state_evt = ws.receive_json()
        assert state_evt["event"] == "state"
        assert state_evt["data"]["state"] == "idle"
