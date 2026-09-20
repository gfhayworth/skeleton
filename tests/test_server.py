"""Unit tests for the FastAPI audio endpoints (/health, /audio_in, and /audio_out)."""

import io
import wave
import pytest
from httpx import ASGITransport, AsyncClient

from skeleton.audio.mouth_sync import MouthSyncProcessor
from skeleton.audio.pipeline import SkeletonAudioPipeline
from skeleton.audio.recorder import MockAudioRecorder
from skeleton.audio.stt import MockSTTClient
from skeleton.audio.tts import MockTTSClient
from skeleton.dialogue.llm_client import MockLLMClient
from skeleton.dialogue.orchestrator import DialogueOrchestrator
from skeleton.server.app import create_app


def _build_test_app():
    mock_stt = MockSTTClient("Hello there, skeleton!")
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


def _make_dummy_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 800)  # 800 samples = 1600 bytes
    return buf.getvalue()


@pytest.mark.asyncio
async def test_health_endpoint():
    app = _build_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "gemini" in data["gemini_model"]
        assert data["tts_voice"] == "onyx"
        assert data["sample_rate"] == 16000
        assert data["mouth_fps"] == 30


@pytest.mark.asyncio
async def test_audio_in_success():
    app = _build_test_app()
    dummy_wav = _make_dummy_wav()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("user_voice.wav", dummy_wav, "audio/wav")}
        resp = await client.post("/audio_in", files=files)

        assert resp.status_code == 200
        data = resp.json()
        assert data["user_transcript"] == "Hello there, skeleton!"
        assert "What do you want, meatbag?" in data["skeleton_response"]
        assert len(data["audio_base64"]) > 100
        assert len(data["mouth_frames"]) > 0
        assert data["total_latency_ms"] > 0


@pytest.mark.asyncio
async def test_audio_in_empty_payload_rejected():
    app = _build_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("empty.wav", b"", "audio/wav")}
        resp = await client.post("/audio_in", files=files)
        assert resp.status_code == 400
        assert "too small" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_audio_out_json_format():
    app = _build_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "text": "I am a talking skeleton.",
            "voice": "onyx",
            "speed": 0.90,
            "format": "json",
        }
        resp = await client.post("/audio_out", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "I am a talking skeleton."
        assert len(data["audio_base64"]) > 100
        assert len(data["mouth_frames"]) > 0
        assert data["latency_ms"] > 0


@pytest.mark.asyncio
async def test_audio_out_binary_format():
    app = _build_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "text": "Raw audio streaming test.",
            "format": "binary",
        }
        resp = await client.post("/audio_out", json=payload)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert len(resp.content) > 100
        # Verify valid WAV container
        with wave.open(io.BytesIO(resp.content), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2


@pytest.mark.asyncio
async def test_audio_out_empty_text_rejected():
    app = _build_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {"text": "   ", "format": "json"}
        resp = await client.post("/audio_out", json=payload)
        assert resp.status_code == 400
        assert "cannot be empty" in resp.json()["detail"]
