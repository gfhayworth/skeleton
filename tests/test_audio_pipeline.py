"""Integration tests for the complete audio pipeline."""

import pytest
from skeleton.audio.mouth_sync import MouthSyncProcessor
from skeleton.audio.pipeline import SkeletonAudioPipeline
from skeleton.audio.recorder import MockAudioRecorder
from skeleton.audio.stt import MockSTTClient
from skeleton.audio.tts import MockTTSClient
from skeleton.dialogue.llm_client import MockLLMClient
from skeleton.dialogue.orchestrator import DialogueOrchestrator


@pytest.mark.asyncio
async def test_audio_pipeline_end_to_end():
    mock_stt = MockSTTClient("Hey skeleton, are you awake?")
    mock_llm = MockLLMClient("I don't sleep, fleshbag. Ever.")
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

    # 1. Process simulated audio turn
    dummy_audio = b"RIFF" + b"\x00" * 200
    resp = await pipeline.process_audio_turn(dummy_audio)

    assert resp.user_transcript == "Hey skeleton, are you awake?"
    assert "I don't sleep, fleshbag." in resp.skeleton_response
    assert len(resp.audio_bytes) > 500
    assert len(resp.mouth_frames) > 0
    assert resp.total_latency_ms > 0

    # 2. Process record_and_process with VAD
    resp_vad = await pipeline.record_and_process(use_vad=True)
    assert resp_vad.user_transcript == "Hey skeleton, are you awake?"
    assert len(resp_vad.audio_bytes) > 500

    await pipeline.close()
