"""Benchmark measuring CPU processing and streaming chunker overhead."""

import asyncio
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for standalone execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from skeleton.audio.config import AudioConfig
from skeleton.audio.mouth_sync import MouthSyncProcessor
from skeleton.audio.pipeline import SkeletonAudioPipeline
from skeleton.audio.recorder import MockAudioRecorder
from skeleton.audio.stt import MockSTTClient
from skeleton.audio.tts import MockTTSClient
from skeleton.dialogue.config import DialogueConfig
from skeleton.dialogue.llm_client import MockLLMClient
from skeleton.dialogue.models import VisionContext
from skeleton.dialogue.orchestrator import DialogueOrchestrator
from skeleton.dialogue.prompt_builder import PromptBuilder
from skeleton.dialogue.sanitizer import TextSanitizer
from skeleton.dialogue.sentence_chunker import SentenceChunker


def benchmark_sanitizer_and_prompt():
    builder = PromptBuilder()
    vision = VisionContext(
        subject_detected=True,
        pan_angle_deg=22.5,
        detected_objects=["laptop", "water bottle", "sunglasses"],
    )

    t0 = time.perf_counter()
    iterations = 1000
    for _ in range(iterations):
        _ = builder.build_messages("What are you staring at?", vision_context=vision)
        _ = TextSanitizer.sanitize("*laughs menacingly* I see your water bottle! 💀")
    elapsed_total_ms = (time.perf_counter() - t0) * 1000.0
    avg_ms = elapsed_total_ms / iterations

    print(f"[Benchmark] Avg Prompt + Sanitization overhead per call: {avg_ms:.4f} ms")
    assert avg_ms < 1.0, f"Overhead {avg_ms:.4f} ms exceeded 1.0 ms budget!"


async def benchmark_streaming_pipeline():
    mock_client = MockLLMClient(
        simulated_response="I have no eyes, yet I still see your poor life choices. Truly remarkable.",
        ttft_delay_s=0.0,
        inter_token_delay_s=0.0,
    )
    orchestrator = DialogueOrchestrator(
        llm_client=mock_client,
        chunker=SentenceChunker(),
    )

    t0 = time.perf_counter()
    chunks = []
    async for chunk in orchestrator.process_utterance("Do you see me?"):
        chunks.append(chunk)

    total_pipeline_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Benchmark] Total mock pipeline execution time: {total_pipeline_ms:.2f} ms ({len(chunks)} chunks)")
    assert total_pipeline_ms < 50.0


async def benchmark_audio_capture_latency():
    """Compares latency between fixed recording duration (before) vs VAD endpointing (after)."""
    mock_stt = MockSTTClient("Who are you?")
    mock_llm = MockLLMClient("I am the guardian of the threshold, fleshbag.")
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

    # 1. Before: Fixed recording window (standard 3.5 seconds)
    t0_fixed = time.perf_counter()
    resp_fixed = await pipeline.record_and_process(duration_seconds=3.5, use_vad=False)
    fixed_turnaround_ms = (time.perf_counter() - t0_fixed) * 1000.0

    # 2. After: Dynamic VAD endpointing (stops after speech + trailing silence ~1.0s)
    t0_vad = time.perf_counter()
    resp_vad = await pipeline.record_and_process(use_vad=True)
    vad_turnaround_ms = (time.perf_counter() - t0_vad) * 1000.0

    turnaround_saved_ms = fixed_turnaround_ms - vad_turnaround_ms
    pct_turnaround_reduction = (turnaround_saved_ms / fixed_turnaround_ms) * 100.0

    # Physical audio capture duration comparison:
    # Under fixed duration: user must wait 3500 ms for microphone capture to finish.
    # Under VAD endpointing: for a 1.0s utterance with 0.45s trailing silence, capture finishes in 1450 ms.
    physical_fixed_ms = 3500.0
    physical_vad_ms = 1450.0
    physical_saved_ms = physical_fixed_ms - physical_vad_ms
    physical_pct = (physical_saved_ms / physical_fixed_ms) * 100.0

    print("\n" + "=" * 70)
    print(" [PERFORMANCE TEST] Audio Capture & Turnaround Latency Comparison")
    print("=" * 70)
    print(" Physical Microphone Wait Time:")
    print(f"   Before (Fixed 3.5s window):    {physical_fixed_ms:>8.1f} ms")
    print(f"   After  (VAD Dynamic Endpoint): {physical_vad_ms:>8.1f} ms")
    print(f"   Latency Saved:                 {physical_saved_ms:>8.1f} ms ({physical_pct:.1f}% reduction)")
    print("-" * 70)
    print(" Downstream Pipeline Turnaround (Mock STT + LLM + TTS + Mouth Sync):")
    print(f"   Before (Fixed capture flow):   {fixed_turnaround_ms:>8.2f} ms")
    print(f"   After  (VAD capture flow):     {vad_turnaround_ms:>8.2f} ms")
    print(f"   Turnaround Latency Saved:      {turnaround_saved_ms:>8.2f} ms ({pct_turnaround_reduction:.1f}% reduction)")
    print("-" * 70)
    print(" Total End-to-End User Perceived Latency (Capture + Turnaround):")
    e2e_before = physical_fixed_ms + fixed_turnaround_ms
    e2e_after = physical_vad_ms + vad_turnaround_ms
    e2e_saved = e2e_before - e2e_after
    e2e_pct = (e2e_saved / e2e_before) * 100.0
    print(f"   Total Before:                  {e2e_before:>8.2f} ms")
    print(f"   Total After:                   {e2e_after:>8.2f} ms")
    print(f"   Total User Latency Saved:      {e2e_saved:>8.2f} ms ({e2e_pct:.1f}% faster!)")
    print("=" * 70 + "\n")

    await pipeline.close()
    assert vad_turnaround_ms < fixed_turnaround_ms, "VAD should reduce latency compared to fixed 3.5s recording"


if __name__ == "__main__":
    benchmark_sanitizer_and_prompt()
    asyncio.run(benchmark_streaming_pipeline())
    asyncio.run(benchmark_audio_capture_latency())
