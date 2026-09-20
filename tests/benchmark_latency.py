"""Benchmark measuring CPU processing and streaming chunker overhead."""

import asyncio
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for standalone execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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


if __name__ == "__main__":
    benchmark_sanitizer_and_prompt()
    asyncio.run(benchmark_streaming_pipeline())
