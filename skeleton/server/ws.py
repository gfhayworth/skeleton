"""WebSocket duplex streaming handler for continuous bidirectional audio & control."""

import asyncio
import base64
import json
import logging
import time
from typing import Any, Dict, List, Literal, Optional
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from skeleton.audio.mouth_sync import MouthSyncFrame
from skeleton.audio.pipeline import AudioChunkResponse, SkeletonAudioPipeline
from skeleton.dialogue.models import VisionContext

logger = logging.getLogger(__name__)


class WsInboundMessage(BaseModel):
    """Inbound client message schema for /ws/audio."""

    type: Literal["start_turn", "audio_chunk", "end_turn", "barge_in", "ping"]
    audio_base64: Optional[str] = Field(default=None, description="Base64-encoded PCM or WAV audio bytes.")
    format: Optional[str] = Field(default="wav", description="Audio format ('wav' or 'raw_pcm').")
    sample_rate: Optional[int] = Field(default=16000, description="Audio sample rate in Hz.")
    vision_context: Optional[VisionContext] = Field(default=None, description="Optional vision environmental metadata.")


def _frames_to_dicts(frames: List[MouthSyncFrame]) -> List[Dict[str, Any]]:
    return [
        {
            "timestamp_ms": f.timestamp_ms,
            "jaw_angle_deg": round(f.jaw_angle_deg, 2),
            "normalized_open": round(f.normalized_open, 3),
        }
        for f in frames
    ]


class WebSocketAudioSession:
    """Manages state and full-duplex streaming for a single WebSocket connection."""

    def __init__(self, websocket: WebSocket, pipeline: SkeletonAudioPipeline):
        self.websocket = websocket
        self.pipeline = pipeline
        self.active_generation_task: Optional[asyncio.Task] = None
        self.audio_buffer = bytearray()
        self.audio_format = "wav"
        self.sample_rate = 16000
        self.current_vision: Optional[VisionContext] = None

    async def send_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Sends a JSON event packet over the duplex WebSocket connection."""
        payload = {"event": event_type, "data": data}
        await self.websocket.send_text(json.dumps(payload))

    async def handle_start_turn(self, msg: WsInboundMessage) -> None:
        """Initializes a new conversational turn, resetting audio buffers."""
        # Barge-in / cancel previous generation if still active
        await self.cancel_active_turn(reason="new_turn")
        self.audio_buffer.clear()
        self.audio_format = msg.format or "wav"
        self.sample_rate = msg.sample_rate or 16000
        self.current_vision = msg.vision_context
        await self.send_event("state", {"state": "listening"})

    def handle_audio_chunk_raw(self, chunk_bytes: bytes) -> None:
        """Appends incoming binary audio bytes to turn buffer."""
        self.audio_buffer.extend(chunk_bytes)

    def handle_audio_chunk_msg(self, msg: WsInboundMessage) -> None:
        """Appends incoming base64-encoded audio bytes to turn buffer."""
        if msg.audio_base64:
            try:
                decoded = base64.b64decode(msg.audio_base64)
                self.audio_buffer.extend(decoded)
            except Exception as exc:
                logger.warning("Failed to decode base64 audio chunk: %s", exc)

    async def handle_barge_in(self) -> None:
        """Immediately interrupts ongoing speech generation and notifies client."""
        cancelled = await self.cancel_active_turn(reason="barge_in")
        self.pipeline.orchestrator.cancel_active_turn()
        await self.send_event("barge_in_ack", {"cancelled": cancelled})
        await self.send_event("state", {"state": "interrupted"})

    async def cancel_active_turn(self, reason: str = "barge_in") -> bool:
        """Cancels background synthesis task if currently running."""
        if self.active_generation_task and not self.active_generation_task.done():
            self.active_generation_task.cancel()
            try:
                await self.active_generation_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self.active_generation_task = None
            return True
        return False

    async def handle_end_turn(self) -> None:
        """Processes collected audio, runs Whisper STT, and streams dialogue speech chunks."""
        raw_audio = bytes(self.audio_buffer)
        self.audio_buffer.clear()

        if len(raw_audio) < 100:
            await self.send_event(
                "error",
                {"error": f"Audio payload too small ({len(raw_audio)} bytes); minimum 100 bytes required."},
            )
            await self.send_event("state", {"state": "idle"})
            return

        # Start asynchronous synthesis pipeline
        self.active_generation_task = asyncio.create_task(
            self._run_synthesis_pipeline(raw_audio, self.current_vision)
        )

    async def _run_synthesis_pipeline(
        self,
        raw_audio: bytes,
        vision_context: Optional[VisionContext],
    ) -> None:
        """Asynchronously executes STT, streaming LLM dialogue, and TTS audio/mouth-sync generation."""
        t0 = time.perf_counter()
        try:
            await self.send_event("state", {"state": "transcribing"})

            # Transcribe user audio
            transcript = await self.pipeline.stt_client.transcribe(
                raw_audio,
                filename="duplex_turn.wav" if self.audio_format == "wav" else "duplex_turn.raw",
            )
            await self.send_event("transcript", {"transcript": transcript})
            await self.send_event("state", {"state": "speaking"})

            # Stream dialogue generation, TTS synthesis, and jaw sync frames
            chunk_count = 0
            async for chunk in self.pipeline.process_text_stream(
                transcript,
                vision_context=vision_context,
                t0=t0,
            ):
                chunk_count += 1
                chunk_data = {
                    "chunk_index": chunk.chunk_index,
                    "is_final": chunk.is_final,
                    "text": chunk.text,
                    "audio_base64": base64.b64encode(chunk.audio_bytes).decode("ascii"),
                    "mouth_frames": _frames_to_dicts(chunk.mouth_frames),
                    "latency_ms": chunk.latency_ms,
                    "cumulative_duration_s": chunk.cumulative_duration_s,
                }
                await self.send_event("chunk", chunk_data)

            total_ms = (time.perf_counter() - t0) * 1000.0
            await self.send_event(
                "done",
                {
                    "total_latency_ms": round(total_ms, 2),
                    "chunks_sent": chunk_count,
                },
            )
            await self.send_event("state", {"state": "idle"})

        except asyncio.CancelledError:
            logger.info("Synthesis pipeline task cancelled by client.")
            await self.send_event("state", {"state": "interrupted"})
            raise
        except Exception as exc:
            logger.error("Synthesis pipeline failure: %s", exc, exc_info=True)
            await self.send_event("error", {"error": str(exc)})
            await self.send_event("state", {"state": "idle"})


async def handle_websocket_audio(websocket: WebSocket, pipeline: SkeletonAudioPipeline) -> None:
    """FastAPI WebSocket endpoint handler for /ws/audio."""
    await websocket.accept()
    session = WebSocketAudioSession(websocket, pipeline)
    await session.send_event("state", {"state": "ready"})

    try:
        while True:
            ws_msg = await websocket.receive()
            if ws_msg.get("type") == "websocket.disconnect":
                break

            # 1. Binary message: raw audio chunk
            if "bytes" in ws_msg and ws_msg["bytes"] is not None:
                session.handle_audio_chunk_raw(ws_msg["bytes"])
                continue

            # 2. Text message: JSON control schema
            text_data = ws_msg.get("text")
            if not text_data:
                continue

            try:
                data_dict = json.loads(text_data)
                inbound = WsInboundMessage(**data_dict)
            except Exception as parse_err:
                await session.send_event("error", {"error": f"Invalid JSON message format: {parse_err}"})
                continue

            if inbound.type == "start_turn":
                await session.handle_start_turn(inbound)
            elif inbound.type == "audio_chunk":
                session.handle_audio_chunk_msg(inbound)
            elif inbound.type == "end_turn":
                await session.handle_end_turn()
            elif inbound.type == "barge_in":
                await session.handle_barge_in()
            elif inbound.type == "ping":
                await session.send_event("pong", {"timestamp": time.time()})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected cleanly.")
    except Exception as exc:
        logger.warning("WebSocket session error: %s", exc)
    finally:
        await session.cancel_active_turn(reason="disconnect")
