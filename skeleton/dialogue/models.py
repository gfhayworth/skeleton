"""Data models for multimodal context, chat messages, and streaming speech chunks."""

import time
from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class DialogueState(str, Enum):
    """Lifecycle states of the dialogue system."""
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


class VisionContext(BaseModel):
    """Multimodal vision information captured from the camera tracking subsystem."""

    subject_detected: bool = Field(
        default=True,
        description="Whether a subject/person is currently tracked in frame.",
    )
    pan_angle_deg: float = Field(
        default=0.0,
        description="Relative horizontal angle to subject in degrees (-90 to +90).",
    )
    tilt_angle_deg: float = Field(
        default=0.0,
        description="Relative vertical angle to subject in degrees (-45 to +45).",
    )
    distance_m: Optional[float] = Field(
        default=1.5,
        description="Estimated distance to subject in meters.",
    )
    detected_objects: List[str] = Field(
        default_factory=list,
        description="Objects identified on or around the subject (e.g. ['coffee mug', 'red hat']).",
    )
    subject_facing_skeleton: bool = Field(
        default=True,
        description="Whether the subject is facing/looking towards the skeleton.",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Timestamp when vision frame was processed.",
    )

    def to_prompt_string(self) -> str:
        """Serializes vision context into a concise summary for prompt inclusion."""
        if not self.subject_detected:
            return "No subject currently detected in visual field."

        orientation = (
            f"{abs(self.pan_angle_deg):.1f}° {'right' if self.pan_angle_deg > 0 else 'left'}"
            if abs(self.pan_angle_deg) > 1.0
            else "directly in front"
        )
        facing = "looking towards you" if self.subject_facing_skeleton else "looking away"
        parts = [f"Subject is {orientation}, {facing}"]

        if self.distance_m is not None:
            parts.append(f"approx {self.distance_m:.1f}m away")

        if self.detected_objects:
            parts.append(f"detected items: {', '.join(self.detected_objects)}")

        return ". ".join(parts) + "."


class ChatMessage(BaseModel):
    """Represents a conversational message turn."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: float = Field(default_factory=time.time)


class SpeechChunk(BaseModel):
    """A coherent phrase or sentence chunk emitted for real-time TTS and mouth actuation."""

    text: str = Field(..., description="Cleaned, playable speech text fragment.")
    sequence_index: int = Field(..., description="0-indexed order of this chunk in the utterance.")
    is_final: bool = Field(default=False, description="True if this is the terminating chunk of the response.")
    elapsed_ms: float = Field(default=0.0, description="Milliseconds elapsed since dialogue request began.")
    word_count: int = Field(default=0, description="Word count of this specific chunk.")

    def model_post_init(self, __context) -> None:
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())
