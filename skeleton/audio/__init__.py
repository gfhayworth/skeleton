"""Skeleton audio subsystem (Speech-to-Text, Text-to-Speech, and Mouth Sync)."""

from skeleton.audio.config import AudioConfig
from skeleton.audio.mouth_sync import MouthSyncFrame, MouthSyncProcessor
from skeleton.audio.pipeline import AudioDialogueResponse, SkeletonAudioPipeline
from skeleton.audio.recorder import (
    AudioDevice,
    AudioDeviceError,
    AudioRecorder,
    BaseAudioRecorder,
    MockAudioRecorder,
)
from skeleton.audio.stt import BaseSTTClient, MockSTTClient, WhisperSTTClient
from skeleton.audio.tts import BaseTTSClient, MockTTSClient, OpenAITTSClient

__all__ = [
    "AudioConfig",
    "AudioDevice",
    "AudioDeviceError",
    "BaseAudioRecorder",
    "AudioRecorder",
    "MockAudioRecorder",
    "BaseSTTClient",
    "WhisperSTTClient",
    "MockSTTClient",
    "BaseTTSClient",
    "OpenAITTSClient",
    "MockTTSClient",
    "MouthSyncFrame",
    "MouthSyncProcessor",
    "SkeletonAudioPipeline",
    "AudioDialogueResponse",
]
