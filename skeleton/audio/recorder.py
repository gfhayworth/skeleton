"""Audio recording module capturing user microphone speech into standard 16-bit PCM WAV."""

import asyncio
import io
import math
import struct
import wave
from typing import List, Optional
from pydantic import BaseModel, Field


class AudioDeviceError(Exception):
    """Raised when an audio input device fails or is inaccessible."""
    pass


class AudioDevice(BaseModel):
    """Metadata representing an available physical audio input device."""

    index: int = Field(..., description="Device index in PortAudio system.")
    name: str = Field(..., description="Human-readable device name.")
    max_input_channels: int = Field(..., description="Number of supported input channels.")
    default_samplerate: float = Field(..., description="Default sampling rate in Hz.")


class BaseAudioRecorder:
    """Abstract base class for audio recorders."""

    def list_input_devices(self) -> List[AudioDevice]:
        raise NotImplementedError

    def record(
        self,
        duration_seconds: float = 3.5,
        device_index: Optional[int] = None,
    ) -> bytes:
        raise NotImplementedError

    async def record_async(
        self,
        duration_seconds: float = 3.5,
        device_index: Optional[int] = None,
    ) -> bytes:
        return await asyncio.to_thread(self.record, duration_seconds, device_index)


class AudioRecorder(BaseAudioRecorder):
    """Captures microphone audio using sounddevice, formatted as 16kHz 16-bit PCM mono WAV.

    Optimized natively for OpenAI Whisper acoustic processing.
    """

    DEFAULT_SAMPLE_RATE = 16000  # 16 kHz optimal Whisper sample rate
    MIN_DURATION_S = 0.5
    MAX_DURATION_S = 30.0

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def list_input_devices(self) -> List[AudioDevice]:
        """Discovers and enumerates all available physical audio input devices."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices: List[AudioDevice] = []
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    input_devices.append(
                        AudioDevice(
                            index=idx,
                            name=dev.get("name", f"Device {idx}"),
                            max_input_channels=dev.get("max_input_channels", 1),
                            default_samplerate=dev.get("default_samplerate", 44100.0),
                        )
                    )
            return input_devices
        except Exception as exc:
            raise AudioDeviceError(f"Failed to query audio input devices: {exc}") from exc

    def record(
        self,
        duration_seconds: float = 3.5,
        device_index: Optional[int] = None,
    ) -> bytes:
        """Synchronously records audio from the microphone for the specified duration.

        Returns:
            Standard 16-bit PCM WAV audio bytes.
        """
        if duration_seconds < self.MIN_DURATION_S:
            raise ValueError(f"Recording duration must be >= {self.MIN_DURATION_S}s (got {duration_seconds}s)")
        if duration_seconds > self.MAX_DURATION_S:
            raise ValueError(f"Recording duration must be <= {self.MAX_DURATION_S}s (got {duration_seconds}s)")

        try:
            import sounddevice as sd
            import numpy as np
        except ImportError as err:
            raise AudioDeviceError(f"Required audio library sounddevice/numpy not available: {err}") from err

        num_frames = int(self.sample_rate * duration_seconds)

        try:
            # Record 16-bit PCM mono audio
            recording = sd.rec(
                num_frames,
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                device=device_index,
            )
            sd.wait()  # Wait until recording is finished
        except Exception as exc:
            raise AudioDeviceError(f"Microphone recording failed on device {device_index}: {exc}") from exc

        # Package raw int16 PCM into standard WAV format
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(recording.tobytes())

        return buffer.getvalue()


class MockAudioRecorder(BaseAudioRecorder):
    """Deterministic offline mock recorder for headless CI and testing without physical microphones."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def list_input_devices(self) -> List[AudioDevice]:
        return [
            AudioDevice(
                index=0,
                name="Mock Microphone",
                max_input_channels=1,
                default_samplerate=16000.0,
            )
        ]

    def record(
        self,
        duration_seconds: float = 3.5,
        device_index: Optional[int] = None,
    ) -> bytes:
        if duration_seconds < 0.5:
            raise ValueError("Duration too short.")
        if duration_seconds > 30.0:
            raise ValueError("Duration too long.")

        num_samples = int(self.sample_rate * duration_seconds)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)

            # Generate modulated 220Hz synthetic speech pulse tone
            frames = bytearray()
            for i in range(num_samples):
                t = i / self.sample_rate
                syllable_mod = (math.sin(2 * math.pi * 3.0 * t) + 1.0) / 2.0
                sample = int(10000 * syllable_mod * math.sin(2 * math.pi * 220.0 * t))
                frames.extend(struct.pack("<h", sample))

            wav_file.writeframes(frames)

        return buffer.getvalue()
