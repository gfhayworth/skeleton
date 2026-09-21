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

    def record_with_vad(
        self,
        device_index: Optional[int] = None,
        silence_duration_s: float = 0.45,
        max_recording_s: float = 10.0,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30,
        on_speech_start=None,
    ) -> bytes:
        """Records microphone audio using Voice Activity Detection (VAD) endpointing.

        Waits for speech onset, accumulates audio during active speech,
        and endpoints when trailing silence is detected.
        """
        raise NotImplementedError

    async def record_async(
        self,
        duration_seconds: float = 3.5,
        device_index: Optional[int] = None,
    ) -> bytes:
        return await asyncio.to_thread(self.record, duration_seconds, device_index)

    async def record_with_vad_async(
        self,
        device_index: Optional[int] = None,
        silence_duration_s: float = 0.45,
        max_recording_s: float = 10.0,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30,
        on_speech_start=None,
    ) -> bytes:
        return await asyncio.to_thread(
            self.record_with_vad,
            device_index=device_index,
            silence_duration_s=silence_duration_s,
            max_recording_s=max_recording_s,
            aggressiveness=aggressiveness,
            frame_duration_ms=frame_duration_ms,
            on_speech_start=on_speech_start,
        )


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

    def record_with_vad(
        self,
        device_index: Optional[int] = None,
        silence_duration_s: float = 0.45,
        max_recording_s: float = 10.0,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30,
        on_speech_start=None,
    ) -> bytes:
        """Records microphone speech streaming frames and endpoints via WebRTC VAD.

        Args:
            device_index: Input device index.
            silence_duration_s: Trailing silence duration to stop recording once speech has started.
            max_recording_s: Maximum allowable recording duration.
            aggressiveness: VAD sensitivity (0=least aggressive, 3=most aggressive).
            frame_duration_ms: Frame chunk in ms (10, 20, or 30).
            on_speech_start: Optional callback invoked when speech onset is detected.

        Returns:
            Standard 16-bit PCM WAV audio bytes.
        """
        try:
            import sounddevice as sd
            import webrtcvad
        except ImportError as err:
            raise AudioDeviceError(f"Required audio library sounddevice/webrtcvad not available: {err}") from err

        if frame_duration_ms not in (10, 20, 30):
            raise ValueError(f"frame_duration_ms must be 10, 20, or 30 (got {frame_duration_ms})")

        vad = webrtcvad.Vad(aggressiveness)
        samples_per_frame = int(self.sample_rate * frame_duration_ms / 1000)
        bytes_per_frame = samples_per_frame * 2  # 16-bit = 2 bytes per sample

        max_frames = int(max_recording_s * 1000 / frame_duration_ms)
        silence_frame_limit = int(silence_duration_s * 1000 / frame_duration_ms)

        # Pre-speech circular ring buffer (keep ~300ms before speech onset for natural transients)
        ring_buffer_size = int(0.3 * 1000 / frame_duration_ms)
        ring_buffer: List[bytes] = []

        voiced_frames: List[bytes] = []
        speech_started = False
        consecutive_silence = 0

        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=samples_per_frame,
                dtype="int16",
                channels=1,
                device=device_index,
            ) as stream:
                for _ in range(max_frames):
                    frame_bytes, overflowed = stream.read(samples_per_frame)
                    if len(frame_bytes) < bytes_per_frame:
                        continue

                    # Evaluate speech presence via WebRTC VAD
                    is_speech = vad.is_speech(bytes(frame_bytes), self.sample_rate)

                    if not speech_started:
                        ring_buffer.append(bytes(frame_bytes))
                        if len(ring_buffer) > ring_buffer_size:
                            ring_buffer.pop(0)

                        if is_speech:
                            speech_started = True
                            if on_speech_start:
                                on_speech_start()
                            voiced_frames.extend(ring_buffer)
                            voiced_frames.append(bytes(frame_bytes))
                    else:
                        voiced_frames.append(bytes(frame_bytes))
                        if not is_speech:
                            consecutive_silence += 1
                            if consecutive_silence >= silence_frame_limit:
                                break
                        else:
                            consecutive_silence = 0
        except Exception as exc:
            raise AudioDeviceError(f"Streaming VAD recording failed: {exc}") from exc

        # If no speech was detected, fallback to whatever ring buffer we accumulated
        all_frames = voiced_frames if voiced_frames else ring_buffer
        raw_pcm = b"".join(all_frames)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(raw_pcm)

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

    def record_with_vad(
        self,
        device_index: Optional[int] = None,
        silence_duration_s: float = 0.45,
        max_recording_s: float = 10.0,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30,
        on_speech_start=None,
    ) -> bytes:
        if on_speech_start:
            on_speech_start()
        # Mock recorder simulates 1.0 second of speech followed by trailing silence
        # which triggers the VAD endpoint after speech_duration + silence_duration
        mock_speech_duration = min(1.0, max_recording_s)
        return self.record(duration_seconds=mock_speech_duration, device_index=device_index)
