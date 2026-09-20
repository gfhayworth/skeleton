"""Acoustic feature extractor mapping speech audio to synchronized mechanical jaw movement."""

import io
import math
import struct
import wave
from typing import List, Optional
from pydantic import BaseModel, Field
from skeleton.audio.config import AudioConfig


class MouthSyncFrame(BaseModel):
    """A single time-indexed jaw-opening command for the mechanical controller."""

    timestamp_ms: float = Field(..., description="Timestamp in milliseconds from start of utterance.")
    jaw_angle_deg: float = Field(..., description="Servo target opening in degrees (0.0 = closed).")
    normalized_open: float = Field(..., description="Normalized jaw opening between 0.0 (closed) and 1.0 (open).")


class MouthSyncProcessor:
    """Extracts energy envelopes from standard WAV audio to drive the physical skeleton jaw.

    Includes a deadband noise gate and EMA smoothing to protect servo gears from chatter.
    """

    def __init__(self, config: Optional[AudioConfig] = None, smoothing_alpha: float = 0.6):
        """Args:

        config: Audio configuration with jaw limits and noise thresholds.
        smoothing_alpha: Exponential smoothing factor (0.0 to 1.0; higher = more responsive).
        """
        self.config = config or AudioConfig()
        self.smoothing_alpha = smoothing_alpha

    def extract_jaw_trajectory(
        self,
        wav_bytes: bytes,
        time_offset_ms: float = 0.0,
    ) -> List[MouthSyncFrame]:
        """Parses WAV audio bytes and computes a 30Hz trajectory of jaw-opening angles.

        Args:
            wav_bytes: Raw uncompressed WAV file bytes (16-bit PCM).
            time_offset_ms: Starting time offset in ms (ensures continuous timeline across streaming chunks).

        Returns:
            List of MouthSyncFrame instances ready for transmission to the Arduino / Makeblock controller.
        """
        if not wav_bytes:
            return []

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                num_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                num_frames = wav_file.getnframes()
                raw_data = wav_file.readframes(num_frames)
        except (wave.Error, EOFError) as e:
            raise ValueError(f"Invalid or unsupported WAV audio format: {e}")

        if sample_width != 2:
            raise ValueError(f"Only 16-bit PCM WAV is supported for mouth sync (got {sample_width * 8}-bit).")

        # Unpack 16-bit signed integers (little-endian)
        total_samples = len(raw_data) // 2
        all_samples = struct.unpack(f"<{total_samples}h", raw_data)

        # If stereo, average left and right channels to mono
        if num_channels > 1:
            mono_samples = [
                sum(all_samples[i : i + num_channels]) // num_channels
                for i in range(0, total_samples, num_channels)
            ]
        else:
            mono_samples = all_samples

        total_mono_samples = len(mono_samples)
        if total_mono_samples == 0:
            return []

        # Calculate sample window size per frame (e.g. 24000 / 30 fps = 800 samples/frame)
        samples_per_frame = max(1, int(sample_rate / self.config.mouth_fps))
        frame_duration_ms = 1000.0 / self.config.mouth_fps

        frames: List[MouthSyncFrame] = []
        prev_smoothed_open = 0.0
        ceiling_rms = 0.35  # Approximate loud speech RMS threshold

        frame_idx = 0
        for start_idx in range(0, total_mono_samples, samples_per_frame):
            chunk = mono_samples[start_idx : start_idx + samples_per_frame]
            if not chunk:
                break

            # Calculate Root Mean Square (RMS) energy
            sum_sq = sum(float(s) * float(s) for s in chunk)
            rms = math.sqrt(sum_sq / len(chunk))
            normalized_rms = min(1.0, rms / 32768.0)

            # Deadband noise-gate: snap quiet ambient background to closed jaw
            if normalized_rms <= self.config.noise_floor_rms:
                target_open = 0.0
            else:
                effective_rms = normalized_rms - self.config.noise_floor_rms
                usable_range = max(0.01, ceiling_rms - self.config.noise_floor_rms)
                target_open = min(1.0, effective_rms / usable_range)

            # Exponential Moving Average (EMA) to prevent rapid servo gear thrashing
            smoothed_open = (
                self.smoothing_alpha * target_open
                + (1.0 - self.smoothing_alpha) * prev_smoothed_open
            )
            prev_smoothed_open = smoothed_open

            # Clamp and compute physical servo angle
            clamped_open = max(0.0, min(1.0, smoothed_open))
            jaw_angle = round(clamped_open * self.config.max_jaw_angle_deg, 2)
            timestamp_ms = round(time_offset_ms + (frame_idx * frame_duration_ms), 1)

            frames.append(
                MouthSyncFrame(
                    timestamp_ms=timestamp_ms,
                    jaw_angle_deg=jaw_angle,
                    normalized_open=round(clamped_open, 3),
                )
            )
            frame_idx += 1

        return frames
