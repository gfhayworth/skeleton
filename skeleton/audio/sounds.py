"""Pre-recorded audio sounds bank and fast-path playback for the animatronic skeleton."""

import enum
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from skeleton.audio.config import AudioConfig
from skeleton.audio.mouth_sync import MouthSyncFrame, MouthSyncProcessor

logger = logging.getLogger(__name__)

# Default sounds assets directory relative to project root
DEFAULT_SOUNDS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "sounds"


class SoundCategory(str, enum.Enum):
    """Categorization for pre-recorded sound assets."""

    FILLER = "filler"            # "Hmm let me think...", "Hold your horses..."
    GREETING = "greeting"        # "Look who decided to show up.", "Well, well, well."
    LAUGH = "laugh"              # Cackles, dry chuckles
    SNARK = "snark"              # Sarcastic remarks, snappy one-liners
    CONFUSED = "confused"        # "Speak up, I haven't got all century."
    WAKE = "wake"                # "I'm listening.", "What now?"


class PreRecordedSound(BaseModel):
    """A pre-recorded audio sound clip with cached mouth-sync trajectory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sound_id: str = Field(..., description="Unique identifier for the sound clip.")
    category: SoundCategory = Field(..., description="Category classifying the type of sound.")
    text: str = Field(..., description="Transcript or description of the sound utterance.")
    audio_path: Path = Field(..., description="Path to the WAV audio file.")
    audio_bytes: bytes = Field(default=b"", description="In-memory WAV audio bytes.")
    mouth_frames: List[MouthSyncFrame] = Field(
        default_factory=list,
        description="Pre-computed 30Hz synchronized jaw servo frames.",
    )
    duration_s: float = Field(default=0.0, description="Duration of the audio clip in seconds.")


class SoundBank:
    """Manages pre-recorded audio assets and pre-computed mouth synchronization frames."""

    def __init__(
        self,
        sounds_dir: Optional[Union[str, Path]] = None,
        config: Optional[AudioConfig] = None,
        mouth_sync: Optional[MouthSyncProcessor] = None,
        autoload: bool = True,
    ):
        self.sounds_dir = Path(sounds_dir) if sounds_dir else DEFAULT_SOUNDS_DIR
        self.config = config or AudioConfig()
        self.mouth_sync = mouth_sync or MouthSyncProcessor(self.config)
        self._sounds: Dict[str, PreRecordedSound] = {}

        if autoload and self.sounds_dir.exists():
            self.load_all()

    @property
    def sounds(self) -> Dict[str, PreRecordedSound]:
        """Returns all currently registered pre-recorded sounds."""
        return self._sounds

    def register_sound(
        self,
        sound_id: str,
        category: SoundCategory,
        text: str,
        audio_bytes: bytes,
        audio_path: Optional[Path] = None,
        mouth_frames: Optional[List[MouthSyncFrame]] = None,
        duration_s: Optional[float] = None,
    ) -> PreRecordedSound:
        """Registers a sound into the bank, computing mouth frames if not already supplied."""
        if mouth_frames is None:
            mouth_frames = self.mouth_sync.extract_jaw_trajectory(audio_bytes)

        if duration_s is None:
            # Approximate or compute duration based on mouth frames (30 fps)
            duration_s = len(mouth_frames) / float(self.config.mouth_fps) if mouth_frames else 0.0

        sound = PreRecordedSound(
            sound_id=sound_id,
            category=category,
            text=text,
            audio_path=audio_path or (self.sounds_dir / f"{sound_id}.wav"),
            audio_bytes=audio_bytes,
            mouth_frames=mouth_frames,
            duration_s=round(duration_s, 3),
        )
        self._sounds[sound_id] = sound
        return sound

    def get(self, sound_id: str) -> Optional[PreRecordedSound]:
        """Retrieves a sound by ID."""
        return self._sounds.get(sound_id)

    def get_by_category(self, category: Union[SoundCategory, str]) -> List[PreRecordedSound]:
        """Retrieves all sounds matching a given category."""
        target_cat = SoundCategory(category) if isinstance(category, str) else category
        return [s for s in self._sounds.values() if s.category == target_cat]

    def get_random(self, category: Optional[Union[SoundCategory, str]] = None) -> Optional[PreRecordedSound]:
        """Returns a random sound, optionally filtered by category."""
        import random
        candidates = self.get_by_category(category) if category else list(self._sounds.values())
        if not candidates:
            return None
        return random.choice(candidates)

    def load_all(self) -> int:
        """Loads all WAV files and corresponding .json metadata/frame sidecars from sounds_dir."""
        if not self.sounds_dir.exists():
            return 0

        loaded_count = 0
        for wav_path in self.sounds_dir.glob("*.wav"):
            sound_id = wav_path.stem
            metadata_path = wav_path.with_suffix(".json")

            audio_bytes = wav_path.read_bytes()
            text = sound_id.replace("_", " ").capitalize()
            category = SoundCategory.SNARK
            cached_frames: Optional[List[MouthSyncFrame]] = None

            if metadata_path.exists():
                try:
                    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                    text = meta.get("text", text)
                    category_str = meta.get("category", "snark")
                    try:
                        category = SoundCategory(category_str)
                    except ValueError:
                        category = SoundCategory.SNARK

                    if "mouth_frames" in meta:
                        cached_frames = [
                            MouthSyncFrame(
                                timestamp_ms=f["timestamp_ms"],
                                jaw_angle_deg=f["jaw_angle_deg"],
                                normalized_open=f["normalized_open"],
                            )
                            for f in meta["mouth_frames"]
                        ]
                except Exception as err:
                    logger.warning(f"Failed to read sound metadata for {metadata_path}: {err}")

            self.register_sound(
                sound_id=sound_id,
                category=category,
                text=text,
                audio_bytes=audio_bytes,
                audio_path=wav_path,
                mouth_frames=cached_frames,
            )
            loaded_count += 1

        return loaded_count

    def save_sound(self, sound: PreRecordedSound) -> None:
        """Persists a sound's WAV audio and metadata/jaw frames sidecar to disk."""
        self.sounds_dir.mkdir(parents=True, exist_ok=True)
        wav_path = self.sounds_dir / f"{sound.sound_id}.wav"
        meta_path = self.sounds_dir / f"{sound.sound_id}.json"

        wav_path.write_bytes(sound.audio_bytes)

        metadata = {
            "sound_id": sound.sound_id,
            "category": sound.category.value,
            "text": sound.text,
            "duration_s": sound.duration_s,
            "mouth_frames": [
                {
                    "timestamp_ms": f.timestamp_ms,
                    "jaw_angle_deg": f.jaw_angle_deg,
                    "normalized_open": f.normalized_open,
                }
                for f in sound.mouth_frames
            ],
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        sound.audio_path = wav_path
        self._sounds[sound.sound_id] = sound
