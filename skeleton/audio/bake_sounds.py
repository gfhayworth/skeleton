"""CLI tool to pre-generate and bake sound assets using OpenAI TTS (or synthetic tones) and pre-compute mouth-sync trajectories."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from skeleton.audio.config import AudioConfig
from skeleton.audio.mouth_sync import MouthSyncProcessor
from skeleton.audio.sounds import DEFAULT_SOUNDS_DIR, SoundBank, SoundCategory
from skeleton.audio.tts import OpenAITTSClient, MockTTSClient

# Built-in skeleton sound library definitions
DEFAULT_SOUND_LIBRARY = [
    # 1. Fillers (Played immediately when user stops speaking, masking latency)
    {
        "sound_id": "filler_hmm_thinking",
        "category": SoundCategory.FILLER,
        "text": "Hmm, let me ponder your existence for a moment.",
    },
    {
        "sound_id": "filler_hold_horses",
        "category": SoundCategory.FILLER,
        "text": "Hold your horses, meatbag. Thinking takes effort.",
    },
    {
        "sound_id": "filler_skull_hurts",
        "category": SoundCategory.FILLER,
        "text": "Give me a second, my empty skull is processing.",
    },
    # 2. Greetings
    {
        "sound_id": "greeting_look_who_it_is",
        "category": SoundCategory.GREETING,
        "text": "Well, well. Look what wandered into my domain.",
    },
    {
        "sound_id": "greeting_alive_still",
        "category": SoundCategory.GREETING,
        "text": "Ah, you are still alive. How terribly disappointing.",
    },
    # 3. Laughs & Cackles
    {
        "sound_id": "laugh_evil_cackle",
        "category": SoundCategory.LAUGH,
        "text": "Ha ha ha ha ha! How delightfully pathetic.",
    },
    {
        "sound_id": "laugh_dry_chuckle",
        "category": SoundCategory.LAUGH,
        "text": "Heh heh heh. You cannot be serious.",
    },
    # 4. Wake acknowledgments
    {
        "sound_id": "wake_listening",
        "category": SoundCategory.WAKE,
        "text": "I am listening, unfortunately.",
    },
    {
        "sound_id": "wake_what_now",
        "category": SoundCategory.WAKE,
        "text": "What do you want now?",
    },
    # 5. Confused / Timeout fallbacks
    {
        "sound_id": "confused_mumble",
        "category": SoundCategory.CONFUSED,
        "text": "Speak clearly. My eardrums decomposed decades ago.",
    },
    {
        "sound_id": "confused_nothing_heard",
        "category": SoundCategory.CONFUSED,
        "text": "Was that speech, or just wind blowing through your teeth?",
    },
]


async def bake_sounds(output_dir: Path, use_mock: bool = False):
    """Generates audio for all library clips, extracts jaw trajectories, and saves them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AudioConfig()
    mouth_sync = MouthSyncProcessor(config)
    bank = SoundBank(sounds_dir=output_dir, config=config, mouth_sync=mouth_sync, autoload=False)

    api_key_str = config.api_key.get_secret_value() if config.api_key else ""
    if use_mock or not api_key_str or "mock" in api_key_str:
        print("[bake_sounds] Using MockTTSClient for sound generation...")
        tts_client = MockTTSClient()
    else:
        print("[bake_sounds] Using OpenAITTSClient (voice='onyx') for sound generation...")
        tts_client = OpenAITTSClient(config)

    for item in DEFAULT_SOUND_LIBRARY:
        sound_id = item["sound_id"]
        category = item["category"]
        text = item["text"]

        print(f"Generating [{category.value}] '{sound_id}': \"{text}\"...")
        audio_bytes = await tts_client.synthesize(
            text,
            voice=config.tts_voice,
            speed=config.tts_speed,
            response_format=config.tts_format,
        )

        mouth_frames = mouth_sync.extract_jaw_trajectory(audio_bytes)
        sound = bank.register_sound(
            sound_id=sound_id,
            category=category,
            text=text,
            audio_bytes=audio_bytes,
            mouth_frames=mouth_frames,
        )
        bank.save_sound(sound)
        print(f"   Saved {sound_id}.wav and {sound_id}.json ({len(mouth_frames)} mouth frames, {sound.duration_s}s)")

    await tts_client.close()
    print(f"\nSuccessfully baked {len(DEFAULT_SOUND_LIBRARY)} sounds to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Bake pre-recorded sounds for skeleton")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_SOUNDS_DIR),
        help="Directory to save generated WAV audio and JSON sidecars",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force using MockTTSClient instead of OpenAI API",
    )
    args = parser.parse_args()
    asyncio.run(bake_sounds(Path(args.output_dir), use_mock=args.mock))


if __name__ == "__main__":
    main()
