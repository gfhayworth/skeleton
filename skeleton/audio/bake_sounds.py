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
        "sound_id": "greeting_nice_costume",
        "category": SoundCategory.GREETING,
        "text": "Nice costume! Oh wait, that's just your face.",
    },
    {
        "sound_id": "greeting_no_guts_no_brains",
        "category": SoundCategory.GREETING,
        "text": "I have no guts, but you clearly have no brains.",
    },
    {
        "sound_id": "greeting_trapped_in_cobweb",
        "category": SoundCategory.GREETING,
        "text": "I've seen scarier things trapped in a cobweb.",
    },
    {
        "sound_id": "greeting_stray_dog_rejected",
        "category": SoundCategory.GREETING,
        "text": "You look like something a stray dog dug up and immediately rejected.",
    },
    {
        "sound_id": "greeting_bone_to_pick",
        "category": SoundCategory.GREETING,
        "text": "I have a bone to pick with you, but you'd probably lose that too.",
    },
    {
        "sound_id": "greeting_zombies_pass",
        "category": SoundCategory.GREETING,
        "text": "Even the zombies would pass on you—nothing to eat up there!",
    },
    {
        "sound_id": "greeting_ghost_transparent",
        "category": SoundCategory.GREETING,
        "text": "Are you a ghost? Because your presence is totally transparent.",
    },
    {
        "sound_id": "greeting_roll_eyes_decayed",
        "category": SoundCategory.GREETING,
        "text": "I would roll my eyes at you, but they decayed fifty years ago.",
    },
    {
        "sound_id": "greeting_keep_walking_meatbag",
        "category": SoundCategory.GREETING,
        "text": "Keep walking, meatbag, you're embarrassing both of us.",
    },
    {
        "sound_id": "greeting_dead_more_life",
        "category": SoundCategory.GREETING,
        "text": "I'm dead, and I still have more life in me than your conversation.",
    },
    {
        "sound_id": "greeting_dust_off_tibia",
        "category": SoundCategory.GREETING,
        "text": "You couldn't scare the dust off my tibia.",
    },
    {
        "sound_id": "greeting_skin_crawling_away",
        "category": SoundCategory.GREETING,
        "text": "If I had skin, it would be crawling away from you right now.",
    },
    {
        "sound_id": "greeting_witch_curse",
        "category": SoundCategory.GREETING,
        "text": "Did a witch curse you, or did you just wake up looking like that?",
    },
    {
        "sound_id": "greeting_gravestones_personality",
        "category": SoundCategory.GREETING,
        "text": "I’ve met gravestones with more personality.",
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
    if use_mock:
        print("[bake_sounds] Using MockTTSClient (forced via --mock)...")
        tts_client = MockTTSClient()
    elif not api_key_str or "mock" in api_key_str:
        raise ValueError(
            "ERROR: OPENAI_API_KEY is not set in .env or environment. "
            "Cannot generate authentic spoken audio without a valid API key. "
            "Pass --mock if synthetic sine tones are explicitly desired."
        )
    else:
        print(f"[bake_sounds] Using OpenAITTSClient (voice='{config.tts_voice}', speed={config.tts_speed})...")
        tts_client = OpenAITTSClient(config)

    try:
        total = len(DEFAULT_SOUND_LIBRARY)
        for idx, item in enumerate(DEFAULT_SOUND_LIBRARY, 1):
            sound_id = item["sound_id"]
            category = item["category"]
            text = item["text"]

            print(f"[{idx}/{total}] Generating [{category.value}] '{sound_id}': \"{text}\"...")

            # Retry with exponential backoff on transient errors
            audio_bytes = b""
            for attempt in range(1, 4):
                try:
                    audio_bytes = await tts_client.synthesize(
                        text,
                        voice=config.tts_voice,
                        speed=config.tts_speed,
                        response_format="wav",
                    )
                    break
                except Exception as err:
                    if attempt == 3:
                        raise
                    print(f"   Warning: Attempt {attempt} failed ({err}). Retrying in {attempt * 2}s...")
                    await asyncio.sleep(attempt * 2.0)

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

        print(f"\nSuccessfully baked {total} authentic sounds to {output_dir}")
    finally:
        await tts_client.close()


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
