from pathlib import Path

code = r'''import json
import os
import time
import wave
from pathlib import Path

from PIL import Image, ImageDraw
from google import genai
from google.genai import types
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TIKTOK_ACCESS_TOKEN = os.environ.get("TIKTOK_ACCESS_TOKEN")

# Posting stays OFF until explicitly enabled in GitHub Actions.
ENABLE_TIKTOK_POST = (
    os.environ.get("ENABLE_TIKTOK_POST", "false").lower() == "true"
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

VIDEO_PATH = OUTPUT_DIR / "amharic_comedy_video.mp4"
AUDIO_PATH = OUTPUT_DIR / "comedy_audio.wav"

WIDTH = 1080
HEIGHT = 1920
FPS = 24


def generate_comedy_package(client):
    """Generate an Amharic comedy script and two scene prompts."""

    instruction = """
You are an Ethiopian Amharic comedy writer.

Create a short, funny, original conversation between two Ethiopian
adult characters in a familiar setting such as a taxi, cafe, family home,
or among friends in Addis Ababa.

Return valid JSON only, using exactly these keys:
{
  "dialogue_script": "Speaker1: ...\\nSpeaker2: ...",
  "scene_prompts": [
    "Detailed scene description for the first half of the joke",
    "Detailed scene description for the punchline"
  ],
  "tiktok_title": "Short Amharic title with relevant hashtags"
}

Rules:
- Dialogue must be in natural Amharic.
- Make it funny, concise, and suitable for a short video.
- Include exactly two speakers named Speaker1 and Speaker2.
- Describe the same two characters consistently in both scenes.
- Keep the TikTok title short.
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Create a fresh Amharic comedy for today's video.",
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            response_mime_type="application/json",
            temperature=0.9,
        ),
    )

    data = json.loads(response.text)

    if not data.get("dialogue_script"):
        raise ValueError("Gemini returned an empty dialogue script.")

    prompts = data.get("scene_prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Gemini did not return any scene prompts.")

    data["scene_prompts"] = prompts[:2]
    data.setdefault("tiktok_title", "#AmharicComedy #Ethiopia")

    return data


def generate_dialogue_audio(client, dialogue_script, wav_path):
    """Generate two-speaker speech using Gemini TTS."""

    print("Generating Amharic dialogue audio...")

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=(
            "Speak this dialogue in Amharic with natural comedic timing. "
            "Speaker1 and Speaker2 are different characters. "
            "Read only the dialogue, not the speaker labels. "
            "Use expressive delivery where supported:\n\n"
            + dialogue_script
        ),
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker="Speaker1",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name="Puck"
                                )
                            ),
                        ),
                        types.SpeakerVoiceConfig(
                            speaker="Speaker2",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name="Kore"
                                )
                            ),
                        ),
                    ]
                )
            ),
        ),
    )

    audio_bytes = None
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if part.inline_data and part.inline_data.data:
                audio_bytes = part.inline_data.data
                break
        if audio_bytes:
            break

    if not audio_bytes:
        raise RuntimeError(
            "Gemini did not return audio. Check TTS model availability "
            "and the API error shown in the workflow logs."
        )

    # Gemini TTS commonly returns raw PCM data. Save it as a WAV file.
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(audio_bytes)

    print(f"Audio saved: {wav_path}")
    return str(wav_path)


def download_scene_image(prompt, output_path, seed):
    """
    Draw a local cartoon scene using Pillow.
    This avoids Pollinations and its previous HTTP 402 payment error.
    Note: these are simple illustrations, not AI-generated photographs.
    """

    import random

    rng = random.Random(seed)
    img = Image.new("RGB", (WIDTH, HEIGHT), (238, 205, 160))
    draw = ImageDraw.Draw(img)

    # Cafe wall and floor
    draw.rectangle((0, 0, WIDTH, 1320), fill=(239, 207, 164))
    draw.rectangle((0, 1320, WIDTH, HEIGHT), fill=(117, 77, 51))

    # Window and outdoor sky
    draw.rounded_rectangle((65, 150, 455, 620), radius=18, fill=(116, 190, 216))
    draw.rectangle((245, 150, 267, 620), fill=(247, 239, 215))
    draw.rectangle((65, 370, 455, 392), fill=(247, 239, 215))

    # Wall decoration
    draw.ellipse((760, 180, 930, 350), fill=(214, 167, 83))
    draw.ellipse((790, 210, 900, 320), fill=(239, 207, 164))

    # Character positions; slight variation between scenes.
    characters = [
        (315, (48, 111, 177), (40, 28, 22)),
        (765, (177, 63, 54), (35, 24, 20)),
    ]

    for index, (cx, shirt, hair) in enumerate(characters):
        bob = rng.randint(-8, 8)

        # Body and arms
        draw.rounded_rectangle(
            (cx - 155, 850 + bob, cx + 155, 1430),
            radius=80,
            fill=shirt,
        )
        draw.rounded_rectangle(
            (cx - 220, 930 + bob, cx - 125, 1190),
            radius=42,
            fill=shirt,
        )
        draw.rounded_rectangle(
            (cx + 125, 930 + bob, cx + 220, 1190),
            radius=42,
            fill=shirt,
        )

        # Neck, ears, face
        skin = (145, 91, 59)
        draw.rectangle((cx - 42, 775 + bob, cx + 42, 900 + bob), fill=skin)
        draw.ellipse((cx - 125, 515 + bob, cx - 78, 610 + bob), fill=skin)
        draw.ellipse((cx + 78, 515 + bob, cx + 125, 610 + bob), fill=skin)
        draw.ellipse((cx - 105, 475 + bob, cx + 105, 825 + bob), fill=skin)

        # Hair
        draw.pieslice(
            (cx - 112, 445 + bob, cx + 112, 670 + bob),
            180,
            360,
            fill=hair,
        )

        # Eyebrows, eyes, nose and smile
        draw.line(
            (cx - 68, 625 + bob, cx - 25, 615 + bob),
            fill=hair,
            width=12,
        )
        draw.line(
            (cx + 25, 615 + bob, cx + 68, 625 + bob),
            fill=hair,
            width=12,
        )
        draw.ellipse((cx - 60, 645 + bob, cx - 30, 680 + bob), fill=(25, 22, 20))
        draw.ellipse((cx + 30, 645 + bob, cx + 60, 680 + bob), fill=(25, 22, 20))
        draw.line((cx, 675 + bob, cx - 12, 720 + bob), fill=(105, 61, 42), width=8)

        # Change the mouth slightly for the punchline scene.
        if seed % 2:
            draw.ellipse((cx - 48, 735 + bob, cx + 48, 780 + bob), fill=(70, 25, 25))
        else:
            draw.arc(
                (cx - 55, 725 + bob, cx + 55, 795 + bob),
                0, 180, fill=(70, 25, 25), width=10
            )

    # Cafe table and cups
    draw.rounded_rectangle((65, 1195, 1015, 1305), radius=22, fill=(91, 52, 33))
    draw.rectangle((130, 1290, 170, 1510), fill=(91, 52, 33))
    draw.rectangle((910, 1290, 950, 1510), fill=(91, 52, 33))

    for cx in (420, 660):
        draw.rounded_rectangle(
            (cx - 42, 1125, cx + 42, 1205),
            radius=12,
            fill=(255, 246, 225),
        )
        draw.ellipse((cx - 42, 1110, cx + 42, 1140), fill=(255, 246, 225))
        draw.arc((cx + 25, 1140, cx + 70, 1190), 270, 90, fill=(255, 246, 225), width=9)

    # Small scene marker shapes only; do not draw any text.
    draw.ellipse((500, 250, 555, 305), fill=(248, 224, 111))
    draw.ellipse((520, 265, 530, 275), fill=(75, 55, 35))

    img.save(output_path, "JPEG", quality=92)
    return str(output_path)


def build_video(client, comedy_data):
    """Combine locally drawn scene images and generated audio into a vertical MP4."""

    prompts = comedy_data.get("scene_prompts", [])[:2]
    if not prompts:
        raise ValueError("No scene prompts; cannot create the video.")

    audio_path = generate_dialogue_audio(
        client,
        comedy_data["dialogue_script"],
        AUDIO_PATH,
    )

    image_paths = []
    for index, prompt in enumerate(prompts):
        image_path = OUTPUT_DIR / f"scene_{index + 1}.jpg"
        download_scene_image(
            prompt,
            image_path,
            seed=int(time.time()) + index,
        )
        image_paths.append(str(image_path))

    audio_clip = None
    clips = []
    final_clip = None

    try:
        audio_clip = AudioFileClip(audio_path)
        if not audio_clip.duration or audio_clip.duration <= 0:
            raise ValueError("Generated audio has no usable duration.")

        scene_duration = audio_clip.duration / len(image_paths)

        for image_path in image_paths:
            clips.append(
                ImageClip(image_path).set_duration(scene_duration)
            )

        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip = final_clip.set_audio(audio_clip)

        final_clip.write_videofile(
            str(VIDEO_PATH),
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=2,
        )

    finally:
        if final_clip is not None:
            final_clip.close()
        for clip in clips:
            clip.close()
        if audio_clip is not None:
            audio_clip.close()

    if not VIDEO_PATH.exists() or VIDEO_PATH.stat().st_size == 0:
        raise RuntimeError("The MP4 video was not created successfully.")

    print(f"Video created: {VIDEO_PATH}")
    return str(VIDEO_PATH)


def post_to_tiktok(video_path, title):
    """Upload the MP4 using TikTok's Content Posting API."""

    if not TIKTOK_ACCESS_TOKEN:
        raise ValueError(
            "TIKTOK_ACCESS_TOKEN is missing from GitHub Secrets."
        )

    video_size = os.path.getsize(video_path)
    if video_size <= 0:
        raise ValueError("Cannot upload an empty video.")

    headers = {
        "Authorization": f"Bearer {TIKTOK_ACCESS_TOKEN}",
        "Content-Type": "application/json; charset=UTF-8",
    }

    init_url = "https://open.tiktokapis.com/v2/post/publish/video/init/"

    payload = {
        "post_info": {
            "title": title[:150],
            "privacy_level": "PUBLIC_TO_EVERYONE",
            "disable_duet": False,
            "disable_comment": False,
            "disable_stitch": False,
        },
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": video_size,
            "chunk_size": video_size,
            "total_chunk_count": 1,
        },
    }

    print("Requesting TikTok upload...")

    response = requests.post(
        init_url,
        headers=headers,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    result = response.json()
    error = result.get("error", {})

    if error.get("code") != "ok":
        raise RuntimeError(f"TikTok upload initialization failed: {result}")

    upload_url = result["data"]["upload_url"]
    publish_id = result["data"]["publish_id"]

    with open(video_path, "rb") as video_file:
        upload_response = requests.put(
            upload_url,
            headers={
                "Content-Type": "video/mp4",
                "Content-Range": f"bytes 0-{video_size - 1}/{video_size}",
                "Content-Length": str(video_size),
            },
            data=video_file,
            timeout=180,
        )

    upload_response.raise_for_status()

    print(
        "Video uploaded. TikTok publish processing may still be underway. "
        f"Publish ID: {publish_id}"
    )

    return {"publish_id": publish_id, "upload_status": "uploaded"}


def main():
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing from GitHub Secrets."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    comedy_data = generate_comedy_package(client)

    print("Generated TikTok title:", comedy_data["tiktok_title"])
    print("Generating video...")

    video_path = build_video(client, comedy_data)

    if not ENABLE_TIKTOK_POST:
        print(
            "Video generation completed. TikTok posting is disabled. "
            f"Video file: {video_path}"
        )
        return

    post_to_tiktok(video_path, comedy_data["tiktok_title"])


if __name__ == "__main__":
    main()
'''
path = Path("/mnt/data/amharic_comedy_bot_fixed.py")
path.write_text(code, encoding="utf-8")
print(f"Created: {path}\nLines: {len(code.splitlines())}\nBytes: {path.stat().st_size}")
