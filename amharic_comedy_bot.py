
import json
import os
import time
import wave
import urllib.parse
from pathlib import Path

import requests
from PIL import Image
from google import genai
from google.genai import types
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TIKTOK_ACCESS_TOKEN = os.environ.get("TIKTOK_ACCESS_TOKEN")

# Posting stays OFF until you explicitly enable it.
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
    """Generate an Amharic comedy script and two image prompts."""

    instruction = """
You are an Ethiopian Amharic comedy writer.

Create a short, funny, original conversation between two Ethiopian
characters in a familiar setting such as a taxi, cafe, family home,
or among friends in Addis Ababa.

Return valid JSON only, using exactly these keys:
{
  "dialogue_script": "Speaker1: ...\\nSpeaker2: ...",
  "scene_prompts": [
    "Detailed photorealistic Ethiopian comedy scene, vertical portrait",
    "Detailed photorealistic Ethiopian comedy punchline scene, vertical portrait"
  ],
  "tiktok_title": "Short Amharic title with relevant hashtags"
}

Rules:
- Dialogue must be in natural Amharic.
- Make it funny, concise, and suitable for a short video.
- Include exactly two speakers named Speaker1 and Speaker2.
- No subtitles, captions, signs, written words, or watermarks in images.
- Image prompts should depict Ethiopian adults and the same characters
  consistently across both scenes.
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

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(audio_bytes)

    print(f"Audio saved: {wav_path}")
    return str(wav_path)


def download_scene_image(prompt, output_path, seed):
    """Request a scene image from the external Pollinations service."""

    encoded_prompt = urllib.parse.quote(
        prompt
        + ", photorealistic, Ethiopian people, portrait 9:16, "
          "no text, no letters, no subtitles, no watermark"
    )

    url = (
        "https://image.pollinations.ai/prompt/"
        f"{encoded_prompt}?width={WIDTH}&height={HEIGHT}"
        f"&seed={seed}&nologo=true"
    )

    print(f"Generating scene image {seed}...")

    response = requests.get(
        url,
        timeout=(20, 180),
        headers={"User-Agent": "AmharicComedyBot/1.0"},
    )
    response.raise_for_status()

    if not response.headers.get("Content-Type", "").startswith("image/"):
        raise RuntimeError(
            "Image service did not return an image. Try again later."
        )

    with open(output_path, "wb") as image_file:
        image_file.write(response.content)

    # Validate and convert the returned image to RGB.
    with Image.open(output_path) as image:
        image.convert("RGB").resize((WIDTH, HEIGHT)).save(output_path, "JPEG")

    return str(output_path)


def build_video(client, comedy_data):
    """Combine generated scene images and audio into a vertical MP4."""

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

    # Do not silently fall back to a different publishing mode.
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

    # Keep posting disabled unless explicitly enabled in repository settings.
    if not ENABLE_TIKTOK_POST:
        print(
            "Video generation completed. TikTok posting is disabled. "
            f"Video file: {video_path}"
        )
        return

    post_to_tiktok(video_path, comedy_data["tiktok_title"])


if __name__ == "__main__":
    main()
