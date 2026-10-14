import json
import os
import time
import requests
from google import genai
from google.genai import types

# ==============================================================================
# 1. ቅንብሮች (Configuration & Environment Variables)
# ==============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TIKTOK_ACCESS_TOKEN = os.environ.get("TIKTOK_ACCESS_TOKEN")
# "DIRECT_POST" (ያለ ሰው ንክኪ ቀጥታ የሚለቅ) ወይም "INBOX_UPLOAD" (ወደ ቲክቶክ Inbox የሚልክ)
TIKTOK_POST_MODE = os.environ.get("TIKTOK_POST_MODE", "DIRECT_POST")
# Veo 3.1 ሞዴል (veo-3.1-lite-generate-preview፣ veo-3.1-fast-generate-preview ወይም veo-3.1-generate-preview)
VEO_MODEL = os.environ.get("VEO_MODEL", "veo-3.1-fast-generate-preview")

OUTPUT_VIDEO_PATH = "amharic_comedy_video.mp4"


def generate_comedy_prompt_and_caption(client: genai.Client) -> dict:
    """በGemini አማካኝነት አስቂኝ የአማርኛ ኮሜዲ የቪዲዮ+ድምጽ Prompt እና የቲክቶክ ርዕስ ያዘጋጃል።"""
    system_instruction = """
    አንተ የኢትዮጵያ (የአማርኛ) ቲክቶክ ኮሜዲ ጸሐፊ እና የVeo 3.1 ቪዲዮ ዳይሬክተር ነህ።
    ተግባርህ፦ ለ8 ሰከንድ የቲክቶክ ቪዲዮ የሚሆን በጣም አስቂኝ፣ የዕለት ተዕለት የኢትዮጵያውያንን ህይወት የሚያሳይ (ለምሳሌ፦ ታክሲ ውስጥ፣ ካፌ ውስጥ፣ ቤተሰብ ጋር፣ ወይም ጓደኛሞች መካከል የሚፈጠር) አጭር የኮሜዲ ትዕይንት መፍጠር ነው።

    ህጎች፦
    1. ቪዲዮው ምንም አይነት ጽሁፍ ወይም ሰብታይትል (No subtitles, no text on screen) እንዳይኖረው በPrompt ውስጥ በግልጽ አስቀምጥ።
    2. ምስሉ እና ድምጹ በአንድ ላይ በVeo 3.1 ስለሚፈጠሩ፣ የትዕይንቱን መግለጫ በእንግሊዝኛ ጽፈህ፣ ገጸ-ባህሪያቱ የሚናገሩትን አስቂኝ የአማርኛ ንግግር (Dialogue) በትዕምርተ ጥቅስ ("...") ውስጥ በአማርኛ እና በአነባበብ (Phonetic Amharic) አስቀምጥ፤ እንዲሁም የጀርባ ድምጾችን (ሳቅ፣ የከተማ ድምጽ) ግለጽ።
    3. መልስህን በJSON ቅርጸት ብቻ መልስ፦
       {
         "veo_prompt": "Full visual + native audio prompt for Veo 3.1 (strictly no subtitles or text on screen)",
         "tiktok_title": "አስቂኝ የአማርኛ ርዕስ ከነ ሃሽታጎቹ (#ethiopia #habesha #amhariccomedy #fyp)"
       }
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="ለዛሬ አዲስ እና በጣም አስቂኝ የአማርኛ ኮሜዲ የቲክቶክ ቪዲዮ Prompt እና ርዕስ አዘጋጅ።",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.9,
        ),
    )
    return json.loads(response.text)


def generate_all_in_one_video(client: genai.Client, veo_prompt: str, output_path: str) -> str:
    """በVeo 3.1 አማካኝነት ቪዲዮውን ከነ ድምጹ (Native Audio) በአንድ ጊዜ ያመነጫል።"""
    print(f"🎬 ቪዲዮ ከነ ድምጹ በ {VEO_MODEL} እየተዘጋጀ ነው...")
    print(f"📝 Prompt: {veo_prompt}")

    operation = client.models.generate_videos(
        model=VEO_MODEL,
        prompt=veo_prompt,
        config=types.GenerateVideosConfig(
            aspect_ratio="9:16",
        ),
    )

    while not operation.done:
        print("⏳ ቪዲዮው እና ድምጹ እየተዋሃዱ ነው... (10 ሰከንድ ይጠብቃል)")
        time.sleep(10)
        operation = client.operations.get(operation)

    generated_video = operation.response.generated_videos[0]
    client.files.download(file=generated_video.video, destination=output_path)
    print(f"✅ ቪዲዮው ከነ ድምጹ ተዘጋጅቶ በ '{output_path}' ተቀምጧል!")
    return output_path


def post_to_tiktok(video_path: str, title: str, mode: str = "DIRECT_POST") -> dict:
    """የተዘጋጀውን ቪዲዮ ወደ ቲክቶክ በቀጥታ ይለቃል (Direct Post) ወይም ወደ Inbox ይልካል።"""
    if not TIKTOK_ACCESS_TOKEN:
        raise ValueError("TIKTOK_ACCESS_TOKEN አልተገኘም! እባክህ በGitHub Secrets ውስጥ አስገባ።")

    video_size = os.path.getsize(video_path)
    headers = {
        "Authorization": f"Bearer {TIKTOK_ACCESS_TOKEN}",
        "Content-Type": "application/json; charset=UTF-8",
    }

    if mode == "DIRECT_POST":
        init_url = "https://open.tiktokapis.com/v2/post/publish/video/init/"
        payload = {
            "post_info": {
                "title": title[:150],
                "privacy_level": "PUBLIC_TO_EVERYONE",
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
                "video_cover_timestamp_ms": 1000,
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": video_size,
                "total_chunk_count": 1,
            },
        }
    else:
        init_url = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
        payload = {
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": video_size,
                "total_chunk_count": 1,
            }
        }

    print(f"🚀 ወደ ቲክቶክ መላክ ተጀምሯል (Mode: {mode})...")
    init_res = requests.post(init_url, headers=headers, json=payload, timeout=30)

    # Direct Post ገና Audit ካላለፈ በራሱ ወደ Inbox Upload ይቀይረዋል
    if init_res.status_code != 200 and mode == "DIRECT_POST":
        print("⚠️ Direct Post ፈቃድ ገና ስላልጸደቀ ወደ Inbox Upload እየተቀየረ ነው...")
        return post_to_tiktok(video_path, title, mode="INBOX_UPLOAD")

    init_data = init_res.json()
    if init_data.get("error", {}).get("code") != "ok":
        raise RuntimeError(f"TikTok Init Error: {init_data}")

    upload_url = init_data["data"]["upload_url"]
    publish_id = init_data["data"]["publish_id"]

    # የቪዲዮ ፋይሉን በPUT መጫን
    upload_headers = {
        "Content-Range": f"bytes 0-{video_size - 1}/{video_size}",
        "Content-Type": "video/mp4",
    }
    with open(video_path, "rb") as f:
        put_res = requests.put(upload_url, headers=upload_headers, data=f, timeout=120)
        put_res.raise_for_status()

    # የፖስቱን ሁኔታ ማረጋገጥ
    status_url = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
    status_res = requests.post(
        status_url,
        headers=headers,
        json={"publish_id": publish_id},
        timeout=30,
    )
    status_data = status_res.json()
    print(f"🎉 የቲክቶክ ውጤት: {status_data}")
    return status_data


def main():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY አልተገኘም! እባክህ በGitHub Secrets ውስጥ አስገባ።")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # ደረጃ 1፦ የአማርኛ ኮሜዲ Prompt እና ርዕስ ማመንጨት
    comedy_data = generate_comedy_prompt_and_caption(client)
    veo_prompt = comedy_data["veo_prompt"]
    tiktok_title = comedy_data["tiktok_title"]
    print(f"🏷️ የቲክቶክ ርዕስ: {tiktok_title}")

    # ደረጃ 2፦ ቪዲዮውን ከነ ድምጹ (ያለ ሰብታይትል) በአንድ ጊዜ መፍጠር
    video_file = generate_all_in_one_video(client, veo_prompt, OUTPUT_VIDEO_PATH)

    # ደረጃ 3፦ ወደ ቲክቶክ መልቀቅ
    post_to_tiktok(video_file, tiktok_title, mode=TIKTOK_POST_MODE)


if __name__ == "__main__":
    main()
