import os
import time
import requests
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment.")

# Create OpenAI client
client = OpenAI(api_key=api_key, timeout=30)

# Output and logging
OUTPUT_DIR = Path("data/ar_generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = Path("data/prompt_log.txt")

# CelebA-style random prompt
def get_random_prompt():
    background = random.choice(["urban", "dim-indoor", "leafy", "abstract", "studio", "home"])
    lighting = random.choice(["natural daylight", "soft shadows", "evening light", "window lighting"])
    age = random.choice(["young adult", "middle-aged", "teenager"])
    gender = random.choice(["male", "female"])
    flaws = random.choice([
        "with visible pores and slight skin imperfections",
        "including uneven skin tone or natural shadows",
        "freckles or fine lines, non-retouched"
    ])
    return (
        f"Candid DSLR-style photo of a real human face, {age} {gender}. "
        f"The subject is centered and facing forward or 3/4. "
        f"Scene has {lighting}, and a {background} background. "
        f"Minimal retouching, visible skin texture, no stylization, {flaws}. "
        f"Include natural flaws like uneven lighting, stray hair strands, or mild shadows. "
        f"Do NOT include watermarks, borders, photo frames, or cameras. No makeup, not stylized or cartoon-like. Captured with soft focus and depth of field like a real lens."
    )

def download_with_retry(url, path, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
            return True
        except Exception as e:
            print(f"Download retry {attempt + 1} failed: {e}")
            time.sleep(2)
    print(f"Failed to download after {max_retries} attempts.")
    return False

def generate_image(prompt: str, index: int):
    filename = f"celeba_style_{index+1:03}.jpg"
    file_path = OUTPUT_DIR / filename

    if file_path.exists():
        print(f"[{index+1}] Already exists, skipping.")
        return

    try:
        print(f"[{index+1}] Generating...")
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        print(f"[{index+1}] URL: {image_url}")

        # Log prompt
        with open(LOG_FILE, "a") as log:
            log.write(f"{index+1}: {prompt}\n")

        # Download
        if download_with_retry(image_url, file_path):
            print(f"[{index+1}] Saved → {file_path}")
        else:
            print(f"[{index+1}] Download failed.")
    except Exception as e:
        print(f"[{index+1}] API error: {e}")


def main():
    TOTAL_IMAGES = 100
    MAX_WORKERS = 3 
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        prompt = get_random_prompt()
        futures = [executor.submit(generate_image, get_random_prompt(), i) for i in range(TOTAL_IMAGES)]
        for future in as_completed(futures):
            _ = future.result()

if __name__ == "__main__":
    main()