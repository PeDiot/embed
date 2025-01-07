from typing import Dict
import json, requests
from PIL import Image


REQUESTS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def load_json(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, file_path: str) -> bool:
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False
    

def download_image_as_pil(url: str, timeout: int = 10) -> Image.Image:
    response = requests.get(url, stream=True, headers=REQUESTS_HEADERS, timeout=timeout)

    if response.status_code == 200:
        return Image.open(response.raw)