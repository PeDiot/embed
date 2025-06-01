from typing import Dict, List

import json, requests, torch
from PIL import Image


def load_json(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, file_path: str) -> bool:
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True

    except Exception as e:
        print(e)
        return False


def download_image_as_pil(url: str, timeout: int = 10) -> Image.Image:
    REQUESTS_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(
            url, stream=True, headers=REQUESTS_HEADERS, timeout=timeout
        )

        if response.status_code == 200:
            return Image.open(response.raw)

    except Exception as e:
        return


def normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(vectors, p=2, dim=1, keepdim=True)
    norms = torch.norm(vectors, p=2, dim=1, keepdim=True)
    norms = torch.where(norms > 1e-8, norms, torch.ones_like(norms))
    normalized_vectors = vectors / norms

    return normalized_vectors


def get_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, y.T)


def get_max_similarity_indices(similarity_matrix: torch.Tensor) -> List[int]:
    return torch.argmax(similarity_matrix, dim=1).tolist()
