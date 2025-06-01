from typing import List, Dict
from PIL.Image import Image

import torch
from transformers import AutoModel, AutoProcessor
from src.utils import normalize_vectors


MODEL_NAME = "Marqo/marqo-fashionCLIP"


class FashionCLIPEncoder:
    def __init__(self, normalize: bool = False):
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.normalize = normalize

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        kwargs = {
            "padding": "max_length",
            "return_tensors": "pt",
            "truncation": True,
        }
        inputs = self.processor(text=texts, **kwargs)

        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in inputs.items()}
            return self._encode_text(batch)

    def encode_images(self, images: List[Image]) -> List[List[float]]:
        inputs = self.processor(images=images, return_tensors="pt")

        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in inputs.items()}
            return self._encode_images(batch)

    def _encode_text(self, batch: Dict) -> List[List[float]]:
        vectors = self.model.get_text_features(**batch).detach().cpu()
        return self._postprocess(vectors)

    def _encode_images(self, batch: Dict) -> List[List[float]]:
        vectors = self.model.get_image_features(**batch).detach().cpu()
        return self._postprocess(vectors)

    def _postprocess(self, vectors: torch.Tensor) -> List[List[float]]:
        vectors = torch.nan_to_num(vectors, nan=0.0)
        if self.normalize:
            vectors = normalize_vectors(vectors)

        return vectors.numpy().tolist()
