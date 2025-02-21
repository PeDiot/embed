from typing import List, Dict
from PIL.Image import Image

import torch
from transformers import AutoModel, AutoProcessor


MODEL_NAME = "Marqo/marqo-fashionCLIP"


class TransformersCLIPEncoder:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, images: List[Image]) -> List[List[float]]:
        kwargs = {
            "return_tensors": "pt",
        }
        inputs = self.processor(images=images, **kwargs)

        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in inputs.items()}
            return self._encode_images(batch)

    def _encode_images(self, batch: Dict) -> List[List[float]]:
        return self.model.get_image_features(**batch).detach().cpu().numpy().tolist()
