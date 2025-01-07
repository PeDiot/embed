from typing import List, Dict, Optional

import torch
from PIL.Image import Image
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "Marqo/marqo-fashionCLIP"


class FashionCLIPEncoder:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

        self.model.eval()
        self.device = self.model.device

    def encode_images(
        self, images: List[Image], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        if batch_size is None:
            batch_size = len(images)

        def transform_fn(el: Dict):
            return self.processor(
                images=[content for content in el["image"]], return_tensors="pt"
            )

        dataset = Dataset.from_dict({"image": images})
        dataset.set_format("torch")
        dataset.set_transform(transform_fn)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        image_embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                embeddings = self._encode_images(batch)
                image_embeddings.extend(embeddings)

        return image_embeddings

    def encode_text(
        self, text: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        if batch_size is None:
            batch_size = len(text)

        def transform_fn(el: Dict):
            kwargs = {
                "padding": "max_length",
                "return_tensors": "pt",
                "truncation": True,
            }
            return self.processor(text=el["text"], **kwargs)

        dataset = Dataset.from_dict({"text": text})
        dataset = dataset.map(
            function=transform_fn, batched=True, remove_columns=["text"]
        )
        dataset.set_format("torch")
        dataloader = DataLoader(dataset, batch_size=batch_size)
        text_embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                embeddings = self._encode_text(batch)
                text_embeddings.extend(embeddings)

        return text_embeddings

    def _encode_images(self, batch: Dict) -> List:
        return self.model.get_image_features(**batch).detach().cpu().numpy().tolist()

    def _encode_text(self, batch: Dict) -> List:
        return self.model.get_text_features(**batch).detach().cpu().numpy().tolist()
