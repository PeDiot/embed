from typing import List, Dict, Optional

import torch
from PIL.Image import Image
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModel, AutoProcessor


MODEL_NAME = "Marqo/marqo-fashionCLIP"


class TransformersCLIPEncoder:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

        self.model.eval()
        self.device = self.model.device

    def encode(
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

    def _encode_images(self, batch: Dict) -> List:
        return self.model.get_image_features(**batch).detach().cpu().numpy().tolist()