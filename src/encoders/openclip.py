from typing import List, Union

import torch, open_clip
from PIL import Image


MODEL_NAME = "hf-hub:Marqo/marqo-fashionCLIP"


class OpenCLIPEncoder:
    def __init__(self):
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def encode(self, images: List[Union[str, Image.Image]]) -> List[List[float]]:
        batch = self._create_batch(images)

        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    embeddings = self.model.encode_image(batch)
            else:
                embeddings = self.model.encode_image(batch)

        return embeddings.cpu().numpy().tolist()

    def _create_batch(self, images: List[Image.Image]) -> torch.Tensor:        
        processed_images = []
        
        for image in images:
            processed_images.append(self.preprocess_val(image))
        
        return torch.stack(processed_images).to(self.device)