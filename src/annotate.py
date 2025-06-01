from typing import List, Tuple, Optional

import torch

from .models import ColorVector
from .utils import (
    load_json,
    normalize_vectors,
    get_cosine_similarity,
    get_max_similarity_indices,
)


def load_color_vectors(path: str) -> Tuple[List[int], torch.Tensor]:
    data = load_json(path)

    vectors = [ColorVector.from_dict(entry) for entry in data]
    color_ids = [vector.id for vector in vectors]
    color_tensor = torch.tensor([vector.values for vector in vectors])

    return color_ids, color_tensor


def get_color_ids(
    color_ids: List[int],
    color_tensor: torch.Tensor,
    vectors: List[List[float]],
    normalize: bool = False,
) -> Optional[List[int]]:
    try:
        input_tensor = torch.tensor(vectors)

        if normalize:
            input_tensor = normalize_vectors(input_tensor)
            color_tensor = normalize_vectors(color_tensor)

        similarity_matrix = get_cosine_similarity(input_tensor, color_tensor)
        max_similarity_indices = get_max_similarity_indices(similarity_matrix)

        return [color_ids[idx] for idx in max_similarity_indices]

    except Exception as e:
        print(e)
        return None
