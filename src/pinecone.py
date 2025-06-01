from typing import List, Dict, Tuple
from collections import defaultdict
import pinecone


def prepare(
    point_ids: List[str], payloads: List[Dict], embeddings: List[float]
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    vectors, rows = defaultdict(list), defaultdict(list)

    for point_id, payload, embedding in zip(point_ids, payloads, embeddings):
        if _is_valid_payload(payload):
            category_type = payload.get("category_type")

            vector = _create_vector(point_id, payload, embedding)
            row = _create_row(vector)

            vectors[category_type].append(vector)
            rows[category_type].append(row)

    return vectors, rows


def upload(index: pinecone.Index, vectors: List[Dict], namespace: str) -> bool:
    if len(vectors) == 0:
        return False

    try:
        index.upsert(vectors=vectors, namespace=namespace)
        return True
    except Exception as e:
        print(e)
        return False


def _create_vector(point_id: str, payload: Dict, embedding: List[float]) -> Dict:
    return {"id": point_id, "values": embedding, "metadata": payload}


def _create_row(vector: Dict) -> Dict:
    return {
        "item_id": vector.get("metadata").get("id"),
        "point_id": vector.get("id"),
        "namespace": vector.get("metadata").get("category_type"),
    }


def _is_valid_payload(payload: Dict) -> bool:
    if not payload.get("id"):
        return False

    if not payload.get("vinted_id"):
        return False

    if not payload.get("url"):
        return False

    if not payload.get("image_location"):
        return False

    return True
