from typing import List, Dict, Tuple, Optional
import pinecone


def prepare(
    point_ids: List[str], payloads: List[Dict], embeddings: List[float]
) -> Tuple[List[Dict], Dict]:
    vectors, rows = [], []

    for point_id, payload, embedding in zip(point_ids, payloads, embeddings):
        payload = _create_payload(payload)

        if payload is None:
            continue

        vector = {"id": point_id, "values": embedding, "metadata": payload}
        row = {"item_id": payload.get("id"), "point_id": point_id}

        vectors.append(vector)
        rows.append(row)

    return vectors, rows


def upload(index: pinecone.Index, vectors: List[Dict]) -> bool:
    if len(vectors) == 0:
        return False

    try:
        index.upsert(vectors=vectors)
        return True
    except:
        return False


def _create_payload(payload: Dict) -> Optional[Dict]:
    if payload.get("id") and payload.get("url") is not None:
        if not payload.get("category_type"): 
            payload["category_type"] = ""
        
        payload["is_available"] = True

        return payload