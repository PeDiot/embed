from typing import List, Dict, Tuple
import pinecone


def prepare(
    point_ids: List[str], payloads: List[Dict], embeddings: List[float]
) -> Tuple[List[Dict], Dict]:
    vectors, rows = [], []

    for point_id, payload, embedding in zip(point_ids, payloads, embeddings):
        if payload.get("id"):
            payload["is_available"] = True
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
