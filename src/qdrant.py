from typing import List, Dict, Tuple
from qdrant_client import QdrantClient, models
from .enums import QDRANT_COLLECTION_NAME, QDRANT_PAYLOAD_FIELDS


def prepare(
    point_ids: List[str], payloads: List[Dict], embeddings: List[float]
) -> Tuple[List[models.PointStruct], Dict]:
    points, rows = [], []

    for point_id, payload, embedding in zip(point_ids, payloads, embeddings):
        if payload.get("id"):
            point = models.PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
            row = {"item_id": payload.get("id"), "point_id": point.id}

            points.append(point)
            rows.append(row)

    return points, rows


def upload(client: QdrantClient, points: models.PointStruct) -> bool:
    try:
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
        return True
    except:
        return False


def update_availability(
    client: QdrantClient, item_ids: List[str], availability_status: bool = False
) -> bool:
    try:
        must_condition = models.FieldCondition(
            key="id",
            match=models.MatchAny(any=item_ids),
        )
        client.set_payload(
            collection_name=QDRANT_COLLECTION_NAME,
            payload={"is_available": availability_status},
            points=models.Filter(must=[must_condition]),
        )

        return True

    except Exception as e:
        return False


def create_payload_index(
    client: QdrantClient,
    field_name: str,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> bool:
    if field_name == "women":
        field_schema = models.PayloadSchemaType.BOOL
    elif field_name in ("category_type", "url"):
        field_schema = models.PayloadSchemaType.KEYWORD
    elif field_name == "price":
        field_schema = models.PayloadSchemaType.FLOAT
    else:
        raise ValueError(f"Invalid field_name, muste be in {QDRANT_PAYLOAD_FIELDS}.")

    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
        return True

    except Exception as e:
        print(e)
        return False
