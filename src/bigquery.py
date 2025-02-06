from typing import List, Dict, Optional

from google.oauth2 import service_account
from google.cloud import bigquery
from .enums import *


BASE_QUERY = f"""
WITH tab AS (
SELECT 
item.*, 
image.url AS image_location, 
category.category_type AS category_type,
catalog.women AS women,
ROW_NUMBER() OVER (PARTITION BY item.vinted_id ORDER BY item.vinted_id DESC) AS row_num
FROM `{PROJECT_ID}.{DATASET_ID}.{ITEM_TABLE_ID}` item
LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.{IMAGE_TABLE_ID}` image USING (vinted_id)
LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.{CATEGORY_TABLE_ID}` category USING (catalog_id)
LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.{CATALOG_TABLE_ID}` catalog ON item.catalog_id = catalog.id
WHERE 
item.id NOT IN (SELECT item_id FROM `{PROJECT_ID}.{DATASET_ID}.{PINECONE_TABLE_ID}`)
AND item.vinted_id NOT IN (SELECT vinted_id FROM `{PROJECT_ID}.{DATASET_ID}.{SOLD_TABLE_ID}`)
)
SELECT * EXCEPT (row_num) FROM tab 
WHERE row_num = 1
"""


def init_client(credentials_dict: Dict) -> bigquery.Client:
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict
    )

    return bigquery.Client(
        credentials=credentials, project=credentials_dict["project_id"]
    )


def load_items_to_embed(
    client: bigquery.Client,
    shuffle: bool = False,
    n: Optional[int] = None,
    shard_index: Optional[int] = None,
    total_shards: Optional[int] = None,
) -> bigquery.table.RowIterator:
    query = _query_items_to_embed(shuffle, n, shard_index, total_shards)

    return client.query(query).result()


def upload(
    client: bigquery.Client, dataset_id: str, table_id: str, rows: List[Dict]
) -> bool:
    if len(rows) == 0:
        return False

    output = client.insert_rows_json(
        table=f"{PROJECT_ID}.{DATASET_ID}.{table_id}", json_rows=rows
    )

    return len(output) == 0


def _query_items_to_embed(
    shuffle: bool = False,
    n: Optional[int] = None,
    shard_index: Optional[int] = None,
    total_shards: Optional[int] = None,
) -> str:
    query = BASE_QUERY

    if shard_index is not None and total_shards is not None:
        query += f" AND MOD(FARM_FINGERPRINT(CAST(vinted_id AS STRING)), {total_shards}) = {shard_index}"

    if shuffle:
        query += " ORDER BY RAND()"

    if (shard_index is None or total_shards is not None) and n is not None:
        query += f" LIMIT {n}"

    return query
