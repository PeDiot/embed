from typing import List, Dict, Any, Union

from google.oauth2 import service_account
from google.cloud import bigquery
from .enums import *


def init_client(credentials_dict: Dict) -> bigquery.Client:
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict
    )

    return bigquery.Client(
        credentials=credentials, project=credentials_dict["project_id"]
    )


def load_items_to_embed(
    client: bigquery.Client, 
    dataset_id: str = DATASET_ID, 
    shuffle: bool = False,
    shard_index: int = 0,
    total_shards: int = 1
) -> bigquery.table.RowIterator:
    query = _query_items_to_embed(dataset_id, shuffle, shard_index, total_shards)
    return client.query(query).result()


def upload(
    client: bigquery.Client, dataset_id: str, table_id: str, rows: List[Dict]
) -> bool:
    if len(rows) == 0:
        return False

    output = client.insert_rows_json(
        table=f"{PROJECT_ID}.{dataset_id}.{table_id}", json_rows=rows
    )

    return len(output) == 0


def _query_items_to_embed(
    dataset_id: str, 
    shuffle: bool = False, 
    shard_index: int = 0,
    total_shards: int = 1
) -> str:
    query = f"""
    WITH tab AS (
    SELECT 
    item.*, 
    image.url AS image_location, 
    category.category_type AS category_type,
    catalog.women AS women,
    ROW_NUMBER() OVER (PARTITION BY item.vinted_id ORDER BY item.vinted_id DESC) AS row_num
    FROM `{PROJECT_ID}.{dataset_id}.{ITEM_TABLE_ID}` item
    LEFT JOIN `{PROJECT_ID}.{dataset_id}.{IMAGE_TABLE_ID}` image USING (vinted_id)
    LEFT JOIN `{PROJECT_ID}.{dataset_id}.{CATEGORY_TABLE_ID}` category USING (catalog_id)
    LEFT JOIN `{PROJECT_ID}.{dataset_id}.{CATALOG_TABLE_ID}` catalog ON item.catalog_id = catalog.id
    WHERE 
    item.id NOT IN (SELECT item_id FROM `{PROJECT_ID}.{dataset_id}.{PINECONE_TABLE_ID}`)
    AND item.vinted_id NOT IN (SELECT vinted_id FROM `{PROJECT_ID}.{dataset_id}.{SOLD_TABLE_ID}`)
    )
    SELECT * FROM tab 
    WHERE row_num = 1
    AND MOD(FARM_FINGERPRINT(CAST(vinted_id AS STRING)), {total_shards}) = {shard_index}
    """

    if shuffle:
        query += " ORDER BY RAND()"

    return query