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
    client: bigquery.Client, dataset_id: str = DATASET_ID, shuffle: bool = False
) -> bigquery.table.RowIterator:
    query = f"""
    SELECT 
    item.*, 
    image.url AS image_location, 
    category.category_type AS category_type,
    catalog.women AS women
    FROM `{PROJECT_ID}.{dataset_id}.{ITEM_TABLE_ID}` item
    LEFT JOIN `{PROJECT_ID}.{dataset_id}.{IMAGE_TABLE_ID}` image USING (vinted_id)
    LEFT JOIN `{PROJECT_ID}.{dataset_id}.{CATEGORY_TABLE_ID}` category USING (catalog_id)
    LEFT JOIN `{PROJECT_ID}.{dataset_id}.{CATALOG_TABLE_ID}` catalog ON item.catalog_id = catalog.id
    WHERE 
    item.id NOT in (SELECT item_id FROM `{PROJECT_ID}.{dataset_id}.{PINECONE_TABLE_ID}`)
    AND item.is_available = TRUE
    """

    if shuffle:
        query += " ORDER BY RAND()"

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
