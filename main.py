import sys

sys.path.append("/app")

from typing import Optional, Tuple, Dict

import uuid, tqdm, json, os, random, gc
from PIL import Image
from google.cloud import bigquery
from pinecone import Pinecone
import src


BATCH_SIZE = 128
NUM_ITEMS = None
SHUFFLE_ALPHA = 0.3


def get_shard_params() -> Tuple[Optional[int], Optional[int]]:
    shard_index = int(os.getenv("SHARD_INDEX", -1))
    total_shards = int(os.getenv("TOTAL_SHARDS", -1))

    if shard_index == -1 or total_shards == -1:
        return None, None

    return shard_index, total_shards


def get_gcp_credentials() -> Dict:
    gcp_credentials = secrets.get("GCP_CREDENTIALS")
    gcp_credentials["private_key"] = gcp_credentials["private_key"].replace("\\n", "\n")

    return gcp_credentials


def get_dataloader() -> bigquery.table.RowIterator:
    shuffle = random.random() < SHUFFLE_ALPHA

    return src.bigquery.load_items_to_embed(
        client=bq_client,
        shuffle=shuffle,
        n=NUM_ITEMS,
        shard_index=shard_index,
        total_shards=total_shards,
    )


def main():
    global shard_index, total_shards
    shard_index, total_shards = get_shard_params()

    global secrets
    secrets = json.loads(os.getenv("SECRETS_JSON"))

    global bq_client
    gcp_credentials = get_gcp_credentials()
    bq_client = src.bigquery.init_client(credentials_dict=gcp_credentials)

    pc_client = Pinecone(api_key=secrets.get("PINECONE_API_KEY"))
    pinecone_index = pc_client.Index(src.enums.PINECONE_INDEX_NAME)
    encoder = src.encoder.FashionCLIPEncoder(normalize=True)

    n_success, n = 0, 0
    index, point_ids, images, payloads, to_delete_ids = [], [], [], [], []

    loader = get_dataloader()
    loop = tqdm.tqdm(iterable=loader, total=loader.total_rows)

    for row in loop:
        row = dict(row)
        vinted_id = row.get("vinted_id")

        if vinted_id in index:
            continue

        index.append(vinted_id)
        image_url = row.get("image_location")
        image = src.utils.download_image_as_pil(url=image_url)

        if isinstance(image, Image.Image):
            point_id = str(uuid.uuid4())

            images.append(image)
            payloads.append(row)
            point_ids.append(point_id)

        else:
            to_delete_ids.append(vinted_id)

        if len(point_ids) > 0 and len(point_ids) % BATCH_SIZE == 0:
            n += len(point_ids)

            try:
                embeddings = encoder.encode(images)
            except Exception as e:
                print(f"Encoding error: {e}")
                continue

            points, rows = src.pinecone.prepare(
                point_ids=point_ids, payloads=payloads, embeddings=embeddings
            )

            if src.pinecone.upload(index=pinecone_index, vectors=points):
                if src.bigquery.upload(
                    client=bq_client,
                    table_id=src.enums.PINECONE_TABLE_ID,
                    rows=rows,
                ):
                    n_success += len(point_ids)

            else:
                valid_rows = []

                for point, row in zip(points, rows):
                    success = src.pinecone.upload(index=pinecone_index, vectors=[point])

                    if success:
                        valid_rows.append(row)

                if src.bigquery.upload(
                    client=bq_client,
                    table_id=src.enums.PINECONE_TABLE_ID,
                    rows=valid_rows,
                ):
                    n_success += len(valid_rows)

            point_ids, images, payloads = [], [], []
            gc.collect()

        success_rate = 0 if n == 0 else n_success / n

        loop.set_description(
            f"Success rate: {success_rate:.2f} | "
            f"Processed: {n} | "
            f"Inserted: {n_success} | "
        )

    if len(to_delete_ids) > 0:
        to_delete_ids = ", ".join([f"'{vinted_id}'" for vinted_id in to_delete_ids])
        conditions = f"vinted_id IN ({to_delete_ids})"

        if src.bigquery.delete(
            client=bq_client, table_id=src.enums.ITEM_TABLE_ID, conditions=conditions
        ):
            print(f"Deleted {len(to_delete_ids)} items from {src.enums.ITEM_TABLE_ID}")


if __name__ == "__main__":
    main()
