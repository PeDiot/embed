import sys
import gc
from typing import Optional, Tuple

sys.path.append("../")

import uuid, tqdm, json, os
from PIL import Image
from pinecone import Pinecone
import src


BATCH_SIZE = 128
NUM_ITEMS = None


def get_shard_params() -> Tuple[Optional[int], Optional[int]]:
    shard_index = int(os.getenv("SHARD_INDEX", -1))
    total_shards = int(os.getenv("TOTAL_SHARDS", -1))

    if shard_index == -1 or total_shards == -1:
        return None, None

    return shard_index, total_shards


def main():
    shard_index, total_shards = get_shard_params()
    secrets = json.loads(os.getenv("SECRETS_JSON"))

    gcp_credentials = secrets.get("GCP_CREDENTIALS")
    gcp_credentials["private_key"] = gcp_credentials["private_key"].replace("\\n", "\n")
    bq_client = src.bigquery.init_client(credentials_dict=gcp_credentials)

    pc_client = Pinecone(api_key=secrets.get("PINECONE_API_KEY"))
    pinecone_index = pc_client.Index(src.enums.PINECONE_INDEX_NAME)
    encoder = src.encoder.FashionCLIPEncoder()

    loader = src.bigquery.load_items_to_embed(
        client=bq_client,
        shuffle=True,
        n=NUM_ITEMS,
        shard_index=shard_index,
        total_shards=total_shards,
    )

    n_success, n = 0, 0
    point_ids, images, payloads = [], [], []
    loop = tqdm.tqdm(iterable=loader, total=loader.total_rows)

    for row in loop:
        row = dict(row)
        image = src.utils.download_image_as_pil(url=row.get("image_location"))

        if isinstance(image, Image.Image):
            point_id = str(uuid.uuid4())

            images.append(image)
            payloads.append(row)
            point_ids.append(point_id)

        if len(point_ids) > 0 and len(point_ids) % BATCH_SIZE == 0:
            n += len(point_ids)

            try:
                embeddings = encoder.encode_images(images)
            except Exception as e:
                print(f"Encoding error: {e}")
                continue

            points, rows = src.pinecone.prepare(
                point_ids=point_ids, payloads=payloads, embeddings=embeddings
            )

            if src.pinecone.upload(index=pinecone_index, vectors=points):
                if src.bigquery.upload(
                    client=bq_client,
                    dataset_id=src.enums.DATASET_ID,
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
                    dataset_id=src.enums.DATASET_ID,
                    table_id=src.enums.PINECONE_TABLE_ID,
                    rows=valid_rows,
                ):
                    n_success += len(valid_rows)

            point_ids, images, payloads = [], [], []
            gc.collect()

            loop.set_description(
                f"Success rate: {n_success / n:.2f} | "
                f"Processed: {n} | "
                f"Inserted: {n_success} | "
            )


if __name__ == "__main__":
    main()
