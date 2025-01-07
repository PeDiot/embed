import sys

sys.path.append("../")

import uuid, tqdm
from qdrant_client import QdrantClient
import src


SECRETS_PATH = "secrets.json"
BATCH_SIZE = 32


def main():
    secrets = src.utils.load_json(file_path=SECRETS_PATH)

    bq_client = src.bigquery.init_client(credentials_dict=secrets.get("GCP_CREDENTIALS"))
    qdrant_client = QdrantClient(api_key=secrets.get("QDRANT_API_KEY"), url=secrets.get("QDRANT_URL"))
    encoder = src.encoder.FashionCLIPEncoder()

    loader = src.bigquery.load_items_to_embed(client=bq_client, shuffle=True)

    n_success, n = 0, 0
    point_ids, images, payloads = [], [], []
    loop = tqdm.tqdm(iterable=loader, total=loader.total_rows)

    for row in loop:
        row = dict(row)
        point_id = str(uuid.uuid4())
        image = src.utils.download_image_as_pil(url=row.get("image_location"))

        images.append(image)
        payloads.append(row)
        point_ids.append(point_id)

        if len(point_ids) > 0 and len(point_ids) % BATCH_SIZE == 0:
            n += len(point_ids)
            
            try:
                embeddings = encoder.encode_images(images)
            except: 
                continue
            
            points, rows = src.qdrant.prepare(
                point_ids=point_ids, 
                payloads=payloads, 
                embeddings=embeddings
            )
            
            if src.qdrant.upload(client=qdrant_client, points=points):
                if src.bigquery.upload(
                    client=bq_client, 
                    dataset_id=src.enums.DATASET_ID, 
                    table_id=src.enums.QDRANT_TABLE_ID, 
                    rows=rows
                ):
                    n_success += len(point_ids)

            else:
                valid_rows = [] 

                for point, row in zip(points, rows):
                    success = src.qdrant.upload(client=qdrant_client, points=[point])

                    if success:
                        valid_rows.append(row)

                if src.bigquery.upload(
                    client=bq_client, 
                    dataset_id=src.enums.DATASET_ID, 
                    table_id=src.enums.QDRANT_TABLE_ID, 
                    rows=rows
                ):
                    n_success += len(point_ids)

            point_ids, images, payloads = [], [], []

            loop.set_description(
                f"Success rate: {n_success / n:.2f} | "
                f"Processed: {n} | "
                f"Inserted: {n_success} | "
            )


if __name__ == "__main__":
    main()