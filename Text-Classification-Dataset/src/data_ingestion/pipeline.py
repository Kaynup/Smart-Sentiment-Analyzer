import os
from . import config
from .loader import stream_texts
from .preprocessor import normalize_texts
from .embedder import Embedder
from .milvus_client import connect, create_collection, create_index, load_collection, insert_batch
from .utils import batch_iterator

def run_ingestion():
    os.makedirs(config.EXPORT_DIR, exist_ok=True)
    connect()
    collection = create_collection()

    embedder = Embedder(config.EMBED_MODEL_NAME)

    text_iter = stream_texts(config.CSV_FILES)
    id_counter = 1

    for batch in batch_iterator(text_iter, config.BATCH_SIZE):
        batch = normalize_texts(batch)
        ids = list(range(id_counter, id_counter + len(batch)))
        id_counter += len(batch)

        embeds = embedder.embed_batch(batch, normalize=True)
        insert_batch(collection, ids, batch, embeds)
        print(f"Inserted batch: {ids[0]} - {ids[-1]}")

    create_index(collection)
    load_collection(collection)
    print("===================")
    print("Ingestion complete.")
    print("===================")

if __name__ == '__main__':
    run_ingestion()
