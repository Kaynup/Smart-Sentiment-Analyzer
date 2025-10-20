# utils.py
import argparse
from pymilvus import connections, utility

'''
Usage examples:-

- Drop a single collection:

python3 utils.py drop --collection texts_collection


- Drop all collections:

python3 utils.py drop-all


- Specify a Milvus host/port (if different from default):

python3 utils.py drop --collection texts_collection --host milvus-standalone --port 19530
'''


def connect_milvus(host: str = "localhost", port: str = "19530"):
    """
    Connects to Milvus server.
    """
    connections.connect(host=host, port=port)
    print(f"Connected to Milvus at {host}:{port}")

def drop_collection(collection_name: str):
    """
    Drops a specific collection if it exists.
    """
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Dropped collection: {collection_name}")
    else:
        print(f"Collection '{collection_name}' does not exist.")

def drop_all_collections():
    """
    Drops all collections in Milvus.
    """
    collections = utility.list_collections()
    if not collections:
        print("No collections found.")
        return

    for c in collections:
        utility.drop_collection(c)
        print(f"Dropped collection: {c}")

def main():
    parser = argparse.ArgumentParser(description="Milvus collection utilities")
    parser.add_argument(
        "action",
        choices=["drop", "drop-all"],
        help="Action to perform: drop a specific collection or drop all collections"
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Name of the collection to drop (required if action is 'drop')"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Milvus host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="19530",
        help="Milvus port (default: 19530)"
    )

    args = parser.parse_args()

    # Connect to Milvus
    connect_milvus(host=args.host, port=args.port)

    if args.action == "drop":
        if not args.collection:
            parser.error("The --collection argument is required when action is 'drop'.")
        drop_collection(args.collection)
    elif args.action == "drop-all":
        drop_all_collections()

if __name__ == "__main__":
    main()
