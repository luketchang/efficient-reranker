import argparse
from pymilvus import utility, connections

def drop_collection(collection_name):
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"Collection '{collection_name}' dropped successfully.")
        else:
            print(f"Collection '{collection_name}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Drop a collection from Milvus.")
    parser.add_argument("--collection_name", type=str, help="Name of the collection to drop.")
    parser.add_argument('--milvus_uri', type=str, default='127.0.0.1:19530', help='Milvus uri')
    parser.add_argument('--milvus_token', type=str, default=None, help='Milvus token')

    args = parser.parse_args()

    print(f"Connecting to Milvus at {args.milvus_uri}")
    connections.connect(alias="default", uri=args.milvus_uri, token=args.milvus_token)

    drop_collection(args.collection_name)

if __name__ == "__main__":
    main()
