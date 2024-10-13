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
    parser.add_argument('--milvus_host', type=str, default='127.0.0.1', help='Milvus host')
    parser.add_argument('--milvus_port', type=str, default='19530', help='Milvus port')

    args = parser.parse_args()

    connections.connect(host=args.milvus_host, port=args.milvus_port)
    drop_collection(args.collection_name)

if __name__ == "__main__":
    main()
