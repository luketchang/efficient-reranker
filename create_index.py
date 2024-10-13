import argparse
from pymilvus import connections, Collection

def recreate_index(client, collection_name, nlist):
    collection = Collection(collection_name)

    # Drop the existing index if it exists
    if collection.has_index():
        print(f"Dropping existing index for collection '{collection_name}'...")
        collection.drop_index()

    # Define the new index parameters
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": nlist}
    }

    # Create the new index
    print(f"Creating new index for collection '{collection_name}' with nlist={nlist}...")
    collection.create_index(field_name="vector", index_params=index_params)
    print(f"New index created successfully for collection '{collection_name}'.")

def main():
    parser = argparse.ArgumentParser(description='Recreate Milvus index with new nlist value')
    parser.add_argument('--collection_name', type=str, required=True, help='Milvus collection name')
    parser.add_argument('--nlist', type=int, default=4096, help='nlist value for the new index')
    parser.add_argument('--milvus_host', type=str, default='127.0.0.1', help='Milvus host')
    parser.add_argument('--milvus_port', type=str, default='19530', help='Milvus port')
    args = parser.parse_args()

    # Connect to Milvus
    print(f"Connecting to Milvus at {args.milvus_host}:{args.milvus_port}")
    connections.connect(host=args.milvus_host, port=args.milvus_port)

    # Recreate the index with the specified nlist
    recreate_index(connections, args.collection_name, args.nlist)

if __name__ == "__main__":
    main()
