from pymilvus import MilvusClient, connections
import argparse

def main():
    parser = argparse.ArgumentParser(description="Drop a collection from Milvus.")
    parser.add_argument('--milvus_uri', type=str, default='127.0.0.1:19530', help='Milvus uri')
    parser.add_argument('--milvus_token', type=str, default=None, help='Milvus token')

    args = parser.parse_args()

    print(f"Connecting to Milvus at {args.milvus_uri}")
    connections.connect(alias="default", uri=args.milvus_uri, token=args.milvus_token)
    client = MilvusClient(uri=args.milvus_uri, token=args.milvus_token)

    print(client.list_collections())
    print(client.list_indexes())

if __name__ == "__main__":
    main()