from pymilvus import MilvusClient, connections
connections.connect(host='127.0.0.1', port='19530')
client = MilvusClient(f"http://127.0.0.1:19530")
print(client.list_collections())
print(client.list_indexes(collection_name="hotpot_corpus"))
