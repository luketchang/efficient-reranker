from pymilvus import connections, Collection
from embed_utils import get_embedding_model
import json

connections.connect(host="127.0.0.1", port="19530")
client = connections.connect()

embedding_model, _ = get_embedding_model("paraphrase-MiniLM-L6-v2")

collection_name = 'test_collection'
collection = Collection(collection_name)
collection.load()

query_vector = embedding_model.encode("Tell me about the Manhattan Project")

# Perform a similarity search
results = collection.search(
    data=[query_vector],
    anns_field="vector",  # Field where embeddings are stored
    param={"metric_type": "COSINE", "params": {"nprobe": 32}},
    limit=10,
    output_fields=["id", "passage"]  # Customize to return more fields if needed
)

# get the IDs of all returned hits
print(results[0].ids)

# get the distances to the query vector from all returned hits
print(results[0].distances)

for hit in results[0]:
    print(hit.entity)