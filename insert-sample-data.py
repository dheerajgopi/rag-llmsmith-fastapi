import json
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding


load_dotenv()

title_to_insert = "Harvard_University"

qdrant_url = os.getenv("QDRANT__URL")
qdrant_api_key = os.getenv("QDRANT__API_KEY")
qdrant_collection_name = os.getenv("QDRANT__COLLECTION_NAME")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

docs_for_embedding = []
with open("test-data/stanford-dataset-dev-v2.0.json", "r") as data_json_file:
    data_json: dict = json.load(data_json_file)
    data_by_title = next((d for d in data_json["data"] if d["title"] == title_to_insert))

    docs_for_embedding = [{"description": paragraph["context"]} for paragraph in data_by_title["paragraphs"]]

client.recreate_collection(
    collection_name=qdrant_collection_name,
    vectors_config=client.get_fastembed_vector_params().get("fast-bge-small-en"),
)

encoder = TextEmbedding("BAAI/bge-small-en")

client.upload_points(
    collection_name=qdrant_collection_name,
    points=[
        models.PointStruct(
            id=idx, vector=list(encoder.embed(doc["description"]))[0], payload=doc
        )
        for idx, doc in enumerate(docs_for_embedding)
    ],
)
