from qdrant_client import QdrantClient, models
import os

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
client = QdrantClient(host=QDRANT_HOST, port=6333)

def init_collection(collection_name: str):
    """Initializes the collection with dense and sparse vector configurations and payload indexes."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            }
        )
        
        # Hardware-accelerated payload filtering
        client.create_payload_index(collection_name, "source_document", field_schema="keyword")
        client.create_payload_index(collection_name, "sub_section", field_schema="keyword")

def upsert_chunk(collection_name: str, point_id: str, dense: list, sparse_idx: list, sparse_val: list, payload: dict):
    """Upserts a hybrid point into Qdrant."""
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(indices=sparse_idx, values=sparse_val)
                },
                payload=payload
            )
        ]
    )
