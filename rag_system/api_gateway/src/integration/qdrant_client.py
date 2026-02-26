import logging
from qdrant_client import AsyncQdrantClient, models
from core.config import settings
from dto.response import RetrievedChunk

logger = logging.getLogger(__name__)

class QdrantIntegration:
    def __init__(self):
        self.client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    async def hybrid_search(self, dense_vec: list, sparse_idx: list, sparse_val: list, limit: int = settings.RETRIEVAL_K) -> list[RetrievedChunk]:
        """Executes a hybrid (Dense + Sparse) search using RRF in Qdrant."""
        
        # We use prefetching to combine sparse and dense results
        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=limit * 2,
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_idx, values=sparse_val),
                using="sparse",
                limit=limit * 2,
            ),
        ]

        # Fusion query combines the prefetched results
        results = await self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=limit,
        )

        parsed_chunks = []
        for point in results.points:
            payload = point.payload or {}
            parsed_chunks.append(RetrievedChunk(
                id=str(point.id),
                content=payload.get("raw_content", ""),
                source_document=payload.get("source_document", "Unknown"),
                section_header=payload.get("section_header", "Unknown"),
                page_number=str(payload.get("page_number", "Unknown")),
                score=point.score
            ))
            
        return parsed_chunks

    async def close(self):
        await self.client.close()

qdrant_client = QdrantIntegration()
