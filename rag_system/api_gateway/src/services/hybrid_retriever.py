import logging
from typing import List
from dto.response import RetrievedChunk
from integration.inference_client import inference_client
from integration.qdrant_client import qdrant_client
from core.config import settings

logger = logging.getLogger(__name__)

class HybridRetriever:
    @staticmethod
    async def retrieve(query: str) -> List[RetrievedChunk]:
        logger.info("Fetching embeddings for hybrid search...")
        dense, s_idx, s_val = await inference_client.get_embedding(query)
        
        logger.info("Executing Qdrant RRF hybrid search...")
        chunks = await qdrant_client.hybrid_search(
            dense_vec=dense, 
            sparse_idx=s_idx, 
            sparse_val=s_val, 
            limit=settings.RETRIEVAL_K
        )
        return chunks
