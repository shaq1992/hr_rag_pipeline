import httpx
import logging
from typing import List, Tuple
from core.config import settings

logger = logging.getLogger(__name__)

class InferenceClient:
    def __init__(self):
        # Persistent connection pooling handles high concurrency
        self.client = httpx.AsyncClient(
            base_url=settings.INFERENCE_API_URL,
            timeout=httpx.Timeout(30.0), # Accommodate heavy tensor operations
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=500)
        )

    async def get_embedding(self, text: str) -> Tuple[List[float], List[int], List[float]]:
        """Calls the /embed endpoint."""
        response = await self.client.post("/embed", json={"text": text})
        response.raise_for_status()
        data = response.json()
        return data["dense_vector"], data["sparse_indices"], data["sparse_values"]

    async def get_rerank_scores(self, query: str, documents: List[str]) -> List[float]:
        """Calls the /rerank endpoint."""
        if not documents:
            return []
            
        response = await self.client.post(
            "/rerank", 
            json={"query": query, "documents": documents}
        )
        response.raise_for_status()
        return response.json()["scores"]

    async def close(self):
        await self.client.aclose()

# Singleton instance
inference_client = InferenceClient()
