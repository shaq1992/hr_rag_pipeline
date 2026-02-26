import logging
from typing import List, Tuple
from dto.response import RetrievedChunk
from integration.inference_client import inference_client
from core.config import settings

logger = logging.getLogger(__name__)

class ContextEvaluator:
    @staticmethod
    async def evaluate_and_rerank(query: str, chunks: List[RetrievedChunk]) -> Tuple[List[RetrievedChunk], bool]:
        """
        Re-ranks chunks and determines if we have enough confidence to answer.
        Returns a tuple: (sorted_filtered_chunks, is_confident)
        """
        if not chunks:
            return [], False

        documents = [chunk.content for chunk in chunks]
        
        logger.info("Calling inference service for cross-encoder re-ranking...")
        scores = await inference_client.get_rerank_scores(query, documents)

        # Attach scores and filter
        valid_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk.score = score
            if score >= settings.RERANK_THRESHOLD:
                valid_chunks.append(chunk)

        # Sort descending by score
        valid_chunks.sort(key=lambda x: x.score, reverse=True)
        
        # We define confidence simply: do we have at least one chunk passing the threshold?
        is_confident = len(valid_chunks) > 0
        
        logger.info(f"Reranking complete. {len(valid_chunks)} chunks passed threshold.")
        return valid_chunks, is_confident
