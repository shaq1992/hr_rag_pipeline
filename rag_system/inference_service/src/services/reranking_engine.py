import logging
from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)

class RerankingEngine:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RerankingEngine, cls).__new__(cls)
        return cls._instance

    def get_model(self):
        if self._model is None:
            logger.info("Loading BGE-Reranker-Base into memory (CPU)...")
            self._model = FlagReranker('BAAI/bge-reranker-base', use_fp16=False)
            logger.info("BGE-Reranker loaded.")
        return self._model

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
            
        model = self.get_model()
        pairs = [[query, doc] for doc in documents]
        
        scores = model.compute_score(pairs)
        
        # If only one document is passed, compute_score returns a single float instead of a list.
        if isinstance(scores, float):
            return [scores]
            
        return scores
