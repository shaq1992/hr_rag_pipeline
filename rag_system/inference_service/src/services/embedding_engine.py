import logging
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingEngine, cls).__new__(cls)
        return cls._instance

    def get_model(self):
        if self._model is None:
            logger.info("Loading BGE-M3 model into memory (CPU)...")
            # use_fp16=False is mandatory for stable CPU execution
            self._model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
            logger.info("BGE-M3 loaded.")
        return self._model

    def embed(self, text: str) -> tuple[list[float], list[int], list[float]]:
        model = self.get_model()
        
        output = model.encode(
            [text], 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False
        )

        dense_vec = output['dense_vecs'][0].tolist()
        lexical_weights = output['lexical_weights'][0]

        # Qdrant schema requirements
        sparse_indices = [int(k) for k in lexical_weights.keys()]
        sparse_values = [float(v) for v in lexical_weights.values()]

        return dense_vec, sparse_indices, sparse_values
