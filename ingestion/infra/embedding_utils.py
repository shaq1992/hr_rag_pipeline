from FlagEmbedding import BGEM3FlagModel
import logging

logger = logging.getLogger(__name__)

# Global variable to hold the model singleton
_model = None

def get_model():
    """Lazy loads the model only when first requested."""
    global _model
    if _model is None:
        logger.info("Downloading/Loading BGE-M3 model weights... (This may take a few minutes on the first run)")
        # use_fp16=False ensures compatibility for CPU execution
        _model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
        logger.info("BGE-M3 loaded successfully.")
    return _model

def get_bge_m3_embeddings(text: str):
    """Generates Dense (1024d) and Sparse (Lexical) embeddings in a single pass."""
    model = get_model()
    
    output = model.encode([text], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    
    dense_vec = output['dense_vecs'][0].tolist()
    lexical_weights = output['lexical_weights'][0]
    
    # Qdrant requires sparse indices to be integers and values to be floats
    sparse_indices = [int(k) for k in lexical_weights.keys()]
    sparse_values = [float(v) for v in lexical_weights.values()]
    
    return dense_vec, sparse_indices, sparse_values
