from fastapi import APIRouter
from dto.request import EmbedRequest, RerankRequest
from dto.response import EmbedResponse, RerankResponse
from services.embedding_engine import EmbeddingEngine
from services.reranking_engine import RerankingEngine

router = APIRouter()
embed_engine = EmbeddingEngine()
rerank_engine = RerankingEngine()

@router.post("/embed", response_model=EmbedResponse)
def generate_embedding(request: EmbedRequest):
    dense, s_idx, s_val = embed_engine.embed(request.text)
    return EmbedResponse(
        dense_vector=dense,
        sparse_indices=s_idx,
        sparse_values=s_val
    )

@router.post("/rerank", response_model=RerankResponse)
def generate_rerank_scores(request: RerankRequest):
    scores = rerank_engine.rerank(request.query, request.documents)
    return RerankResponse(scores=scores)
