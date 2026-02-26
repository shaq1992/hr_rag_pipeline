from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from dto.request import UserQuery
from services.query_router import QueryRouter
from services.hybrid_retriever import HybridRetriever
from services.context_evaluator import ContextEvaluator
from services.generation_engine import GenerationEngine
from utils.event_logger import EventLogger

router = APIRouter()

@router.post("/query")
async def query_endpoint(request: UserQuery):
    # 1. Zero-Shot Routing
    routing_decision = await QueryRouter.route(request)
    
    # Initialize our evaluation payload
    log_payload = {
        "query": request.query,
        "user_id": request.user_id,
        "query_type": routing_decision.query_type,
        "retrieved_chunks": [],
        "final_answer": ""
    }

    # Handle Out-of-Scope Gracefully
    if routing_decision.query_type == "out-of-scope":
        msg = "This question appears to be outside the scope of HR policies and workplace guidelines. I can only assist with HR-related inquiries."
        log_payload["final_answer"] = msg
        await EventLogger.log_event(log_payload)
        
        async def mock_stream():
            yield msg
        return StreamingResponse(mock_stream(), media_type="text/event-stream")

    # 2. Hybrid Retrieval
    chunks = await HybridRetriever.retrieve(request.query)
    
    # 3. Cross-Encoder Re-ranking & Confidence Check
    valid_chunks, is_confident = await ContextEvaluator.evaluate_and_rerank(request.query, chunks)
    
    # Populate the log with context IDs for Recall@K calculations later
    log_payload["retrieved_chunks"] = [
        {
            "id": c.id, 
            "score": c.score, 
            "source": c.source_document, 
            "section": c.section_header,
            "content": c.content  # <--- ADD THIS LINE
        } 
        for c in valid_chunks
    ]

    # Handle Low Confidence Gracefully
    if not is_confident:
        msg = "I do not have enough information in the provided policies to answer that question accurately."
        log_payload["final_answer"] = msg
        await EventLogger.log_event(log_payload)
        
        async def mock_stream():
            yield msg
        return StreamingResponse(mock_stream(), media_type="text/event-stream")

    # 4. The Streaming Interceptor
    async def response_generator():
        full_response = ""
        # Consume the generator from our Engine
        async for token in GenerationEngine.generate_response(request.query, valid_chunks):
            full_response += token
            yield token
        
        # Once the stream is finished, log the complete state
        log_payload["final_answer"] = full_response
        await EventLogger.log_event(log_payload)

    # Return the open connection to the client
    return StreamingResponse(response_generator(), media_type="text/event-stream")
