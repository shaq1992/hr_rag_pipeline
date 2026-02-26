from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
import os
import logging
import json

from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared

from infra.llm_utils import summarize_table
from infra.embedding_utils import get_bge_m3_embeddings
from infra.qdrant_utils import init_collection, upsert_chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="HR Knowledge Base Ingestion Pipeline")
COLLECTION_NAME = "hr_policies"

# Initialize the Unstructured Serverless Client
unstructured_client = UnstructuredClient(
    api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
    server_url=os.getenv("UNSTRUCTURED_ENDPOINT"),
)

@app.on_event("startup")
def startup_event():
    logger.info("Initializing Qdrant Collection...")
    init_collection(COLLECTION_NAME)

class IngestRequest(BaseModel):
    file_path: str

@app.post("/ingest")
def ingest_document(request: IngestRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    file_name = os.path.basename(request.file_path)
    
    def process_stream():
        logger.info(f"Starting API ingestion for {file_name}")
        yield json.dumps({"status": "init", "message": f"Starting {file_name}..."}) + "\n"
        
        llm_calls = 0
        prompt_tokens = 0
        candidate_tokens = 0
        
        def update_telemetry(usage):
            nonlocal llm_calls, prompt_tokens, candidate_tokens
            llm_calls += 1
            if usage:
                prompt_tokens += getattr(usage, 'prompt_token_count', 0)
                candidate_tokens += getattr(usage, 'candidates_token_count', 0)

        try:
            yield json.dumps({"status": "parsing", "message": "Calling Unstructured API..."}) + "\n"
            with open(request.file_path, "rb") as f:
                req = operations.PartitionRequest(
                    partition_parameters=shared.PartitionParameters(
                        files=shared.Files(content=f.read(), file_name=file_name),
                        strategy=shared.Strategy.HI_RES,
                    )
                )
                res = unstructured_client.general.partition(request=req)
                elements = res.elements
            
            # Strip structural noise
            filtered_elements = [e for e in elements if e.get("type") not in ["Header", "Footer"]]
            current_section_header = "General Policy"
            chunks_upserted = 0
            
            yield json.dumps({"status": "start_chunks", "total": len(filtered_elements)}) + "\n"
            
            for idx, element in enumerate(filtered_elements):
                chunk_text = str(element.get("text", "")).strip()
                if not chunk_text:
                    continue
                    
                content_type = "text"
                raw_content = chunk_text
                embed_text = chunk_text
                
                if element.get("type") == "Title":
                    current_section_header = chunk_text
                    yield json.dumps({"status": "chunk_progress", "current": idx + 1}) + "\n"
                    continue
                    
                elif element.get("type") == "Table":
                    content_type = "table"
                    metadata = element.get("metadata", {})
                    raw_content = metadata.get("text_as_html", chunk_text)
                    
                    # Only call Gemini for table elements
                    embed_text, usage = summarize_table(raw_content)
                    update_telemetry(usage)
                
                # Extract page number dynamically
                page_number = element.get("metadata", {}).get("page_number", "Unknown")
                
                # Generate vectors
                dense, s_idx, s_val = get_bge_m3_embeddings(embed_text)
                
                # Leaner Payload Construction
                payload = {
                    "source_document": file_name,
                    "page_number": page_number,
                    "section_header": current_section_header,
                    "content_type": content_type,
                    "raw_content": raw_content,
                    "chunk_index": idx
                }
                
                point_id = str(uuid.uuid4())
                upsert_chunk(COLLECTION_NAME, point_id, dense, s_idx, s_val, payload)
                chunks_upserted += 1
                
                yield json.dumps({"status": "chunk_progress", "current": idx + 1}) + "\n"
                
            logger.info(f"Successfully processed {file_name}.")
            yield json.dumps({
                "status": "success", 
                "file": file_name, 
                "chunks_upserted": chunks_upserted,
                "llm_calls": llm_calls
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}")
            yield json.dumps({"status": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(process_stream(), media_type="application/x-ndjson")
