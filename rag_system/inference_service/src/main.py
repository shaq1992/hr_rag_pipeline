from fastapi import FastAPI
import logging
from api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="RAG Inference Microservice")

app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "inference_engine"}
