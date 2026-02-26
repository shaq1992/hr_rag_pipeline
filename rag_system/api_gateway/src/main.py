import logging
from fastapi import FastAPI
from api.routes import router

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(title="HR RAG API Gateway")

app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "api_gateway"}
