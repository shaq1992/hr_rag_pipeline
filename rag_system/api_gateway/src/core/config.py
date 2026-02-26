import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME = "HR RAG API Gateway"
    
    # Internal Docker Network Hostnames
    INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://inference_api:8001")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    
    # External APIs
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Pipeline Parameters
    COLLECTION_NAME = "hr_policies"
    RETRIEVAL_K = 10  # Number of documents to fetch before re-ranking
    RERANK_THRESHOLD = -3.0  # Minimum score to consider a chunk relevant

settings = Settings()
