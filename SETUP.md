⚙️ Setup & Execution Guide

This guide details how to spin up the Enterprise HR RAG Pipeline locally using Docker. The architecture relies on nested microservices communicating over a shared Docker bridge network.

📋 Prerequisites

Docker & Docker Compose installed and running.

Google Gemini API Key (Using gemini-2.5-flash).

Unstructured API Key & URL (For document parsing).

Step 1: Clone the Repository

Clone the repository to your local machine and navigate into the project root.

git clone [https://github.com/shaq1992/hr_rag_pipeline.git](https://github.com/shaq1992/hr_rag_pipeline.git)
cd hr_rag_pipeline


Step 2: Configure Environment Variables

You need to create two .env files to securely inject your API keys into the containers.

1. Ingestion Environment (ingestion/.env)
Create a file at ingestion/.env and add your keys:

UNSTRUCTURED_API_KEY=your_unstructured_api_key_here
UNSTRUCTURED_API_URL=your_unstructured_api_url_here
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_HOST=qdrant


2. RAG API Gateway Environment (rag_system/api_gateway/.env)
Create a file at rag_system/api_gateway/.env and add:

GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_HOST=ingestion-qdrant-1


Step 3: Start the Core Infrastructure

First, we establish the shared network, the Vector Database (Qdrant), and the PyTorch Inference engine.

# 1. Start Qdrant
cd ingestion
docker-compose up -d

# 2. Create the shared docker network and attach Qdrant
docker network create rag_net
docker network connect rag_net ingestion-qdrant-1

# 3. Start the Inference Service 
# NOTE: The first boot will take several minutes as it downloads ~3GB of 
# HuggingFace model weights (BGE-M3 and BGE-Reranker-Base) to your local cache volume.
cd ../rag_system/inference_service
docker-compose up --build -d


Step 4: Ingest the HR Policies

Place all your PDF policies into the ingestion/sources/ directory (sample documents are already provided). Then, execute the ingestion script to parse, chunk, embed, and store the documents in Qdrant.

cd ../../ingestion

# Ensure the script is executable
chmod +x ingest_batch.sh

# Run the ingestion pipeline
./ingest_batch.sh


Wait for the script to finish processing all documents before proceeding to Step 5.

Step 5: Start the RAG API Gateway

Once data is successfully indexed, boot up the main orchestration gateway.

cd ../rag_system/api_gateway
docker-compose up --build -d


The gateway is now actively listening for requests on http://localhost:8080.

Step 6: Query the System

You can interact with the RAG pipeline using curl.
Tip: Use the -N flag to disable buffering so you can watch the Server-Sent Events (SSE) stream generate in real-time.

Test a Factual HR Query:

curl -N -X POST "http://localhost:8080/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "How many days of annual leave am I entitled to?"}'


Test the Zero-Shot Router (Out of Scope Rejection):

curl -N -X POST "http://localhost:8080/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the best restaurant near the office?"}'


Step 7: Run the Evaluation Suite

The system includes an automated LLM-as-a-judge evaluation container. It fires 15 predefined test cases at the API Gateway, parses the internal telemetry logs, and computes system metrics (Recall@K, Mean Reciprocal Rank, Correctness, and Hallucination).

cd ../../eval_system

# Build and run the evaluation container
docker-compose up --build -d

# Follow the logs to watch the LLM judge evaluate the pipeline in real-time!
docker logs eval_system-evaluator-1 -f
