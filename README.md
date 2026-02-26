# 🏛️ Enterprise HR Policy RAG Pipeline

An enterprise-grade, asynchronous Retrieval-Augmented Generation (RAG) system built to parse, index, and query Human Resources policies for the Government of Prince Edward Island. 

This pipeline strictly enforces anti-hallucination safeguards, utilizes hybrid vector search, and delivers responses with deterministic citations via Server-Sent Events (SSE).



## 🏗️ Architecture Overview

This project is built using a decoupled, service-oriented architecture containerized with Docker.

1. **Ingestion Pipeline (`/ingestion`)**
   * **Parser:** Uses the Unstructured Serverless API (vision-capable `HI_RES` strategy) for document-structure-aware chunking (preserving tables, lists, and section headers).
   * **Embeddings:** Natively generates both dense semantic vectors (1024d) and sparse lexical vectors locally using `BAAI/bge-m3`.
   * **Storage:** Stores payload and vectors in **Qdrant**, utilizing hardware-accelerated payload filtering.

2. **Inference Microservice (`/rag_system/inference_service`)**
   * A dedicated FastAPI worker that handles heavy PyTorch tensor operations.
   * Isolates the `bge-m3` embedding model and the `bge-reranker-base` cross-encoder to prevent event-loop blocking on the main gateway.

3. **API Gateway (`/rag_system/api_gateway`)**
   * An entirely I/O-bound asynchronous traffic director.
   * **Query Routing:** Uses Gemini 2.5 Flash for zero-shot classification to reject out-of-scope queries instantly.
   * **Retrieval:** Executes **Reciprocal Rank Fusion (RRF)** via Qdrant to merge sparse and dense search results.
   * **Generation:** Streams the final response to the client with dynamically appended, programmatic citations.

4. **Evaluation Engine (`/eval_system`)**
   * An automated LLM-as-a-judge framework.
   * Parses the JSONL telemetry logs from the API gateway to deterministically calculate **Recall@K** and **Mean Reciprocal Rank (MRR)**.
   * Prompts Gemini 2.5 Flash to evaluate the generated answers for **Correctness**, **Citation Accuracy**, and **Grounding (Anti-Hallucination)**.

## 🚀 Key Features
* **Time-to-First-Token (TTFT) < 3 seconds** via SSE streaming.
* **Strict Chunk Boundaries:** Context is passed via metadata (`section_header`, `source_document`), ensuring policy rules are never severed from their governing titles.
* **100% Citation Accuracy:** LLMs are restricted from generating citations; the backend appends verified document metadata directly to the output.

## 📄 Setup & Installation
Please see [SETUP.md](SETUP.md) for step-by-step instructions on cloning the repository, injecting API keys, and spinning up the Docker environment.
