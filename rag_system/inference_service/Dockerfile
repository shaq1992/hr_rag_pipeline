FROM python:3.11-slim

WORKDIR /app

# System dependencies for compiling some python packages if needed
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# --- THE CACHE-FRIENDLY HOTFIX ---
# Injects the missing import into the library without busting the pip install cache
RUN sed -i '1s/^/from typing import Optional\n/' /usr/local/lib/python3.11/site-packages/FlagEmbedding/BGE_M3/trainer.py
# ---------------------------------

COPY src/ /app/src/

ENV PYTHONPATH=/app/src

# Pre-download weights (optional but recommended for stability)
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
