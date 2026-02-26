FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Force the CPU-only version of PyTorch to avoid a 2.5GB CUDA download
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt
