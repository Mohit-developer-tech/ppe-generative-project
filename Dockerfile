FROM python:3.11-slim

# Install system libraries needed by OpenCV and PyTorch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies (CPU version of PyTorch for fast build and lightweight image)
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained models
COPY common/ common/
COPY dashboard/ dashboard/
COPY artifacts/ artifacts/
COPY results/ results/
COPY data_cache/cvae_exemplars.pt data_cache/cvae_exemplars.pt
COPY run_server.py .

# Hugging Face Spaces expects the container to listen on port 7860
ENV PORT=7860
EXPOSE 7860

# Run FastAPI app via Uvicorn on 0.0.0.0:7860
CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "7860"]
