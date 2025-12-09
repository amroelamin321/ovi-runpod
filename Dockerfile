FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libsndfile1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Clone Ovi
RUN git clone https://github.com/character-ai/Ovi.git /workspace/ovi

# Install base packages
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Install moviepy separately (avoids conflicts)
RUN pip install --no-cache-dir moviepy==1.0.3 || \
    pip install --no-cache-dir moviepy || \
    echo "moviepy will be handled at runtime"

# Verify core packages
RUN python -c "import torch; print(f'torch: {torch.__version__}')"
RUN python -c "import diffusers; print(f'diffusers: {diffusers.__version__}')"
RUN python -c "import transformers; print(f'transformers: {transformers.__version__}')"

# Create models directory
RUN mkdir -p /models /tmp/video-output

# Download models (lightweight script)
RUN python -c "\
from huggingface_hub import snapshot_download; \
import os; \
print('Downloading models...'); \
try: \
    snapshot_download('character-ai/Ovi', local_dir='/models/ovi', cache_dir='/models/.cache', resume_download=True); \
    print('Models downloaded'); \
except Exception as e: \
    print(f'Warning: {e}'); \
"

# Copy handler
COPY handler.py /workspace/handler.py

# Environment
ENV PYTHONPATH="/workspace/ovi:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/.cache
ENV TRANSFORMERS_CACHE=/models/.cache

RUN echo "Container ready"

CMD ["python", "-u", "handler.py"]
