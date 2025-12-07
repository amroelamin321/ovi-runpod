FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone Ovi repository
RUN git clone https://github.com/character-ai/Ovi.git /ovi

WORKDIR /ovi

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch 2.5.1 (2.6.0 not available for CUDA 12.1)
RUN pip install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Ovi dependencies from their requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Flash Attention (optional, skip on failure)
RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || echo "Flash Attention skipped - will use regular attention"

# Install RunPod + Cloudinary
RUN pip install --no-cache-dir runpod cloudinary requests pillow python-dotenv

# ⭐⭐⭐ CRITICAL: Download Ovi model weights (40GB+) ⭐⭐⭐
RUN python3 download_weights.py --output-dir /root/.cache/ovi_models

# Create output directories
RUN mkdir -p /tmp/ovi_output

# Copy handler files
COPY handler.py /handler.py
COPY utils.py /utils.py

# Environment
ENV PYTHONUNBUFFERED=1

# Start RunPod handler
CMD ["python3", "-u", "/handler.py"]
