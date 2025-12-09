FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
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

# Set python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Clone Ovi repository
RUN git clone https://github.com/character-ai/Ovi.git /workspace/ovi

# Copy and install requirements
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# ============================================
# PRE-DOWNLOAD ALL MODELS AT BUILD TIME
# ============================================

# Create models directory
RUN mkdir -p /models

# Download Ovi models from Hugging Face
RUN python -c "
from huggingface_hub import snapshot_download
import os

print('📥 Downloading Ovi models...')

# Download main Ovi model
try:
    snapshot_download(
        repo_id='character-ai/Ovi',
        local_dir='/models/ovi',
        cache_dir='/models/.cache',
        resume_download=True,
        ignore_patterns=['*.md', '*.txt', '.gitattributes']
    )
    print('✓ Ovi model downloaded')
except Exception as e:
    print(f'⚠ Ovi model download warning: {e}')

# Download FLUX model (used by Ovi)
try:
    snapshot_download(
        repo_id='black-forest-labs/FLUX.1-dev',
        local_dir='/models/flux',
        cache_dir='/models/.cache',
        resume_download=True,
        allow_patterns=['*.safetensors', '*.json', '*.txt'],
        ignore_patterns=['*.md', '.gitattributes']
    )
    print('✓ FLUX model downloaded')
except Exception as e:
    print(f'⚠ FLUX download warning: {e}')

# Download T5 text encoder (used by Ovi)
try:
    snapshot_download(
        repo_id='google/t5-v1_1-xxl',
        local_dir='/models/t5',
        cache_dir='/models/.cache',
        resume_download=True,
        allow_patterns=['*.safetensors', '*.json'],
        ignore_patterns=['*.md', '.gitattributes']
    )
    print('✓ T5 encoder downloaded')
except Exception as e:
    print(f'⚠ T5 download warning: {e}')

print('✅ Model downloads complete')
"

# Verify model files exist
RUN ls -lh /models/ovi || echo "Ovi models location check"
RUN ls -lh /models/flux || echo "FLUX models location check"
RUN ls -lh /models/t5 || echo "T5 models location check"

# ============================================
# VERIFY ALL DEPENDENCIES WORK
# ============================================

RUN python -c "
import torch
import transformers
import diffusers
from moviepy.editor import ImageSequenceClip
import pandas, pydub, librosa, omegaconf

print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA Available: {torch.cuda.is_available()}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ Diffusers: {diffusers.__version__}')

from diffusers import FluxPipeline
print('✓ FluxPipeline available')

# Test Ovi imports
import sys
sys.path.insert(0, '/workspace/ovi')
from ovi.utils.io_utils import save_video
print('✓ Ovi imports successful')

print('🚀 ALL SYSTEMS READY')
"

# Copy handler
COPY handler.py /workspace/handler.py

# Create necessary directories
RUN mkdir -p /tmp/video-output

# Set environment variables
ENV PYTHONPATH="/workspace/ovi:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/.cache
ENV TRANSFORMERS_CACHE=/models/.cache
ENV HF_DATASETS_CACHE=/models/.cache

# Run handler
CMD ["python", "-u", "handler.py"]
