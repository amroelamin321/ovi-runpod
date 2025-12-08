# Multi-stage Dockerfile for Ovi 1.1 + RunPod Serverless

# Base image with CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_HOME=/models/.torch \
    HF_HOME=/models/.cache/huggingface \
    PYTHONPATH=/ovi:$PYTHONPATH

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

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /

# Clone Ovi repository
RUN git clone https://github.com/character-ai/Ovi.git /ovi

# Download Ovi 1.1 model weights (~28GB unquantized)
# This layer caches the model after first build
RUN mkdir -p /models/ovi-1-1 && \
    cd /ovi && \
    python -c "\
import torch; \
import huggingface_hub; \
print('Downloading Ovi 1.1 model...'); \
huggingface_hub.snapshot_download( \
    'character-ai/Ovi-1.1', \
    cache_dir='/models/.cache', \
    local_dir='/models/ovi-1-1', \
    allow_patterns=['*.safetensors', '*.json', '*.yaml'], \
    resume_download=True, \
    max_workers=4 \
); \
print('✓ Model download complete')"

# Download VAE models (WAN 2.2, MMAudio)
RUN python -c "\
import huggingface_hub; \
print('Downloading VAE models...'); \
huggingface_hub.snapshot_download( \
    'Wan-AI/Wan2.2-TI2V-5B', \
    cache_dir='/models/.cache', \
    allow_patterns=['*.safetensors', 'vae/*'], \
    resume_download=True \
); \
print('✓ VAE models cached')"

# Download T5 encoder
RUN python -c "\
import transformers; \
print('Downloading T5 encoder...'); \
transformers.AutoTokenizer.from_pretrained('google-t5/t5-base', cache_dir='/models/.cache'); \
transformers.AutoModel.from_pretrained('google-t5/t5-base', cache_dir='/models/.cache'); \
print('✓ T5 encoder cached')"

# Install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy handler code
COPY handler.py /handler.py
COPY config/ /config/

# Verify installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
RUN python -c "import runpod; print(f'RunPod SDK version: {runpod.__version__}')"
RUN python -c "import cloudinary; print('✓ Cloudinary SDK loaded')"

# Create model directory structure
RUN mkdir -p /models /workspace /tmp/video-output && \
    chmod 777 /models /workspace /tmp/video-output

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import torch; print('CUDA OK' if torch.cuda.is_available() else 'CPU')" || exit 1

# Set entrypoint
CMD ["python", "-u", "/handler.py"]
