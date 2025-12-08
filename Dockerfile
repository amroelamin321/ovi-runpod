FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

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

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

WORKDIR /

# Clone Ovi repository
RUN git clone https://github.com/character-ai/Ovi.git /ovi

# Install PyTorch first (largest dependency)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Install core ML packages
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    diffusers==0.25.0 \
    accelerate==0.25.0 \
    safetensors==0.4.1 \
    huggingface-hub==0.20.1

# Install remaining dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt || \
    (echo "Some packages failed, continuing..." && exit 0)

# Download model weights
RUN mkdir -p /models/ovi-1-1 && \
    python -c "import huggingface_hub; \
huggingface_hub.snapshot_download('character-ai/Ovi-1.1', \
cache_dir='/models/.cache', local_dir='/models/ovi-1-1', \
allow_patterns=['*.safetensors', '*.json', '*.yaml'], \
resume_download=True)" || echo "Model download will happen at runtime"

# Copy handler
COPY handler.py /handler.py
COPY config/ /config/

RUN mkdir -p /models /workspace /tmp/video-output && \
    chmod 777 /models /workspace /tmp/video-output

CMD ["python", "-u", "/handler.py"]
