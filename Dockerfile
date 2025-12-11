FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev build-essential git \
    wget curl ca-certificates libsndfile1 ffmpeg libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Clone Ovi code (no models)
RUN git clone https://github.com/character-ai/Ovi.git /workspace/ovi

# Install dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Install FlashAttention - EXACT wheel for Python 3.10, torch 2.6.0, CUDA 12.1
# This specific version (2.7.4.post1 with cxx11abiFALSE) is confirmed compatible with torch 2.6.0
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Verify imports
RUN python -c "import torch; print('PyTorch:', torch.__version__)"
RUN python -c "from diffusers import FluxPipeline; print('Diffusers OK')"
RUN python -c "import flash_attn; print('FlashAttention:', flash_attn.__version__)"

# Copy handler
COPY handler.py /workspace/handler.py

RUN mkdir -p /tmp/video-output

ENV PYTHONPATH="/workspace/ovi:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
