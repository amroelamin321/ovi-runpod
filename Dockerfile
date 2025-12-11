FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev build-essential git \
    wget curl ca-certificates libsndfile1 ffmpeg libsm6 libxext6 libxrender-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Clone Ovi code (no models)
RUN git clone https://github.com/character-ai/Ovi.git /workspace/ovi

# Install dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Install FlashAttention - try pip first (has pre-built wheels)
RUN pip install flash-attn --no-build-isolation || \
    (echo "Pre-built wheel failed, trying source build..." && \
     git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
     cd /tmp/flash-attention && \
     MAX_JOBS=4 pip install . --no-build-isolation && \
     cd /workspace && \
     rm -rf /tmp/flash-attention)

# Verify imports
RUN python -c "import torch; print('PyTorch OK')"
RUN python -c "from diffusers import FluxPipeline; print('Diffusers OK')"
RUN python -c "import flash_attn; print('FlashAttention OK')"

# Copy handler
COPY handler.py /workspace/handler.py

RUN mkdir -p /tmp/video-output

ENV PYTHONPATH="/workspace/ovi:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
