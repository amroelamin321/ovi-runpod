FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

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

# Install deps
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Set HF token if provided (optional but recommended)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Download ALL models - THIS HAPPENS IN BUILD
RUN cd /workspace/ovi && \
    python download_weights.py --output-dir /workspace/ckpts && \
    ls -lh /workspace/ckpts/ && \
    du -sh /workspace/ckpts/ && \
    echo "✓ Models downloaded and verified"

# Verify everything works
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
RUN python -c "from diffusers import FluxPipeline; print('FluxPipeline OK')"
RUN python -c "from moviepy.editor import ImageSequenceClip; print('Moviepy OK')"
RUN python -c "import sys; sys.path.insert(0, '/workspace/ovi'); from ovi.utils.io_utils import save_video; print('Ovi imports OK')"

# Verify checkpoints exist
RUN test -d /workspace/ckpts || (echo "ERROR: Checkpoints not found!" && exit 1)
RUN find /workspace/ckpts -name "*.safetensors" | head -3 || (echo "WARNING: No safetensors found")

COPY handler.py /workspace/handler.py

RUN mkdir -p /tmp/video-output

ENV PYTHONPATH="/workspace/ovi:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

RUN echo "🚀 Build complete - Models ready at /workspace/ckpts 🚀"

CMD ["python", "-u", "handler.py"]
