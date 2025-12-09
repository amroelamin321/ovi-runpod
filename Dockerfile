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

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Clone Ovi repository
RUN git clone https://github.com/character-ai/Ovi.git /workspace/ovi

# Copy requirements.txt
COPY requirements.txt /workspace/requirements.txt

# Install all dependencies from requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Verify ALL critical imports work
RUN python -c "import torch; print(f'✓ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && \
    python -c "from diffusers import FluxPipeline; print('✓ FluxPipeline available')" && \
    python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')" && \
    python -c "import omegaconf; print('✓ OmegaConf')" && \
    python -c "import pandas; print('✓ Pandas')" && \
    python -c "import pydub; print('✓ Pydub')" && \
    python -c "from moviepy.editor import ImageSequenceClip; print('✓ Moviepy')" && \
    python -c "import librosa; print('✓ Librosa')" && \
    python -c "import einops; print('✓ Einops')" && \
    python -c "import timm; print('✓ Timm')" && \
    echo "✓✓✓ ALL DEPENDENCIES VERIFIED ✓✓✓"

# Copy handler
COPY handler.py /workspace/handler.py

# Create necessary directories
RUN mkdir -p /models /tmp/video-output

# Set environment variables
ENV PYTHONPATH="/workspace/ovi:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

# Run handler
CMD ["python", "-u", "handler.py"]
