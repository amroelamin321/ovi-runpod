FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

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
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Clone Ovi
RUN git clone https://github.com/character-ai/Ovi.git /workspace/ovi

# Copy and install requirements
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Verify EVERYTHING works
RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✓ PyTorch {torch.__version__} with CUDA')"
RUN python -c "from diffusers import FluxPipeline; print('✓ FluxPipeline available')"
RUN python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
RUN python -c "from moviepy.editor import ImageSequenceClip, AudioFileClip; print('✓ Moviepy complete')"
RUN python -c "import pandas, pydub, librosa, omegaconf, einops, timm; print('✓ All core libraries')"
RUN python -c "import sys; sys.path.insert(0, '/workspace/ovi'); from ovi.utils.io_utils import save_video; print('✓ Ovi imports work')"

# Copy handler
COPY handler.py /workspace/handler.py

# Create directories
RUN mkdir -p /models /tmp/video-output

# Environment
ENV PYTHONPATH="/workspace/ovi:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Health check
RUN python -c "print('🚀 Container ready for Ovi video generation')"

CMD ["python", "-u", "handler.py"]
