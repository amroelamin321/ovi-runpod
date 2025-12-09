FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_HOME=/models/.torch \
    HF_HOME=/models/.cache/huggingface \
    PYTHONPATH=/workspace:$PYTHONPATH

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

RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Clone Ovi repository FIRST
RUN git clone https://github.com/character-ai/Ovi.git /workspace/ovi

# Copy requirements.txt and install ALL dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Verify critical packages are installed
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')"
RUN python -c "import omegaconf; print(f'OmegaConf: {omegaconf.__version__}')"
RUN python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
RUN python -c "import runpod; print(f'RunPod: {runpod.__version__}')"

# Create directories
RUN mkdir -p /models /workspace /tmp/video-output && \
    chmod 777 /models /workspace /tmp/video-output

# Copy handler and config
COPY handler.py /workspace/handler.py

CMD ["python", "-u", "/workspace/handler.py"]
