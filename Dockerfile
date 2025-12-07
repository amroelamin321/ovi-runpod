FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone Ovi repository
RUN git clone https://github.com/character-ai/Ovi.git /ovi

WORKDIR /ovi

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch 2.5.1
RUN pip install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ⭐⭐⭐ Install Ovi's dependencies from THEIR requirements.txt ⭐⭐⭐
RUN pip install --no-cache-dir -r requirements.txt

# ⭐⭐⭐ Explicitly install missing packages (fix einops issue) ⭐⭐⭐
RUN pip install --no-cache-dir \
    einops>=0.6 \
    omegaconf>=2.1 \
    safetensors>=0.3.1 \
    diffusers>=0.21 \
    transformers>=4.30 \
    accelerate>=0.20

# Install Flash Attention (optional, skip on failure)
RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || echo "Flash Attention skipped"

# Install RunPod + Cloudinary + additional dependencies
RUN pip install --no-cache-dir \
    runpod>=1.0.0 \
    cloudinary>=1.33.0 \
    requests>=2.28.0 \
    pillow>=9.0 \
    python-dotenv>=0.19.0 \
    imageio>=2.9.0 \
    imageio-ffmpeg>=0.4.5 \
    numpy>=1.21 \
    scipy>=1.7

# Create directories
RUN mkdir -p /root/.cache/ovi_models /tmp/ovi_output

# Copy handler files
COPY handler.py /handler.py
COPY utils.py /utils.py

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Start RunPod handler
CMD ["python3", "-u", "/handler.py"]
