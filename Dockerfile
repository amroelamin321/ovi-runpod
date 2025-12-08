FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_HOME=/models/.torch \
    HF_HOME=/models/.cache/huggingface \
    PYTHONPATH=/workspace:$PYTHONPATH

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

# Install PyTorch 2.1.0 (compatible with RTX 6000 Ada)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# Install core ML packages
RUN pip install --no-cache-dir \
    transformers==4.36.0 \
    diffusers==0.25.0 \
    accelerate==0.25.0 \
    safetensors==0.4.1 \
    huggingface-hub==0.20.1 \
    einops==0.7.0 \
    timm==0.9.12

# Install Cloudinary and RunPod
RUN pip install --no-cache-dir cloudinary==1.40.0 runpod==1.6.2

# Install image/audio processing
RUN pip install --no-cache-dir \
    Pillow==10.1.0 \
    opencv-python-headless==4.8.1.78 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    audioread==3.0.1 \
    numpy==1.24.3 \
    scipy==1.11.4 \
    pyyaml==6.0.1

# Clone Ovi repository
RUN git clone https://github.com/character-ai/Ovi.git /workspace/ovi

# Download models (commented out for now - will check if models exist at runtime)
RUN mkdir -p /models && python -c "\
import huggingface_hub; \
print('Downloading Ovi models...'); \
try: \
    huggingface_hub.snapshot_download('character-ai/Ovi-1.1', \
        cache_dir='/models/.cache', \
        local_dir='/models/ovi-1.1', \
        allow_patterns=['*.safetensors', '*.json', '*.yaml'], \
        resume_download=True); \
    print('✓ Downloaded'); \
except Exception as e: \
    print(f'Model download will happen at runtime: {e}')"

COPY handler.py /workspace/handler.py
COPY config/ /workspace/config/

RUN mkdir -p /models /workspace /tmp/video-output && \
    chmod 777 /models /workspace /tmp/video-output

CMD ["python", "-u", "/workspace/handler.py"]
