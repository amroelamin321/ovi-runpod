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

# Set python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Clone Ovi repository
RUN git clone https://github.com/character-ai/Ovi.git /workspace/ovi

# Copy and install requirements
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Create models directory
RUN mkdir -p /models

# Create model download script
RUN echo 'from huggingface_hub import snapshot_download\n\
import os\n\
\n\
print("📥 Downloading Ovi models...")\n\
\n\
try:\n\
    snapshot_download(\n\
        repo_id="character-ai/Ovi",\n\
        local_dir="/models/ovi",\n\
        cache_dir="/models/.cache",\n\
        resume_download=True,\n\
        ignore_patterns=["*.md", "*.txt", ".gitattributes"]\n\
    )\n\
    print("✓ Ovi model downloaded")\n\
except Exception as e:\n\
    print(f"⚠ Ovi model download warning: {e}")\n\
\n\
try:\n\
    snapshot_download(\n\
        repo_id="black-forest-labs/FLUX.1-dev",\n\
        local_dir="/models/flux",\n\
        cache_dir="/models/.cache",\n\
        resume_download=True,\n\
        allow_patterns=["*.safetensors", "*.json", "*.txt"],\n\
        ignore_patterns=["*.md", ".gitattributes"]\n\
    )\n\
    print("✓ FLUX model downloaded")\n\
except Exception as e:\n\
    print(f"⚠ FLUX download warning: {e}")\n\
\n\
try:\n\
    snapshot_download(\n\
        repo_id="google/t5-v1_1-xxl",\n\
        local_dir="/models/t5",\n\
        cache_dir="/models/.cache",\n\
        resume_download=True,\n\
        allow_patterns=["*.safetensors", "*.json"],\n\
        ignore_patterns=["*.md", ".gitattributes"]\n\
    )\n\
    print("✓ T5 encoder downloaded")\n\
except Exception as e:\n\
    print(f"⚠ T5 download warning: {e}")\n\
\n\
print("✅ Model downloads complete")\n' > /tmp/download_models.py

# Run model download
RUN python /tmp/download_models.py

# Verify model files exist
RUN ls -lh /models/ && echo "Models directory contents listed"

# Verify all dependencies work
RUN python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA Available: {torch.cuda.is_available()}')"
RUN python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
RUN python -c "import diffusers; from diffusers import FluxPipeline; print(f'✓ Diffusers: {diffusers.__version__} with FluxPipeline')"
RUN python -c "from moviepy.editor import ImageSequenceClip; print('✓ Moviepy complete')"
RUN python -c "import pandas, pydub, librosa, omegaconf; print('✓ All core libraries OK')"

# Test Ovi imports
RUN python -c "import sys; sys.path.insert(0, '/workspace/ovi'); from ovi.utils.io_utils import save_video; print('✓ Ovi imports successful')"

# Copy handler
COPY handler.py /workspace/handler.py

# Create output directory
RUN mkdir -p /tmp/video-output

# Set environment variables
ENV PYTHONPATH="/workspace/ovi:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/.cache
ENV TRANSFORMERS_CACHE=/models/.cache
ENV HF_DATASETS_CACHE=/models/.cache

# Final ready message
RUN echo "🚀 OVI CONTAINER BUILD COMPLETE - READY FOR VIDEO GENERATION 🚀"

CMD ["python", "-u", "handler.py"]
