#!/usr/bin/env python3
import os
import sys
import json
import uuid
import traceback
import time
from typing import Dict, Any
import requests
from io import BytesIO

import torch
import runpod
from PIL import Image
import cloudinary
import cloudinary.uploader
from omegaconf import OmegaConf

sys.path.insert(0, '/ovi')

from ovi.ovi_fusion_engine import OviFusionEngine
from ovi.utils.io_utils import save_video

print("[INIT] Starting Ovi 1.1 initialization...")

os.makedirs('/tmp/ovi_output', exist_ok=True)

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)
print("[INIT] Cloudinary configured")

if not torch.cuda.is_available():
    print("[ERROR] No GPU available!")
    sys.exit(1)

device = 0
torch.cuda.set_device(device)
print(f"[INIT] GPU: {torch.cuda.get_device_name(device)}")

# Auto-download weights on first load
model_path = '/root/.cache/ovi_models'
if not os.path.exists(f'{model_path}/checkpoints'):
    print(f"[INIT] Model weights not found. Downloading (~30GB, may take 5-10 minutes)...")
    os.system(f"cd /ovi && python3 download_weights.py --output-dir {model_path}")
    print("[INIT] ✅ Model weights downloaded!")

print("[INIT] Loading Ovi config...")
config = OmegaConf.create({
    'ckpt_dir': model_path,
    'output_dir': '/tmp/ovi_output',
    'video_frame_height_width': [720, 720],
    'num_steps': 50,
    'solver_name': 'unipc',
    'shift': 5.0,
    'video_guidance_scale': 4.0,
    'audio_guidance_scale': 3.0,
    'slg_layer': 11,
    'video_negative_prompt': '',
    'audio_negative_prompt': '',
    'cpu_offload': True,   # ⭐⭐⭐ ENABLE CPU OFFLOAD FOR 32GB GPU ⭐⭐⭐
    'fp8': False,
    'seed': 100
})

# ⭐⭐⭐ SET PYTORCH MEMORY OPTIMIZATION ⭐⭐⭐
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("[INIT] Loading OVI Fusion Engine...")
try:
    ovi_engine = OviFusionEngine(
        config=config,
        device=device,
        target_dtype=torch.bfloat16
    )
    print("[INIT] ✅ OVI Fusion Engine loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load Ovi engine: {str(e)}")
    print(traceback.format_exc())
    sys.exit(1)

print("[INIT] ✅ Ready to process requests")

# ... REST OF HANDLER CODE (download_image_from_url, generate_video_with_ovi, etc.) ...
