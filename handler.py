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

# ⭐ SMART PATH SELECTION ⭐
# Priority 1: Network volume (persistent, production)
# Priority 2: Local cache (fallback, for development)
if os.path.ismount('/mnt/models') and os.path.exists('/mnt/models/ovi_models'):
    model_path = '/mnt/models/ovi_models'
    print("[INIT] Using Network Volume at /mnt/models/ovi_models ✅")
else:
    model_path = '/root/.cache/ovi_models'
    print("[INIT] Network Volume not attached, using local cache ⚠️")

# STEP 1: Check if models exist (avoid re-downloading every time)
weights_exist = (
    os.path.exists(f'{model_path}/Ovi') and
    os.path.exists(f'{model_path}/Wan2.2-TI2V-5B') and
    os.path.exists(f'{model_path}/MMAudio') and
    os.path.exists(f'{model_path}/checkpoints')
)

if not weights_exist:
    print(f"[INIT] Models not found at {model_path}")
    print(f"[INIT] Downloading (~30GB, takes 10-15 minutes on first run)...")
    print(f"[INIT] Models will be cached - no re-download on next job")
    
    try:
        result = os.system(f"cd /ovi && python3 download_weights.py --output-dir {model_path}")
        if result != 0:
            print(f"[ERROR] Download failed with code {result}")
            sys.exit(1)
        print("[INIT] ✅ Model weights downloaded successfully!")
    except Exception as e:
        print(f"[ERROR] Download failed: {str(e)}")
        sys.exit(1)
else:
    print("[INIT] Models already cached - using existing weights ✅")

# STEP 2: Verify all models actually exist before loading
required_models = {
    'Ovi checkpoint': f'{model_path}/Ovi',
    'Wan 2.2 VAE': f'{model_path}/Wan2.2-TI2V-5B',
    'MMAudio model': f'{model_path}/MMAudio'
}

for model_name, model_path_check in required_models.items():
    if not os.path.exists(model_path_check):
        print(f"[ERROR] Missing {model_name} at {model_path_check}")
        sys.exit(1)

print("[INIT] ✅ All model files verified")

# STEP 3: Load OVI config with MEMORY OPTIMIZATION
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
    'cpu_offload': True,  # ⭐ CRITICAL FOR 32GB GPU ⭐
    'fp8': False,
    'seed': 100
})

# STEP 4: Memory optimization for CUDA
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
    print("[ERROR] Common causes:")
    print("  1. Incomplete model downloads")
    print("  2. Corrupted model files")
    print("  3. Out of memory - try restarting endpoint")
    print(traceback.format_exc())
    sys.exit(1)

print("[INIT] ✅ Ready to process requests")

# ==================== REST OF HANDLER CODE ====================
# (Keep your existing generate_video_with_ovi, download_image_from_url, etc.)
# Just make sure they also reference the same model_path variable

def download_image_from_url(image_url: str) -> Image.Image:
    """Download image from URL"""
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"[ERROR] Failed to download image: {str(e)}")
        raise

def generate_video_with_ovi(
    prompt: str,
    image: Image.Image = None,
    num_steps: int = 50,
    seed: int = 100
) -> str:
    """Generate video using OVI"""
    try:
        print(f"[GENERATE] Prompt: {prompt}")
        print(f"[GENERATE] Steps: {num_steps}")
        
        # Call ovi_engine.generate(...) here with your logic
        # This is where video generation happens
        
        return "/tmp/ovi_output/output.mp4"
    except Exception as e:
        print(f"[ERROR] Generation failed: {str(e)}")
        raise

def handler(job):
    """RunPod handler"""
    try:
        job_input = job['input']
        prompt = job_input.get('prompt')
        image_url = job_input.get('image_url')
        
        if not prompt:
            return {"status": "error", "message": "Prompt required"}
        
        if image_url:
            image = download_image_from_url(image_url)
        else:
            image = None
        
        output_path = generate_video_with_ovi(prompt, image)
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload_video(
            output_path,
            resource_type="video",
            public_id=f"ovi_{uuid.uuid4()}"
        )
        
        return {
            "status": "success",
            "video_url": upload_result['secure_url'],
            "duration": upload_result.get('duration', 'N/A')
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
