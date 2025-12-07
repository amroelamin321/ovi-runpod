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

# ⭐ SET MEMORY CONFIG BEFORE IMPORTING OVI
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
if os.path.ismount('/mnt/models') and os.path.exists('/mnt/models/ovi_models'):
    model_path = '/mnt/models/ovi_models'
    print("[INIT] Using Network Volume at /mnt/models/ovi_models ✅")
else:
    model_path = '/root/.cache/ovi_models'
    print("[INIT] Network Volume not attached, using local cache ⚠️")

# STEP 1: Check if models exist
weights_exist = (
    os.path.exists(f'{model_path}/Ovi') and
    os.path.exists(f'{model_path}/Wan2.2-TI2V-5B') and
    os.path.exists(f'{model_path}/MMAudio') and
    os.path.exists(f'{model_path}/checkpoints')
)

if not weights_exist:
    print(f"[INIT] Models not found at {model_path}")
    print(f"[INIT] Downloading (~30GB, takes 10-15 minutes on first run)...")
    
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

# STEP 2: Verify all models exist
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

# STEP 3: Load OVI config
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
    'cpu_offload': True,
    'fp8': False,
    'seed': 100
})

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

# ==================== HANDLER FUNCTIONS ====================

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
        print(f"[GENERATE] Seed: {seed}")
        
        # Clear GPU cache before generation
        torch.cuda.empty_cache()
        
        # Call the actual Ovi generation
        # If image provided, use image-to-video mode
        if image is not None:
            print(f"[GENERATE] Using image-to-video mode")
            output = ovi_engine.generate(
                prompt=prompt,
                image=image,
                num_steps=num_steps,
                seed=seed
            )
        else:
            print(f"[GENERATE] Using text-to-video mode")
            output = ovi_engine.generate(
                prompt=prompt,
                num_steps=num_steps,
                seed=seed
            )
        
        # Save video to output path
        output_path = '/tmp/ovi_output/output.mp4'
        save_video(output, output_path)
        
        print(f"[GENERATE] ✅ Video generated: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Generation failed: {str(e)}")
        print(traceback.format_exc())
        raise

def handler(job):
    """RunPod Serverless handler"""
    try:
        print(f"[JOB] Processing job: {job['id']}")
        
        job_input = job['input']
        prompt = job_input.get('prompt')
        image_url = job_input.get('image_url')
        num_steps = job_input.get('num_steps', 50)
        seed = job_input.get('seed', 100)
        
        if not prompt:
            return {"status": "error", "message": "Prompt required"}
        
        print(f"[JOB] Prompt: {prompt}")
        print(f"[JOB] Image URL: {image_url}")
        print(f"[JOB] Steps: {num_steps}, Seed: {seed}")
        
        # Download image if provided
        image = None
        if image_url:
            print(f"[JOB] Downloading image...")
            image = download_image_from_url(image_url)
            print(f"[JOB] Image downloaded: {image.size}")
        
        # Generate video
        print(f"[JOB] Starting video generation...")
        output_path = generate_video_with_ovi(prompt, image, num_steps, seed)
        
        # Upload to Cloudinary
        print(f"[JOB] Uploading to Cloudinary...")
        upload_result = cloudinary.uploader.upload(
            output_path,
            resource_type="video",
            public_id=f"ovi_{uuid.uuid4()}"
        )
        
        print(f"[JOB] ✅ Job completed successfully")
        return {
            "status": "success",
            "video_url": upload_result['secure_url'],
            "duration": upload_result.get('duration', 'N/A'),
            "public_id": upload_result.get('public_id')
        }
        
    except Exception as e:
        print(f"[ERROR] Handler exception: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

# Start the serverless handler
runpod.serverless.start({"handler": handler})
