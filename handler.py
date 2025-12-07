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

# Add Ovi to Python path
sys.path.insert(0, '/ovi')

from ovi.ovi_fusion_engine import OviFusionEngine
from ovi.utils.io_utils import save_video

print("[INIT] Starting Ovi 1.1 initialization...")

# Create output directory
os.makedirs('/tmp/ovi_output', exist_ok=True)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)
print("[INIT] Cloudinary configured")

# Check GPU
if not torch.cuda.is_available():
    print("[ERROR] No GPU available!")
    sys.exit(1)

device = 0
torch.cuda.set_device(device)
print(f"[INIT] GPU: {torch.cuda.get_device_name(device)}")

# ⭐⭐⭐ CORRECTED CONFIG - Points to actual downloaded weights ⭐⭐⭐
print("[INIT] Loading Ovi config...")
config = OmegaConf.create({
    'ckpt_dir': '/root/.cache/ovi_models',  # ⚠️ CHANGED FROM model_path
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
    'cpu_offload': False,  # Set to True if using <32GB GPU
    'fp8': False,  # Set to True if using 24GB GPU
    'seed': 100
})

# Load Ovi model
print("[INIT] Loading OVI Fusion Engine (2-3 minutes, downloading weights if needed)...")
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

def download_image_from_url(image_url: str) -> str:
    """Download image from URL and save to temp file"""
    print(f"[DOWNLOAD] Fetching image: {image_url}")
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        temp_image_path = f"/tmp/ovi_output/input_{uuid.uuid4()}.png"
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image.save(temp_image_path)
        
        print(f"[DOWNLOAD] ✅ Image saved: {temp_image_path} ({image.size})")
        return temp_image_path
    except Exception as e:
        raise ValueError(f"Failed to download image from {image_url}: {str(e)}")

def generate_video_with_ovi(
    mode: str,
    prompt: str,
    image_path: str = None,
    seed: int = 100,
    num_steps: int = 50,
    video_guidance: float = 4.0,
    audio_guidance: float = 3.0
) -> str:
    """Generate video using Ovi model"""
    
    output_path = f"/tmp/ovi_output/video_{uuid.uuid4()}.mp4"
    
    print(f"[GENERATE] Running Ovi {mode.upper()} inference...")
    print(f"[GENERATE] Prompt: {prompt}")
    print(f"[GENERATE] Steps: {num_steps}, Seed: {seed}")
    if image_path:
        print(f"[GENERATE] Input image: {image_path}")
    
    try:
        # Generate with Ovi
        generated_video, generated_audio, generated_image = ovi_engine.generate(
            text_prompt=prompt,
            image_path=image_path,
            video_frame_height_width=[720, 720],
            seed=seed,
            solver_name='unipc',
            sample_steps=num_steps,
            shift=5.0,
            video_guidance_scale=video_guidance,
            audio_guidance_scale=audio_guidance,
            slg_layer=11,
            video_negative_prompt='',
            audio_negative_prompt=''
        )
        
        print(f"[GENERATE] ✅ Inference complete, saving video...")
        
        # Save video with audio
        save_video(
            output_path,
            generated_video,
            generated_audio,
            fps=24,
            sample_rate=16000
        )
        
        file_size = os.path.getsize(output_path)
        print(f"[GENERATE] ✅ Video saved: {output_path} ({file_size} bytes)")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Generation failed: {str(e)}")
        print(traceback.format_exc())
        raise

def upload_to_cloudinary(file_path: str) -> Dict[str, Any]:
    """Upload video to Cloudinary"""
    print(f"[UPLOAD] Uploading: {file_path} ({os.path.getsize(file_path)} bytes)")
    
    try:
        result = cloudinary.uploader.upload(
            file_path,
            resource_type='video',
            public_id=f'ovi_videos/{uuid.uuid4()}',
            quality='auto:best',
            fetch_format='auto',
            video_codec='h264',
            audio_codec='aac'
        )
        
        video_url = result.get('secure_url')
        print(f"[UPLOAD] ✅ Success: {video_url}")
        
        return {
            'url': video_url,
            'public_id': result.get('public_id'),
            'bytes': result.get('bytes'),
            'duration': result.get('duration')
        }
    except Exception as e:
        print(f"[ERROR] Upload failed: {str(e)}")
        raise

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler"""
    start_time = time.time()
    output_path = None
    temp_image_path = None
    
    try:
        # Extract input
        input_data = job.get('input', {})
        mode = input_data.get('mode', 't2v').lower()
        prompt = input_data.get('prompt', '').strip()
        image_url = input_data.get('image_url', '').strip()
        seed = input_data.get('seed', 100)
        num_steps = input_data.get('num_steps', 50)
        video_guidance = input_data.get('video_guidance_scale', 4.0)
        audio_guidance = input_data.get('audio_guidance_scale', 3.0)
        
        print(f"[JOB] ═══════════════════════════════════")
        print(f"[JOB] Mode: {mode}")
        print(f"[JOB] Prompt: {prompt[:100]}...")
        if image_url:
            print(f"[JOB] Image URL: {image_url}")
        print(f"[JOB] ═══════════════════════════════════")
        
        # Validate mode
        if mode not in ['t2v', 'i2v']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 't2v' or 'i2v'")
        
        # Validate prompt
        if not prompt or len(prompt) < 3:
            raise ValueError("prompt must be at least 3 characters")
        
        # Handle i2v mode
        if mode == 'i2v':
            if not image_url:
                raise ValueError("i2v mode requires 'image_url' parameter")
            temp_image_path = download_image_from_url(image_url)
        
        # Generate video
        output_path = generate_video_with_ovi(
            mode=mode,
            prompt=prompt,
            image_path=temp_image_path,
            seed=seed,
            num_steps=num_steps,
            video_guidance=video_guidance,
            audio_guidance=audio_guidance
        )
        
        # Upload to Cloudinary
        upload_result = upload_to_cloudinary(output_path)
        
        # Clean up
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        generation_time = time.time() - start_time
        
        print(f"[SUCCESS] ✅ Total time: {round(generation_time, 2)}s")
        
        return {
            'status': 'success',
            'mode': mode,
            'video_url': upload_result['url'],
            'cloudinary_public_id': upload_result['public_id'],
            'generation_time_seconds': round(generation_time, 2),
            'model_version': 'ovi-1.1',
            'prompt_used': prompt,
            'video_duration_seconds': upload_result.get('duration'),
            'file_size_bytes': upload_result.get('bytes'),
            'seed': seed,
            'num_steps': num_steps
        }
    
    except Exception as e:
        print(f"[ERROR] ❌ {str(e)}")
        print(traceback.format_exc())
        
        # Clean up on error
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except:
                pass
        
        generation_time = time.time() - start_time
        
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'generation_time_seconds': round(generation_time, 2)
        }

# Start RunPod serverless worker
if __name__ == '__main__':
    print("[RUNPOD] Starting serverless worker...")
    runpod.serverless.start({'handler': handler})
