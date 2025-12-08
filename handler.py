import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTHONUNBUFFERED'] = '1'

import sys
import json
import uuid
import traceback
from pathlib import Path

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

# Create output directory
os.makedirs('/tmp/ovi_output', exist_ok=True)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)
print("[INIT] ✅ Cloudinary configured")

# Check GPU
if not torch.cuda.is_available():
    print("[ERROR] No GPU available!")
    sys.exit(1)

device = 0
torch.cuda.set_device(device)
gpu_name = torch.cuda.get_device_name(device)
gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
print(f"[INIT] GPU: {gpu_name} ({gpu_memory:.1f}GB)")

# Determine model path (Network Volume or local cache)
if os.path.ismount('/mnt/models') and os.path.exists('/mnt/models/ovi_models'):
    model_path = '/mnt/models/ovi_models'
    print("[INIT] Using Network Volume at /mnt/models/ovi_models")
else:
    model_path = '/root/.cache/ovi_models'
    print("[INIT] Using local cache at /root/.cache/ovi_models")

# Create model directory if needed
Path(model_path).mkdir(parents=True, exist_ok=True)

# Check if models exist
weights_exist = (
    os.path.exists(f'{model_path}/Ovi') and
    os.path.exists(f'{model_path}/Wan2.2-TI2V-5B') and
    os.path.exists(f'{model_path}/MMAudio')
)

if not weights_exist:
    print("[INIT] ⏳ Model weights not found. Downloading (~30GB, may take 10-15 minutes)...")
    from huggingface_hub import snapshot_download
    try:
        for model_id, local_dir in [
            ('chetwinlow1/Ovi', f'{model_path}/Ovi'),
            ('Wan-AI/Wan2.2-TI2V-5B', f'{model_path}/Wan2.2-TI2V-5B'),
            ('hkchengrex/MMAudio', f'{model_path}/MMAudio'),
        ]:
            print(f"[INIT] Downloading {model_id}...")
            snapshot_download(model_id, local_dir=local_dir, local_dir_use_symlinks=False)
            print(f"[INIT] ✅ Downloaded {model_id}")
        print("[INIT] ✅ All models downloaded")
    except Exception as e:
        print(f"[ERROR] Download failed: {str(e)}")
        sys.exit(1)
else:
    print("[INIT] ✅ Model weights cached")

# Load OVI config - EXACTLY as per official code
print("[INIT] Loading OVI config...")
config = OmegaConf.create({
    'ckpt_dir': model_path,
    'output_dir': '/tmp/ovi_output',
    'cpu_offload': True,  # CRITICAL for 24GB VRAM
    'mode': 't2v'         # Text-to-video mode
})

# Initialize OVI engine - EXACTLY as per official code
print("[INIT] Loading OVI Fusion Engine (this may take 2-3 minutes)...")
try:
    ovi_engine = OviFusionEngine(
        config=config,
        device=device,
        target_dtype=torch.bfloat16
    )
    print("[INIT] ✅ OVI Engine loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load OVI engine: {str(e)}")
    print(traceback.format_exc())
    sys.exit(1)

print("[INIT] ✅ System ready for inference!")

# ==================== HANDLER ====================

def handler(job):
    """RunPod serverless handler for OVI 1.1"""
    job_id = job.get('id', 'unknown')
    print(f"\n[JOB {job_id}] Starting...")
    
    try:
        job_input = job['input']
        text_prompt = job_input.get('prompt', '')
        seed = job_input.get('seed', 100)
        num_steps = job_input.get('num_steps', 50)
        image_url = job_input.get('image_url', None)
        
        if not text_prompt:
            return {"status": "error", "message": "Prompt is required"}
        
        print(f"[JOB {job_id}] Prompt: {text_prompt}")
        print(f"[JOB {job_id}] Steps: {num_steps}, Seed: {seed}")
        
        try:
            # EXACT signature from official OviFusionEngine.generate()
            generated_video, generated_audio, generated_image = ovi_engine.generate(
                text_prompt=text_prompt,
                image_path=image_url,  # None for T2V, path string for I2V
                video_frame_height_width=[960, 960],  # OFFICIAL: must be [h, w]
                seed=seed,
                solver_name='unipc',
                sample_steps=num_steps,  # OFFICIAL param name
                shift=5.0,
                video_guidance_scale=5.0,  # OFFICIAL default
                audio_guidance_scale=4.0,  # OFFICIAL default
                slg_layer=9,  # OFFICIAL default
                video_negative_prompt='',
                audio_negative_prompt=''
            )
            
            if generated_video is None:
                return {"status": "error", "message": "Generation returned None"}
            
            print(f"[JOB {job_id}] ✅ Video generated")
            
        except Exception as e:
            print(f"[JOB {job_id}] ❌ Generation failed: {str(e)}")
            print(traceback.format_exc())
            return {"status": "error", "message": f"Generation: {str(e)}"}
        
        # Save video
        try:
            output_path = f'/tmp/ovi_output/output_{uuid.uuid4()}.mp4'
            print(f"[JOB {job_id}] Saving video...")
            save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
            print(f"[JOB {job_id}] ✅ Video saved")
            
        except Exception as e:
            print(f"[JOB {job_id}] ❌ Save failed: {str(e)}")
            print(traceback.format_exc())
            return {"status": "error", "message": f"Save: {str(e)}"}
        
        # Upload to Cloudinary
        try:
            print(f"[JOB {job_id}] Uploading to Cloudinary...")
            upload_result = cloudinary.uploader.upload(
                output_path,
                resource_type="video",
                public_id=f"ovi_{uuid.uuid4()}",
                timeout=600
            )
            
            print(f"[JOB {job_id}] ✅ Upload successful")
            
            try:
                os.remove(output_path)
            except:
                pass
            
            return {
                "status": "success",
                "video_url": upload_result.get('secure_url'),
                "public_id": upload_result.get('public_id'),
                "duration": upload_result.get('duration')
            }
            
        except Exception as e:
            print(f"[JOB {job_id}] ⚠️ Cloudinary failed: {str(e)}")
            return {
                "status": "success_local",
                "message": "Generated but Cloudinary upload failed",
                "error": str(e),
                "local_path": output_path
            }
        
    except Exception as e:
        print(f"[JOB {job_id}] ❌ Handler exception: {str(e)}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

print("[INIT] Starting RunPod serverless handler...")
runpod.serverless.start({"handler": handler})
