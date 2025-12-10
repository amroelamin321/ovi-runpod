import runpod
import os
import sys
import torch
import logging
import traceback
from pathlib import Path
from datetime import datetime
import tempfile
from typing import Dict, Any, Optional
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import cloudinary
    import cloudinary.uploader
    cloudinary.config(
        cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
        api_key=os.environ.get('CLOUDINARY_API_KEY'),
        api_secret=os.environ.get('CLOUDINARY_API_SECRET')
    )
    CLOUDINARY_ENABLED = True
    logger.info("✓ Cloudinary configured")
except:
    CLOUDINARY_ENABLED = False
    logger.warning("⚠ Cloudinary not configured")


class OviVideoGenerator:
    def __init__(self):
        self.model_initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("="*80)
        logger.info("OVI 1.1 INITIALIZING")
        logger.info("="*80)
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"VRAM: {vram_gb:.2f} GB")
        
        try:
            self._initialize_model()
            self.model_initialized = True
            logger.info("✓ Model initialized")
        except Exception as e:
            logger.error(f"✗ Init failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.model_initialized = False

    def _initialize_model(self):
        """Initialize using network volume models"""
        ovi_path = '/workspace/ovi'
        if ovi_path not in sys.path:
            sys.path.insert(0, ovi_path)
        
        if not os.path.exists(ovi_path):
            raise RuntimeError("Ovi repo not found")
        
        logger.info(f"✓ Ovi path: {ovi_path}")
        
        # Network volume mounts at /runpod-volume in serverless
        self.ckpt_dir = '/runpod-volume/ckpts'
        
        if not os.path.exists(self.ckpt_dir):
            raise RuntimeError(f"Models not found at {self.ckpt_dir}. Check network volume mount.")
        
        # Verify models
        size_mb = sum(f.stat().st_size for f in Path(self.ckpt_dir).rglob('*') if f.is_file()) / (1024**2)
        logger.info(f"✓ Models: {self.ckpt_dir} ({size_mb/1024:.1f} GB)")
        
        # Verify inference script
        self.inference_script = os.path.join(ovi_path, 'inference.py')
        if not os.path.exists(self.inference_script):
            raise RuntimeError("inference.py not found")
        
        logger.info("✓ Ready for generation")

    def generate_video(self, prompt: str, duration: int = 10, image_url: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate video"""
        
        if not self.model_initialized:
            raise RuntimeError("Model not initialized")
        
        if not prompt:
            raise ValueError("Prompt required")
        
        if duration not in [5, 10]:
            raise ValueError("Duration must be 5 or 10")
        
        mode = 'i2v' if image_url else 't2v'
        logger.info(f"GENERATING {duration}s {mode.upper()}: {prompt[:60]}...")
        
        os.makedirs("/tmp/video-output", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"/tmp/video-output/ovi_{mode}_{duration}s_{timestamp}.mp4"
        
        cmd = [
            "python", self.inference_script,
            "--config-file", os.path.join('/workspace/ovi', 'ovi/configs/inference/inference_fusion.yaml'),
            "--ckpt-dir", self.ckpt_dir,
            "--text-prompt", prompt,
            "--output-path", output_file,
        ]
        
        if seed:
            cmd.extend(["--seed", str(seed)])
        
        if image_url:
            image_path = self._download_image(image_url)
            cmd.extend(["--image-prompt", image_path])
        
        logger.info("⏳ Generating (60-90 seconds)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd='/workspace/ovi')
        
        if result.returncode != 0:
            logger.error(f"Generation failed: {result.stderr}")
            raise RuntimeError(f"Generation failed: {result.stderr}")
        
        if not os.path.exists(output_file):
            raise RuntimeError("Output file not created")
        
        file_size = os.path.getsize(output_file)
        logger.info(f"✓ Video: {file_size/1024/1024:.2f} MB")
        
        return {
            'status': 'success',
            'video_path': output_file,
            'mode': mode,
            'duration': duration,
            'prompt': prompt,
            'file_size_mb': round(file_size/1024/1024, 2)
        }

    def _download_image(self, image_url: str) -> str:
        """Download image for i2v"""
        import urllib.request
        from PIL import Image
        
        logger.info(f"Downloading image: {image_url}")
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            urllib.request.urlretrieve(image_url, tmp.name)
            Image.open(tmp.name).verify()
            logger.info(f"✓ Image: {tmp.name}")
            return tmp.name

    def upload_to_cloudinary(self, video_path: str, prompt: str, duration: int, mode: str) -> Dict[str, Any]:
        """Upload to Cloudinary"""
        
        if not CLOUDINARY_ENABLED:
            logger.warning("Cloudinary not configured")
            return {
                'status': 'success',
                'url': f'file://{video_path}',
                'public_id': os.path.basename(video_path),
                'video_id': os.path.basename(video_path)
            }
        
        logger.info("Uploading to Cloudinary...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        response = cloudinary.uploader.upload(
            video_path,
            resource_type='video',
            public_id=f"ovi-{mode}-{duration}s-{timestamp}",
            folder='ovi-videos',
            tags=['ovi_1.1', mode, f'{duration}s'],
            timeout=600,
            chunk_size=6000000
        )
        
        logger.info(f"✓ Uploaded: {response['secure_url']}")
        
        return {
            'status': 'success',
            'url': response['secure_url'],
            'public_id': response['public_id'],
            'video_id': response['public_id']
        }

    def cleanup(self, video_path: str):
        """Clean up temp files"""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"✓ Cleaned: {video_path}")
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")


# Initialize generator
logger.info("Starting Ovi generator...")
try:
    generator = OviVideoGenerator()
    logger.info("✓ Generator ready")
except Exception as e:
    logger.error(f"✗ Generator init failed: {str(e)}")
    generator = None


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler
    
    Input:
    {
        "input": {
            "prompt": "A cat playing piano",
            "duration": 10,
            "image_url": "https://...",  // optional for i2v
            "seed": 42  // optional
        }
    }
    """
    
    job_input = job.get('input', {})
    start_time = datetime.now()
    
    try:
        if generator is None or not generator.model_initialized:
            return {
                'status': 'error',
                'message': 'Model initialization failed. Check worker logs.'
            }
        
        # Extract params
        prompt = job_input.get('prompt', '').strip()
        duration = job_input.get('duration', 10)
        image_url = job_input.get('image_url')
        seed = job_input.get('seed')
        
        if not prompt:
            return {'status': 'error', 'message': 'Prompt is required'}
        
        mode = 'i2v' if image_url else 't2v'
        logger.info(f"REQUEST: {mode.upper()} - {duration}s")
        
        # Generate video
        gen_result = generator.generate_video(
            prompt=prompt,
            duration=duration,
            image_url=image_url,
            seed=seed
        )
        
        # Upload to Cloudinary
        upload_result = generator.upload_to_cloudinary(
            video_path=gen_result['video_path'],
            prompt=prompt,
            duration=duration,
            mode=gen_result['mode']
        )
        
        # Cleanup
        generator.cleanup(gen_result['video_path'])
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✓ SUCCESS: {upload_result['url']} ({generation_time:.1f}s)")
        
        return {
            'status': 'success',
            'video_url': upload_result['url'],
            'video_id': upload_result['video_id'],
            'mode': gen_result['mode'],
            'duration': duration,
            'generation_time_seconds': round(generation_time, 2),
            'prompt': prompt
        }
        
    except Exception as e:
        logger.error(f"✗ ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'message': str(e)
        }


if __name__ == '__main__':
    logger.info("Starting RunPod serverless worker...")
    runpod.serverless.start({'handler': handler})
