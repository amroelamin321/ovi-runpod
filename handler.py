import runpod
import os
import sys
import torch
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
import tempfile
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import cloudinary
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
except Exception as e:
    CLOUDINARY_ENABLED = False
    logger.warning(f"⚠ Cloudinary not available: {e}")


class OviVideoGenerator:
    def __init__(self):
        self.model_initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("=" * 60)
        logger.info("INITIALIZING OVI VIDEO GENERATOR")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            capability = torch.cuda.get_device_capability(0)
            logger.info(f"CUDA Capability: sm_{capability[0]}{capability[1]}")
        
        try:
            self._initialize_model()
            self.model_initialized = True
            logger.info("✓ Ovi model initialized successfully")
        except Exception as e:
            logger.error(f"✗ Model initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.model_initialized = False

    def _initialize_model(self):
        """Initialize Ovi model with pre-downloaded models"""
        try:
            # Add Ovi repo to Python path
            ovi_path = '/workspace/ovi'
            if ovi_path not in sys.path:
                sys.path.insert(0, ovi_path)
            
            logger.info(f"Ovi path: {ovi_path}")
            
            # Check if Ovi repo exists
            if not os.path.exists(ovi_path):
                raise RuntimeError(f"Ovi repository not found at {ovi_path}")
            
            ovi_contents = os.listdir(ovi_path)
            logger.info(f"Ovi directory contains {len(ovi_contents)} items")
            
            # Use pre-downloaded models from build time
            self.model_paths = {
                'ovi': '/models/ovi',
                'flux': '/models/flux',
                't5': '/models/t5'
            }
            
            # Verify models exist
            models_ready = True
            for name, path in self.model_paths.items():
                if os.path.exists(path):
                    size_mb = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file()) / (1024 * 1024)
                    logger.info(f"✓ {name} models found at {path} ({size_mb:.1f} MB)")
                else:
                    logger.warning(f"⚠ {name} models not found at {path}")
                    models_ready = False
            
            # Check for inference script
            self.inference_script = os.path.join(ovi_path, 'inference.py')
            
            if os.path.exists(self.inference_script):
                logger.info("✓ Found inference.py")
            else:
                logger.warning("⚠ inference.py not found")
                self.inference_script = None
            
            if not models_ready:
                logger.warning("⚠ Some models missing - will attempt to download at runtime")
            
            logger.info("✓ Model initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def generate_video(
        self,
        prompt: str,
        duration: int = 10,
        image_url: Optional[str] = None,
        seed: Optional[int] = None,
        num_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate video using Ovi
        
        Args:
            prompt: Text description of the video
            duration: Video length in seconds (5 or 10)
            image_url: Optional image URL for image-to-video mode
            seed: Optional random seed for reproducibility
            num_steps: Optional number of diffusion steps
        
        Returns:
            Dict with status, video_path, mode, duration, and prompt
        """
        
        if not self.model_initialized:
            raise RuntimeError("Model not initialized. Check worker startup logs.")
        
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        
        if duration not in [5, 10]:
            raise ValueError("Duration must be 5 or 10 seconds")
        
        try:
            mode = 'i2v' if image_url else 't2v'
            logger.info(f"Generating {duration}s video ({mode}): {prompt[:80]}...")
            
            # Create output directory
            os.makedirs("/tmp/video-output", exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"/tmp/video-output/ovi_{mode}_{timestamp}.mp4"
            
            if self.inference_script and os.path.exists(self.inference_script):
                # Use Ovi's inference script
                import subprocess
                
                cmd = [
                    "python", self.inference_script,
                    "--prompt", prompt,
                    "--output", output_file,
                    "--duration", str(duration)
                ]
                
                # Add model path if available
                if self.model_paths.get('ovi'):
                    cmd.extend(["--model_path", self.model_paths['ovi']])
                
                # Add seed if provided
                if seed is not None:
                    cmd.extend(["--seed", str(seed)])
                
                # Add num_steps if provided
                if num_steps is not None:
                    cmd.extend(["--num_steps", str(num_steps)])
                
                # Handle image-to-video mode
                if image_url:
                    image_path = self._download_image(image_url)
                    cmd.extend(["--image", image_path])
                    logger.info(f"Using image: {image_path}")
                
                logger.info(f"Running: {' '.join(cmd)}")
                
                # Run inference with timeout
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutes max
                )
                
                if result.returncode != 0:
                    logger.error(f"Stderr: {result.stderr}")
                    raise RuntimeError(f"Generation failed: {result.stderr}")
                
                logger.info(f"Stdout: {result.stdout}")
                
                # Clean up downloaded image
                if image_url and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except:
                        pass
                
            else:
                raise RuntimeError("inference.py not found. Cannot generate video.")
            
            # Verify output file was created
            if not os.path.exists(output_file):
                raise RuntimeError(f"Video generation failed - output file not created")
            
            file_size = os.path.getsize(output_file)
            logger.info(f"✓ Video generated: {output_file} ({file_size/1024/1024:.2f}MB)")
            
            return {
                'status': 'success',
                'video_path': output_file,
                'mode': mode,
                'duration': duration,
                'prompt': prompt,
                'file_size_mb': round(file_size/1024/1024, 2)
            }
            
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _download_image(self, image_url: str) -> str:
        """Download image from URL for image-to-video mode"""
        try:
            import urllib.request
            from PIL import Image
            
            logger.info(f"Downloading image from: {image_url}")
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                urllib.request.urlretrieve(image_url, tmp.name)
                
                # Verify it's a valid image
                img = Image.open(tmp.name)
                img.verify()
                
                logger.info(f"✓ Image downloaded: {tmp.name} ({img.size[0]}x{img.size[1]})")
                return tmp.name
                
        except Exception as e:
            raise ValueError(f"Failed to download image: {str(e)}")

    def upload_to_cloudinary(
        self,
        video_path: str,
        prompt: str,
        duration: int,
        mode: str
    ) -> Dict[str, Any]:
        """Upload video to Cloudinary"""
        
        if not CLOUDINARY_ENABLED:
            # Return local file path if Cloudinary not available
            logger.warning("Cloudinary not configured, returning local path")
            return {
                'status': 'success',
                'url': f'file://{video_path}',
                'public_id': os.path.basename(video_path),
                'video_id': os.path.basename(video_path)
            }
        
        try:
            logger.info(f"Uploading to Cloudinary: {video_path}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            public_id = f"ovi-{mode}-{duration}s-{timestamp}"
            
            # Upload with progress
            response = cloudinary.uploader.upload(
                video_path,
                resource_type='video',
                public_id=public_id,
                folder='ovi-videos',
                tags=['ovi_generated', f'ovi_{mode}', f'{duration}s'],
                timeout=300,
                chunk_size=6000000  # 6MB chunks
            )
            
            logger.info(f"✓ Upload successful: {response['secure_url']}")
            
            return {
                'status': 'success',
                'url': response['secure_url'],
                'public_id': response['public_id'],
                'video_id': response['public_id'],
                'cloudinary_response': {
                    'size': response.get('bytes'),
                    'duration': response.get('duration'),
                    'format': response.get('format'),
                    'width': response.get('width'),
                    'height': response.get('height')
                }
            }
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            raise

    def cleanup(self, video_path: str):
        """Clean up temporary video file"""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"✓ Cleaned up: {video_path}")
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")


# Initialize generator
logger.info("=" * 60)
logger.info("STARTING OVI VIDEO GENERATOR")
logger.info("=" * 60)

try:
    generator = OviVideoGenerator()
    logger.info("✓ Generator initialized successfully")
except Exception as e:
    logger.error(f"✗ Failed to initialize generator: {str(e)}")
    logger.error(traceback.format_exc())
    generator = None


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function
    
    Expected input:
    {
        "input": {
            "prompt": "A cat playing piano",
            "duration": 10,  # optional, default 10 (5 or 10)
            "image_url": "https://...",  # optional, for i2v mode
            "seed": 42  # optional
        }
    }
    
    Returns:
    {
        "status": "success",
        "video_url": "https://cloudinary.com/...",
        "video_id": "ovi-t2v-10s-20241209_130000",
        "mode": "t2v",
        "duration": 10,
        "generation_time_seconds": 45.2,
        "prompt": "A cat playing piano"
    }
    """
    job_input = job.get('input', {})
    start_time = datetime.now()
    
    try:
        # Check if generator is initialized
        if generator is None or not generator.model_initialized:
            return {
                'status': 'error',
                'error_type': 'initialization_error',
                'message': 'Ovi model failed to initialize. Check worker logs for details.'
            }
        
        # Extract parameters
        prompt = job_input.get('prompt', '').strip()
        duration = job_input.get('duration', 10)
        image_url = job_input.get('image_url')
        seed = job_input.get('seed')
        
        # Validate prompt
        if not prompt:
            return {
                'status': 'error',
                'error_type': 'validation_error',
                'message': 'Prompt is required and cannot be empty'
            }
        
        mode = 'i2v' if image_url else 't2v'
        logger.info("=" * 60)
        logger.info(f"PROCESSING REQUEST: {mode.upper()} - {duration}s")
        logger.info(f"Prompt: {prompt}")
        if image_url:
            logger.info(f"Image: {image_url}")
        logger.info("=" * 60)
        
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
        
        # Cleanup local file
        generator.cleanup(gen_result['video_path'])
        
        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info(f"✓ SUCCESS - Video ready: {upload_result['url']}")
        logger.info(f"Generation time: {generation_time:.1f}s")
        logger.info("=" * 60)
        
        return {
            'status': 'success',
            'video_url': upload_result['url'],
            'video_id': upload_result['video_id'],
            'mode': gen_result['mode'],
            'duration': duration,
            'generation_time_seconds': round(generation_time, 2),
            'prompt': prompt,
            'file_size_mb': gen_result.get('file_size_mb')
        }
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        
        return {
            'status': 'error',
            'error_type': 'generation_error',
            'message': f"Generation failed: {str(e)}"
        }


# Start the serverless worker
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("STARTING OVI RUNPOD SERVERLESS WORKER")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info("=" * 60)
    
    runpod.serverless.start({'handler': handler})
