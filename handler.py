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

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import cloudinary, but don't crash if it fails
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
            # DON'T raise - allow worker to start even if model init fails
            # This way we can debug via API calls
            self.model_initialized = False

    def _download_models_if_needed(self):
        """Download Ovi models if they don't exist"""
        model_path = '/models/ovi-1-1'
        
        if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
            logger.info(f"✓ Models already exist at {model_path}")
            return model_path
        
        logger.info("Models not found, will use models from Ovi repo")
        return None

    def _initialize_model(self):
        """Initialize Ovi model"""
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
            logger.info(f"Ovi directory exists with {len(ovi_contents)} items")
            logger.info(f"Ovi contents: {', '.join(ovi_contents[:10])}")
            
            # Download models if needed
            self.model_path = self._download_models_if_needed()
            
            # Check for inference script
            self.inference_script = os.path.join(ovi_path, 'inference.py')
            
            if os.path.exists(self.inference_script):
                logger.info("✓ Found inference.py")
            else:
                logger.warning("⚠ inference.py not found")
                self.inference_script = None
            
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
        
        if not self.model_initialized:
            raise RuntimeError("Model not initialized. Check worker startup logs.")
        
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        
        if duration not in [5, 10]:
            raise ValueError("Duration must be 5 or 10 seconds")
        
        try:
            logger.info(f"Generating {duration}s video: {prompt[:50]}...")
            
            # Create output directory
            os.makedirs("/tmp/video-output", exist_ok=True)
            output_file = f"/tmp/video-output/ovi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            if self.inference_script and os.path.exists(self.inference_script):
                # Use Ovi's inference script
                import subprocess
                
                cmd = [
                    "python", self.inference_script,
                    "--prompt", prompt,
                    "--output", output_file,
                ]
                
                if seed is not None:
                    cmd.extend(["--seed", str(seed)])
                
                if image_url:
                    image_path = self._download_image(image_url)
                    cmd.extend(["--image", image_path])
                
                logger.info(f"Running: {' '.join(cmd)}")
                
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
                
            else:
                raise RuntimeError("inference.py not found. Cannot generate video.")
            
            if not os.path.exists(output_file):
                raise RuntimeError(f"Video generation failed - output file not created")
            
            file_size = os.path.getsize(output_file)
            logger.info(f"✓ Video generated: {output_file} ({file_size/1024/1024:.2f}MB)")
            
            return {
                'status': 'success',
                'video_path': output_file,
                'mode': 'i2v' if image_url else 't2v',
                'duration': duration,
                'prompt': prompt
            }
            
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _download_image(self, image_url: str) -> str:
        try:
            import urllib.request
            from PIL import Image
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                urllib.request.urlretrieve(image_url, tmp.name)
                Image.open(tmp.name).verify()
                logger.info(f"✓ Image downloaded: {tmp.name}")
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
            
            response = cloudinary.uploader.upload(
                video_path,
                resource_type='video',
                public_id=public_id,
                folder='ovi-videos',
                tags=['ovi_generated', f'ovi_{mode}', f'{duration}s'],
                timeout=300
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
                    'format': response.get('format')
                }
            }
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            raise

    def cleanup(self, video_path: str):
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"✓ Cleaned up: {video_path}")
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")


# Initialize generator - catch errors but don't crash
logger.info("Initializing Ovi generator...")
try:
    generator = OviVideoGenerator()
    logger.info("✓ Generator initialized successfully")
except Exception as e:
    logger.error(f"✗ Failed to initialize generator: {str(e)}")
    logger.error(traceback.format_exc())
    generator = None


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod handler function"""
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
        
        logger.info(f"Processing: duration={duration}s, mode={'i2v' if image_url else 't2v'}")
        
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
        logger.error(f"Handler error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            'status': 'error',
            'error_type': 'generation_error',
            'message': f"Generation failed: {str(e)}"
        }


# Start the serverless worker
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Ovi RunPod serverless worker...")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info("=" * 60)
    
    runpod.serverless.start({'handler': handler})
