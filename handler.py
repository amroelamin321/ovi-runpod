import runpod
import os
import torch
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
import cloudinary
import cloudinary.uploader
import cloudinary.api
import subprocess
import tempfile
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)

class OviVideoGenerator:
    def __init__(self):
        """Initialize Ovi model and dependencies"""
        self.model_initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        try:
            self._initialize_model()
            self.model_initialized = True
            logger.info("✓ Ovi model initialized successfully")
        except Exception as e:
            logger.error(f"✗ Model initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _initialize_model(self):
        """Load Ovi 1.1 model from local storage"""
        try:
            # Import Ovi dependencies
            import sys
            sys.path.insert(0, '/ovi')
            
            from ovi.models.diffusion import DiT
            from ovi.utils.inference import OviInference
            
            # Load model configuration
            self.config = {
                'num_steps': 30,  # Optimized for RTX 5090
                'solver_name': 'unipc',
                'shift': 5.0,
                'audio_guidance_scale': 3.0,
                'video_guidance_scale': 4.0,
                'slg_layer': 11,
                'sp_size': 1,
                'seed': 42,
                'cp_size': 1,
                'cpu_offload': False,
                'fp8': False,
            }
            
            # Initialize inference engine
            self.ovi_engine = OviInference(
                model_path='/models/ovi-1-1',
                config=self.config,
                device=self.device
            )
            
            logger.info("Ovi engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ovi: {str(e)}")
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
        Generate video using Ovi 1.1
        
        Args:
            prompt: Text description for video generation
            duration: 5 or 10 seconds (default: 10)
            image_url: Optional image URL for I2V mode
            seed: Random seed for reproducibility
            num_steps: Custom number of denoising steps
            
        Returns:
            Dict with video path and metadata
        """
        
        if not self.model_initialized:
            raise RuntimeError("Model not initialized")
        
        # Validate inputs
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        
        if duration not in [5, 10]:
            raise ValueError("Duration must be 5 or 10 seconds")
        
        if len(prompt) > 500:
            raise ValueError("Prompt exceeds 500 character limit")
        
        try:
            logger.info(f"Generating {duration}s video from prompt: {prompt[:50]}...")
            
            # Prepare inference config
            inference_config = self.config.copy()
            if seed is not None:
                inference_config['seed'] = seed
            if num_steps is not None:
                inference_config['num_steps'] = num_steps
            
            # Determine mode
            mode = "i2v" if image_url else "t2v"
            
            # Download image if I2V mode
            image_data = None
            if mode == "i2v":
                logger.info(f"I2V Mode: Downloading image from {image_url}")
                image_data = self._download_image(image_url)
            
            # Generate video
            if duration == 5:
                video_output = self.ovi_engine.generate_5s(
                    prompt=prompt,
                    image=image_data,
                    **inference_config
                )
            else:  # 10 seconds
                video_output = self.ovi_engine.generate_10s(
                    prompt=prompt,
                    image=image_data,
                    **inference_config
                )
            
            logger.info("Video generation completed")
            
            return {
                'status': 'success',
                'video_path': video_output,
                'mode': mode,
                'duration': duration,
                'prompt': prompt
            }
            
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            raise

    def _download_image(self, image_url: str) -> Optional[str]:
        """Download image from URL for I2V processing"""
        try:
            import urllib.request
            from PIL import Image
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                urllib.request.urlretrieve(image_url, tmp.name)
                # Validate image
                Image.open(tmp.name)
                logger.info(f"Image downloaded: {tmp.name}")
                return tmp.name
                
        except Exception as e:
            logger.error(f"Image download failed: {str(e)}")
            raise ValueError(f"Failed to download image: {str(e)}")

    def upload_to_cloudinary(
        self,
        video_path: str,
        prompt: str,
        duration: int,
        mode: str
    ) -> Dict[str, Any]:
        """Upload generated video to Cloudinary"""
        
        try:
            logger.info(f"Uploading video to Cloudinary: {video_path}")
            
            # Create public_id for organization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            public_id = f"ovi-{mode}-{duration}s-{timestamp}"
            
            # Upload with optimizations
            response = cloudinary.uploader.upload(
                video_path,
                resource_type='video',
                public_id=public_id,
                folder='ovi-videos',
                tags=['ovi_generated', f'ovi_{mode}', f'{duration}s'],
                overwrite=False,
                timeout=300,  # 5 minute timeout for upload
                # Optimization options
                eager=[
                    {'quality': 'auto', 'fetch_format': 'auto'}
                ],
                # Metadata
                context={
                    'prompt': prompt[:100],
                    'mode': mode,
                    'duration': str(duration)
                }
            )
            
            logger.info(f"✓ Upload successful: {response['secure_url']}")
            
            return {
                'status': 'success',
                'url': response['secure_url'],
                'public_id': response['public_id'],
                'secure_url': response['secure_url'],
                'video_id': response['public_id'],
                'cloudinary_response': {
                    'size': response.get('bytes'),
                    'duration': response.get('duration'),
                    'format': response.get('format')
                }
            }
            
        except cloudinary.exceptions.Error as e:
            logger.error(f"Cloudinary upload failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            raise

    def cleanup(self, video_path: str):
        """Clean up temporary files"""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Cleaned up: {video_path}")
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")


# Initialize generator once
try:
    generator = OviVideoGenerator()
except Exception as e:
    logger.error(f"Failed to initialize generator: {str(e)}")
    generator = None


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function
    
    Input format:
    {
        "input": {
            "prompt": "A cat dancing",
            "duration": 10,  # 5 or 10 seconds (default: 10)
            "image_url": "https://...",  # Optional for I2V
            "seed": 42,  # Optional
            "num_steps": 30  # Optional
        }
    }
    
    Output format:
    {
        "status": "success",
        "video_url": "https://cloudinary.com/...",
        "video_id": "ovi-t2v-10s-...",
        "mode": "t2v" or "i2v",
        "duration": 10,
        "generation_time_seconds": 45.2,
        "metadata": {...}
    }
    """
    
    job_input = job.get('input', {})
    start_time = datetime.now()
    
    try:
        # Validate generator
        if generator is None:
            return {
                'status': 'error',
                'error': 'Model initialization failed',
                'message': 'Ovi model failed to initialize. Check worker logs.'
            }
        
        # Extract and validate inputs
        prompt = job_input.get('prompt', '').strip()
        duration = job_input.get('duration', 10)
        image_url = job_input.get('image_url')
        seed = job_input.get('seed')
        num_steps = job_input.get('num_steps')
        
        if not prompt:
            return {'status': 'error', 'error': 'Prompt is required'}
        
        logger.info(f"Processing job: prompt='{prompt[:50]}...', duration={duration}s, mode={'i2v' if image_url else 't2v'}")
        
        # Generate video
        gen_result = generator.generate_video(
            prompt=prompt,
            duration=duration,
            image_url=image_url,
            seed=seed,
            num_steps=num_steps
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
        
        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Return success response
        return {
            'status': 'success',
            'video_url': upload_result['url'],
            'video_id': upload_result['video_id'],
            'secure_url': upload_result['secure_url'],
            'mode': gen_result['mode'],
            'duration': duration,
            'generation_time_seconds': round(generation_time, 2),
            'cloudinary_metadata': upload_result['cloudinary_response'],
            'prompt': prompt
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': str(e)
        }
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'error_type': 'generation_error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }


# Start serverless worker
if __name__ == '__main__':
    logger.info("Starting Ovi RunPod serverless worker...")
    runpod.serverless.start({
        'handler': handler,
        'return_aggregate_stream': False,  # Streaming disabled for binary video
    })
