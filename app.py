from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import requests

app = FastAPI(title="Ovi 1.1 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENDPOINT_ID = os.environ.get('RUNPOD_ENDPOINT_ID')
API_KEY = os.environ.get('RUNPOD_API_KEY')

class VideoRequest(BaseModel):
    prompt: str
    num_steps: int = 50
    seed: int = -1

class ImageVideoRequest(BaseModel):
    prompt: str
    image_url: str
    num_steps: int = 50
    seed: int = -1

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ovi-1.1"}

@app.post("/api/v1/generate/t2v")
async def generate_t2v(request: VideoRequest):
    try:
        response = requests.post(
            f"https://api.runpod.io/v2/{ENDPOINT_ID}/run",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "input": {
                    "mode": "t2v",
                    "prompt": request.prompt,
                    "num_steps": request.num_steps,
                    "seed": request.seed
                }
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate/i2v")
async def generate_i2v(request: ImageVideoRequest):
    try:
        response = requests.post(
            f"https://api.runpod.io/v2/{ENDPOINT_ID}/run",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "input": {
                    "mode": "i2v",
                    "prompt": request.prompt,
                    "image_url": request.image_url,
                    "num_steps": request.num_steps,
                    "seed": request.seed
                }
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status/{task_id}")
async def check_status(task_id: str):
    try:
        response = requests.get(
            f"https://api.runpod.io/v2/{ENDPOINT_ID}/status/{task_id}",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
