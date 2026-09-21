"""
main.py - FastAPI Backend Server
Pituitary Tumor Detection + Grad-CAM API

Endpoints:
  GET  /         -> Welcome message
  GET  /health   -> Model load status
  POST /predict  -> Upload MRI image -> Get prediction + Grad-CAM
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add backend directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
import predictor


# -----------------------------------------------------------
# Lifespan: Load model once at server startup
# -----------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model when server starts, cleanup when server stops."""
    print("=" * 60)
    print("  Pituitary Tumor Detection API - Starting...")
    print("=" * 60)
    try:
        predictor.load_model()
        print("  Server is READY to accept requests!")
        print("=" * 60)
    except Exception as e:
        print(f"  FATAL: Could not load model: {e}")
        print("=" * 60)
    yield
    print("Server shutting down...")


# -----------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------
app = FastAPI(
    title="Pituitary Tumor Detection API",
    description="AI-powered MRI analysis with Grad-CAM explainability",
    version="1.0.0",
    lifespan=lifespan
)

# -----------------------------------------------------------
# CORS Middleware (Allow React frontend to connect)
# -----------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server and any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------

@app.get("/")
async def root():
    """Welcome endpoint."""
    return {
        "message": "Pituitary Tumor Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Upload MRI image for tumor detection + Grad-CAM",
            "GET /health": "Check model and server status"
        }
    }


@app.get("/health")
async def health_check():
    """Check if the model is loaded and server is healthy."""
    model_loaded = predictor._model is not None
    gradcam_ready = predictor._gradcam_model is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "gradcam_ready": gradcam_ready,
        "model_path": str(predictor.MODEL_PATH)
    }


@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    """
    Upload an MRI image and receive:
      - prediction: "Pituitary Tumor" or "No Tumor"
      - confidence: float (0.0 to 1.0)
      - confidence_percent: string e.g. "94.20%"
      - gradcam_image: base64-encoded PNG (3-panel visualization)
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/bmp", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: JPEG, PNG, BMP, WEBP"
        )

    # Read file bytes
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {str(e)}")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Run prediction + Grad-CAM
    try:
        result = predictor.predict(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return JSONResponse(content=result)


# -----------------------------------------------------------
# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# -----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
