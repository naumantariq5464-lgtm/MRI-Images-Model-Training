"""
main.py - FastAPI Backend Server
Brain Tumor MRI Scanner API (Pituitary + Glioma + Grad-CAM)

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
# Lifespan: Load models once at server startup
# -----------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models when server starts, cleanup when server stops."""
    print("=" * 60)
    print("  Brain Tumor MRI Scanner API - Starting...")
    print("=" * 60)
    try:
        predictor.load_model()
        print("  Server is READY to accept requests!")
        print("=" * 60)
    except Exception as e:
        print(f"  FATAL: Could not load models: {e}")
        print("=" * 60)
    yield
    print("Server shutting down...")


# -----------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------
app = FastAPI(
    title="Brain Tumor MRI Scanner API",
    description="AI-powered MRI analysis with unified Pituitary & Glioma detection + Grad-CAM explainability",
    version="2.0.0",
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
        "message": "Brain Tumor MRI Scanner API",
        "version": "2.0.0",
        "supported_tumors": ["Pituitary Tumor", "Glioma Tumor"],
        "endpoints": {
            "POST /predict": "Upload MRI image for unified tumor detection + Grad-CAM",
            "GET /health": "Check model and server status"
        }
    }


@app.get("/health")
async def health_check():
    """Check if models are loaded and server is healthy."""
    pituitary_loaded = predictor._pituitary_model is not None
    glioma_loaded = predictor._glioma_model is not None
    any_loaded = pituitary_loaded or glioma_loaded
    return {
        "status": "healthy" if any_loaded else "no_models_loaded",
        "pituitary_model_loaded": pituitary_loaded,
        "glioma_model_loaded": glioma_loaded,
        "pituitary_gradcam_ready": predictor._pituitary_gradcam is not None,
        "glioma_gradcam_ready": predictor._glioma_gradcam is not None,
    }


@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    """
    Upload an MRI image and receive:
      - prediction: "Pituitary Tumor", "Glioma Tumor", or "No Tumor"
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

    # Run unified prediction + Grad-CAM
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
# Run with: python backend/main.py
# -----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
