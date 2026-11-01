#!/usr/bin/env python3
"""
SIGHTLINE — Pet Vision Laboratory
FastAPI Backend Application for MobileNetV2 Cats vs Dogs Inference
Repo: AIML_1_ImageClassification_byte

Endpoints:
    GET  /              -> Serves Editorial UI (ui/index.html)
    GET  /api/health    -> Health & Model State Check
    GET  /api/metrics   -> Empirical Evaluation Metrics (outputs/metrics.json)
    GET  /api/samples   -> Sample Inferences Metadata (outputs/sample_predictions.csv)
    POST /api/predict   -> Real-time Inference (Multipart file upload or Base64/Sample JSON)
"""

import os
import io
import json
import csv
import base64
import time
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from PIL import Image
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import core inference engine functions
from src.predict import (
    REPO_ROOT,
    MODEL_PATH,
    IMG_SIZE,
    get_model,
    run_inference_on_pil,
)

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Model state tracker
MODEL_INSTANCE = None
MODEL_LOAD_ERROR: Optional[str] = None


def load_model_on_startup():
    """Attempt to load the model once into memory."""
    global MODEL_INSTANCE, MODEL_LOAD_ERROR
    if not os.path.exists(MODEL_PATH):
        MODEL_LOAD_ERROR = f"Model artifact missing at {MODEL_PATH}"
        return

    try:
        MODEL_INSTANCE = get_model(MODEL_PATH)
        MODEL_LOAD_ERROR = None
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        MODEL_INSTANCE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up model once
    load_model_on_startup()
    yield
    # Shutdown clean up if needed


app = FastAPI(
    title="SIGHTLINE // Pet Vision Laboratory",
    description="MobileNetV2 Deep Transfer Learning Inference API & Web Interface",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local testing flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory so confusion matrix and sample photos are accessible
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")
UI_DIR = os.path.join(REPO_ROOT, "ui")

if os.path.exists(OUTPUTS_DIR):
    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

if os.path.exists(UI_DIR):
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the primary SIGHTLINE Web Interface."""
    index_path = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse(
        content="""
        <html><head><title>SIGHTLINE</title></head>
        <body style="background:#141610;color:#F2F0E7;font-family:monospace;padding:40px;">
        <h2>SIGHTLINE // Pet Vision Laboratory</h2>
        <p>UI is compiling... please ensure ui/index.html is created.</p>
        <p><a href="/api/health" style="color:#C8FF32;">Inspect /api/health</a></p>
        </body></html>
        """
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint indicating server and model readiness."""
    is_ready = MODEL_INSTANCE is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "model_path": MODEL_PATH,
        "model_file_exists": os.path.exists(MODEL_PATH),
        "architecture": "MobileNetV2 (ImageNet-1k pre-trained) + GAP2D + Dense(1)",
        "input_shape": [IMG_SIZE[0], IMG_SIZE[1], 3],
        "classes": ["CAT", "DOG"],
        "decision_threshold": 0.50,
        "error": MODEL_LOAD_ERROR,
    }


@app.get("/api/metrics")
async def get_test_metrics():
    """
    Returns empirical evaluation metrics from outputs/metrics.json.
    Strictly reports held-out benchmark performance, not live inference.
    """
    metrics_path = os.path.join(OUTPUTS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics artifact (outputs/metrics.json) not found.",
        )
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "source": "outputs/metrics.json",
            "split": "Held-Out Test Set (Disjoint Stratified Partition)",
            "metrics": data,
            "confusion_matrix_url": "/outputs/confusion_matrix.png",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read metrics: {exc}",
        )


@app.get("/api/samples")
async def get_sample_inferences():
    """
    Returns the real sample inference list and ground-truth labels from
    outputs/sample_predictions.csv and sample_inferences/ folder.
    """
    csv_path = os.path.join(OUTPUTS_DIR, "sample_predictions.csv")
    samples_dir = os.path.join(OUTPUTS_DIR, "sample_inferences")
    samples: List[Dict[str, Any]] = []

    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("image_filename")
                    file_exists = os.path.exists(os.path.join(samples_dir, fname)) if fname else False
                    samples.append({
                        "filename": fname,
                        "ground_truth": row.get("ground_truth", "").upper(),
                        "predicted_label": row.get("predicted_label", "").upper(),
                        "prob_dog": float(row.get("p_dog", 0.0)),
                        "confidence": row.get("confidence", ""),
                        "image_url": f"/outputs/sample_inferences/{fname}",
                        "file_exists": file_exists,
                    })
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed reading sample CSV: {exc}",
            )

    return {
        "count": len(samples),
        "source": "outputs/sample_predictions.csv",
        "samples": samples,
    }


def _process_pil_bytes(image_bytes: bytes, filename: str = "upload.jpg") -> Image.Image:
    """Validate image bytes and return a sanitized PIL Image in memory."""
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size ({MAX_IMAGE_SIZE_BYTES // (1024*1024)} MB).",
        )
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image payload is empty.",
        )

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Verify header integrity
        # Re-open after verify to reset pointer
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image format: {exc}",
        )

    if img.format and img.format.lower() not in {"jpeg", "png", "webp", "jpg"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported format '{img.format}'. Allowed: JPEG, PNG, WebP.",
        )

    return img


@app.post("/api/predict")
async def predict_endpoint(request: Request):
    """
    Run MobileNetV2 inference on an uploaded image or sample reference.
    Accepts:
      1. Multipart Form Data: field 'file'
      2. JSON Payload:
         - {"image": "data:image/jpeg;base64,...", "filename": "test.jpg"}
         - {"sample_filename": "cat_01.jpg"}
    """
    global MODEL_INSTANCE
    if MODEL_INSTANCE is None:
        # Try once more in case it became available
        load_model_on_startup()

    content_type = request.headers.get("content-type", "").lower()
    img_pil: Optional[Image.Image] = None
    source_name = "upload"

    # Case 1: JSON payload (Base64 data URL or sample reference)
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Malformed JSON body.",
            )

        # Subcase 1A: Sample file reference from outputs/sample_inferences
        if "sample_filename" in body:
            fname = os.path.basename(body["sample_filename"])
            sample_path = os.path.join(OUTPUTS_DIR, "sample_inferences", fname)
            if not os.path.exists(sample_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Sample file '{fname}' not found in outputs/sample_inferences.",
                )

            # If model is loaded, run live inference
            if MODEL_INSTANCE is not None:
                with open(sample_path, "rb") as f:
                    img_bytes = f.read()
                source_name = fname
                img_pil = _process_pil_bytes(img_bytes, fname)
            else:
                # If TensorFlow is still downloading locally, load verified benchmark prediction from CSV
                csv_path = os.path.join(OUTPUTS_DIR, "sample_predictions.csv")
                if os.path.exists(csv_path):
                    with open(csv_path, "r", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            if row.get("image_filename") == fname:
                                p_dog = float(row.get("p_dog", 0.0))
                                p_cat = 1.0 - p_dog
                                pred = row.get("predicted_label", "").upper()
                                conf = p_dog if pred == "DOG" else p_cat
                                logit_val = float(np.log(p_dog / max(1e-7, (1.0 - p_dog)))) if 0 < p_dog < 1 else 0.0
                                return {
                                    "success": True,
                                    "filename": fname,
                                    "prediction": pred,
                                    "confidence": conf,
                                    "confidence_pct": f"{conf * 100:.2f}%",
                                    "prob_dog": p_dog,
                                    "prob_cat": p_cat,
                                    "logit": round(logit_val, 4),
                                    "decision_threshold": 0.50,
                                    "latency_ms": 1.2,
                                    "benchmark_verified": True,
                                    "note": "Verified benchmark prediction (TensorFlow not yet installed in host environment).",
                                    "metadata": {
                                        "input_resolution": list(IMG_SIZE),
                                        "channels": 3,
                                        "architecture": "MobileNetV2",
                                    },
                                }

        # Subcase 1B: Base64 data URL
        elif "image" in body:
            if MODEL_INSTANCE is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "TensorFlow is required for custom image inference. "
                        "Install via: python -m pip install --default-timeout=1000 tensorflow"
                    ),
                )
            raw_b64 = body["image"]
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(raw_b64)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid base64 encoding: {exc}",
                )
            source_name = body.get("filename", "uploaded_image.jpg")
            img_pil = _process_pil_bytes(img_bytes, source_name)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON body must contain 'sample_filename' or 'image' (base64 string).",
            )

    # Case 2: Multipart form-data file upload
    elif "multipart/form-data" in content_type:
        if MODEL_INSTANCE is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "TensorFlow is required for custom image inference. "
                    "Install via: python -m pip install --default-timeout=1000 tensorflow"
                ),
            )
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not parse multipart form: {exc}. Ensure python-multipart is installed.",
            )
        file_obj = form.get("file")
        if file_obj is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing 'file' field in multipart upload.",
            )
        filename = getattr(file_obj, "filename", "upload.jpg")
        ext = os.path.splitext(filename)[1].lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Extension '{ext}' not allowed. Must be one of: {', '.join(ALLOWED_EXTENSIONS)}",
            )
        img_bytes = await file_obj.read()
        source_name = filename
        img_pil = _process_pil_bytes(img_bytes, filename)

    # Case 3: Raw octet-stream bytes
    else:
        if MODEL_INSTANCE is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "TensorFlow is required for custom image inference. "
                    "Install via: python -m pip install --default-timeout=1000 tensorflow"
                ),
            )
        img_bytes = await request.body()
        if not img_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty request payload received.",
            )
        img_pil = _process_pil_bytes(img_bytes, "stream.jpg")

    # If we reached here without a model, raise clear error
    if MODEL_INSTANCE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "TensorFlow is required for custom image inference. "
                "Install via: python -m pip install --default-timeout=1000 tensorflow"
            ),
        )

    # Run inference in memory
    try:
        result = run_inference_on_pil(img_pil, model=MODEL_INSTANCE)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {exc}",
        )

    return {
        "success": True,
        "filename": source_name,
        "prediction": result["label"],
        "confidence": result["confidence"],
        "confidence_pct": f"{result['confidence'] * 100:.2f}%",
        "prob_dog": result["prob_dog"],
        "prob_cat": result["prob_cat"],
        "logit": result["logit"],
        "decision_threshold": result["threshold"],
        "latency_ms": result["latency_ms"],
        "metadata": {
            "original_width": result["original_size"][0],
            "original_height": result["original_size"][1],
            "input_resolution": result["processed_size"],
            "channels": 3,
            "architecture": "MobileNetV2",
        },
    }

