"""
FastAPI entrypoint for Flood Risk Prediction API.
"""

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from datetime import datetime

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ErrorResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from api.service import predict_service
from src.config import get_paths
from src.utils.exceptions import InputValidationError, SecurityError, ModelNotFoundError
from src.utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Flood Risk Prediction API",
    version="1.0.0",
    description="Production-ready flood risk classification API with calibrated probabilities and drift detection.",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.exception_handler(InputValidationError)
async def validation_exception_handler(request: Request, exc: InputValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(detail=str(exc), error_type="InputValidationError").model_dump(),
    )


@app.exception_handler(SecurityError)
async def security_exception_handler(request: Request, exc: SecurityError):
    return JSONResponse(
        status_code=401,
        content=ErrorResponse(detail=str(exc), error_type="SecurityError").model_dump(),
    )


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(detail=str(exc), error_type="ModelNotFoundError").model_dump(),
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(detail=str(exc), error_type="ValidationError").model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail="Internal server error", error_type="InternalError").model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    registry_path = get_paths()["models"].parent / "registry.json"
    model_version = "unknown"
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
        model_version = registry.get("model_version", "unknown")
    
    return HealthResponse(
        status="ok",
        model_version=model_version,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """
    Single prediction endpoint.
    
    Returns calibrated flood risk classification with confidence and decision.
    """
    try:
        result = predict_service(req.model_dump())
        return PredictionResponse(**result)
    except InputValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except SecurityError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(req: BatchPredictionRequest):
    """
    Batch prediction endpoint (max 100 samples).
    
    Returns list of predictions with individual results.
    """
    if len(req.predictions) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 100")
    
    results = []
    for pred_req in req.predictions:
        try:
            result = predict_service(pred_req.model_dump())
            results.append(PredictionResponse(**result))
        except Exception as e:
            logger.error(f"Batch prediction failed for sample: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    return BatchPredictionResponse(results=results, count=len(results))


@app.get("/model/info")
async def model_info():
    """Get model metadata from registry."""
    registry_path = get_paths()["models"].parent / "registry.json"
    if not registry_path.exists():
        raise HTTPException(status_code=404, detail="Model registry not found")
    
    with open(registry_path, "r") as f:
        registry = json.load(f)
    return registry


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)