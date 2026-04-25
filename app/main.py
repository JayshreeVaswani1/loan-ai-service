"""
FastAPI ML Service - Fraud Detection
Loads trained Isolation Forest model and serves predictions
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import logging
import os
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Loan AI Service",
    description="AI-powered fraud detection and risk scoring for loan origination system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage (singleton pattern)
class ModelSingelton:
    """Singelton to load model once and reuse"""
    _model = None
    _metrics = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._load_model()
        return cls._model

    @classmethod
    def get_metrics(cls):
        if cls._metrics is None:
            cls._load_model()
        return cls._metrics

    @classmethod
    def _load_model(cls):
        """Load model and metrics from disk"""
        try:
            # Get the models directory (relative to this file)
            base_dir = Path(__file__).parent.parent
            models_dir = base_dir / "models"

            model_path = models_dir / "fraud_detector_latest.pkl"
            metrics_path = models_dir / "metrics_latest.json"

            logger.info(f"Loading model from: {model_path}")

            # Load the trained model
            cls._model = joblib.load(model_path)
            logger.info("Model loaded successfully")

            # Load metrics
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    cls._metrics = json.load(f)
                logger.info(f"Metrics loaded: {metrics_path}")
            else:
                logger.warn("Metrics file not found")
                cls._metrics = {}
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

@app.on_event("startup")
async def startup_event():
    """Load model when the application starts"""
    logger.info("Starting Fraud Detection ML Service")
    ModelSingelton.get_model()
    logger.info("ML Service ready!")

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "Loan AI Service",
        "status": "healthy",
        "model_loaded" : ModelSingelton._model is not None,
        "version": "1.0.0",
        "description": "AI-powered fraud detection and risk scoring"
    }

@app.get("/health")
def health():
    """Detailed health check"""
    model = ModelSingelton.get_model()
    metrics = ModelSingelton.get_metrics()

    return {
        "status": "healthy",
        "service": "ai-service",
        "model" : {
            "loaded" : model is not None,
            "type" : str(type(model).__name__) if model else None,
        },
        "metrics" : metrics,
        "endpoints": [
            "GET /",
            "GET /health",
            "POST /api/detect-fraud"
        ]
    }

@app.get("/metrics")
def get_metrics():
    """Get model performance metrics"""
    return ModelSingelton.get_metrics()

# Include API routes
from app.api import routes
app.include_router(routes.router)