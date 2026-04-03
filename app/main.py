from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes

app = FastAPI(
    title="Loan AI Service",
    description="AI-powered fraud detection and risk scoring for loan origination system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(routes.router)

@app.get("/")
def read_root():
    return {
        "service": "Loan AI Service",
        "status": "running",
        "version": "1.0.0",
        "description": "AI-powered fraud detection and risk scoring"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ai-service",
        "models_loaded": False,
        "endpoints": [
            "GET /",
            "GET /health",
            "POST /api/detect-fraud"
        ]
    }
