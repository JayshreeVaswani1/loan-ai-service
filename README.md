# Loan AI Service

Fraud detection microservice with ML-based risk assessment. FastAPI backend integrating with Spring Boot loan platform.

## Stack
Python 3.11 · FastAPI 0.109 · Pydantic 2.5 · scikit-learn 1.4 · Uvicorn

## Quick Start
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Docs: `http://localhost:8000/docs`

## Architecture
┌──────────────┐         ┌──────────────┐
│ Spring Boot  │ ◄─REST─►│  FastAPI     │
│ Port: 8080   │         │  Port: 8000  │
└──────────────┘         └──────┬───────┘
│
┌──────▼───────┐
│  ML Models   │
│  (Isolation  │
│   Forest)    │
└──────────────┘
Stateless design. Model persistence with joblib. Horizontal scaling ready.

## API
**POST** `/api/detect-fraud`
```json
Request:  {"customer_id": "x", "transaction_amount": 50000, "location": "...", "time_of_day": "03:00"}
Response: {"fraud_probability": 0.85, "risk_score": 85, "recommendation": "BLOCK", "reasons": [...]}
```
**GET** `/health` - Service status

## Implementation
**Current (v1.0):** Rule-based detection (baseline)  
**Next (v2.0):** Isolation Forest + Random Forest models  
**Pipeline:** Feature engineering → Model training → Evaluation (F1/precision/recall) → Deployment

## Structure
app/
├── main.py           # FastAPI app, CORS config
├── api/routes.py     # Endpoints
├── schemas/          # Pydantic models
├── models/           # ML models (v2)
└── services/         # Business logic (v2)
## Development
```bash
pytest                              # Tests
docker build -t loan-ai-service .   # Container
```

## Integration
Part of [loan-origination-system](https://github.com/JayshreeVaswani1/loan-origination-system) microservices platform.

---
*Built for financial intelligence · Production ML deployment*
