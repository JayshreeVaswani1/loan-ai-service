from fastapi import APIRouter
from app.schemas.requests import FraudDetectionRequest, FraudDetectionResponse

router = APIRouter(prefix="/api", tags=["AI Services"])

@router.post("/detect-fraud", response_model=FraudDetectionResponse)
def detect_fraud(request: FraudDetectionRequest):
    """Detect fraud using rule-based logic (will be replaced with ML in Days 10-11)."""
    
    fraud_probability = 0.0
    reasons = []
    
    # RULE 1: Check transaction amount
    if request.transaction_amount > 100000:
        fraud_probability += 0.3
        reasons.append("Unusually high transaction amount")
    elif request.transaction_amount > 50000:
        fraud_probability += 0.15
        reasons.append("High transaction amount")
    
    # RULE 2: Check location
    high_risk_locations = ["Nigeria", "Somalia", "Afghanistan", "Yemen", "Syria", "North Korea", "Iran"]
    if request.location and request.location in high_risk_locations:
        fraud_probability += 0.4
        reasons.append(f"High-risk location: {request.location}")
    
    # RULE 3: Check transaction time
    if request.time_of_day:
        try:
            hour = int(request.time_of_day.split(":")[0])
            if 0 <= hour <= 5:
                fraud_probability += 0.2
                reasons.append("Unusual transaction time (late night)")
        except:
            pass
    
    fraud_probability = min(fraud_probability, 1.0)
    risk_score = int(fraud_probability * 100)
    
    if risk_score >= 70:
        recommendation = "BLOCK"
    elif risk_score >= 40:
        recommendation = "REVIEW"
    else:
        recommendation = "APPROVE"
    
    if not reasons:
        reasons = ["Transaction appears normal"]
    
    return FraudDetectionResponse(
        fraud_probability=fraud_probability,
        risk_score=risk_score,
        recommendation=recommendation,
        reasons=reasons
    )
