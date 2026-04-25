from fastapi import APIRouter
from app.schemas.requests import FraudDetectionRequest, FraudDetectionResponse
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["AI Services"])

@router.post("/detect-fraud", response_model=FraudDetectionResponse)
def detect_fraud(request: FraudDetectionRequest):
    """
    Detect fraud using trained Isolation Forest ML model
    Replaced rule-based logic with real ML predictions.
    """
    try:
        start_time = time.time()

        # Import model singleton
        from app.main import ModelSingelton
        model = ModelSingelton.get_model()

        if model is None:
            raise HTTPException(status_code=503, detail="ML Model not loaded")

    #===================
    # FEATURE MAPPING
    #===================

        # Extract hour of the day from time_of_day
        hour_of_day = 0
        if request.time_of_day:
            try:
                hour_of_day = int(request.time_of_day.split(":")[0])
            except:
                hour_of_day = 12    # Default to noon if parsing fails

        # Map location to risk score
        high_risk_locations = ["Nigeria", "Somalia", "Afghanistan", "Yemen",
                           "Syria", "North Korea", "Iran"]
        location_risk_score = 90.0 if request.location in high_risk_locations else 20.0

        customer_age = getattr(request, 'customer_age', 35)     # Default age
        day_of_week = getattr(request, 'day_of_week', 2)        # Default Tuesday
        transaction_velocity = getattr(request, 'transaction_velocity', 3)  # Default 3 txns

        # Prepare features in exact order model expects
        features = np.array([[
            request.transaction_amount,
            hour_of_day,
            day_of_week,
            location_risk_score,
            customer_age,
            transaction_velocity
        ]])

        logger.info(f"Predicting for features: {features}")

    # =====================
    # ML MODEL PREDICTION
    # =====================

        # Get prediction (-1 = fraud, 1 = normal)
        prediction = model.predict(features)[0]

        # Get anomaly score (more negative = more anomalous)
        anomaly_score = model.score_samples(features)[0]

        # Convert to interpretable samples
        is_fraud = (prediction == -1)

        # Convert anomaly score to 0-100 risk score
        risk_score = min(100, max(0, (-anomaly_score) * 200))

        # Calculate fraud probability (0.0 to 1.0)
        fraud_probability = risk_score / 100.0

        # Calculate the prediction time
        prediction_time_ms = (time.time() - start_time) * 1000

    # ============================================
    # BUSINESS LOGIC & RECOMMENDATIONS
    # ============================================

        # Determine recommendation based on risk score
        if risk_score >= 70:
            recommendation = "BLOCK"
        elif risk_score >=40:
            recommendation = "REVIEW"
        else:
            recommendation = "APPROVE"

        # Generate reasons based on model decision
        reasons = []

        if is_fraud:
            if request.transaction_amount > 100000:
                reasons.append("Unusually high transaction amount")
            elif request.transaction_amount > 50000:
                reasons.append("High transaction amount")

            if hour_of_day >= 0 and hour_of_day <= 5:
                reasons.append("Unusual transaction time (late night)")

            if request.location in high_risk_locations:
                reasons.append(f"High-risk location : {request.location}")

            if transaction_velocity > 10:
                reasons.append("High transaction velocity (many recent transactions)")

            # Add ML-specific reason
            reasons.append(f"ML model detected anomalous pattern (confidence: {risk_score:.1f}%)")
        else:
            reasons.append("Transaction appears normal based on ML analysis")
            if risk_score > 30:
                reasons.append(f"Some anomalies detected (risk: {risk_score:.1f}%)")

        if not reasons:
            reasons = ["Transaction pattern is normal"]

        logger.info(
            f"Prediction: fraud={is_fraud}, risk={risk_score:.1f}%, "
            f"recommendation={recommendation}, time={prediction_time_ms:.2f}ms"
        )

        # Return response in your existing format
        return FraudDetectionResponse(
            fraud_probability=round(fraud_probability, 3),
            risk_score=int(risk_score),
            recommendation=recommendation,
            reasons=reasons
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)

        # Fallback to safe recommendation on error
        return FraudDetectionResponse(
            fraud_probability=0.5,
            risk_score=50,
            recommendation="REVIEW",
            reasons=["Model prediction failed - manual review recommended", str(e)]
        )


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
