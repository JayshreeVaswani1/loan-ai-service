from pydantic import BaseModel, Field
from typing import Optional, List

class FraudDetectionRequest(BaseModel):
    """Request schema for fraud detection endpoint."""
    customer_id: str = Field(..., description="Customer unique identifier", example="abc-123")
    transaction_amount: float = Field(..., gt=0, description="Transaction amount (must be positive)", example=50000.0)
    location: Optional[str] = Field(None, description="Transaction location", example="Nigeria")
    time_of_day: Optional[str] = Field(None, description="Time of transaction (HH:MM:SS)", example="03:00:00")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "abc-123",
                "transaction_amount": 50000,
                "location": "Nigeria",
                "time_of_day": "03:00:00"
            }
        }

class FraudDetectionResponse(BaseModel):
    """Response schema for fraud detection endpoint."""
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud (0.0 to 1.0)")
    risk_score: int = Field(..., ge=0, le=100, description="Risk score (0 to 100)")
    recommendation: str = Field(..., description="Recommendation: APPROVE, REVIEW, or BLOCK")
    reasons: List[str] = Field(..., description="List of reasons for the risk score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "fraud_probability": 0.85,
                "risk_score": 85,
                "recommendation": "BLOCK",
                "reasons": ["Unusually high transaction amount", "High-risk location: Nigeria"]
            }
        }
