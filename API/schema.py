# api/schema.py
from pydantic import BaseModel, Field
from typing   import Optional

# -------------------------------------------------------
# Input schema — what the API expects from the client
# -------------------------------------------------------
class CustomerFeatures(BaseModel):
    """
    Features needed to predict customer lifetime value.
    All fields match what our model was trained on.
    """

    recency_days    : float = Field(
        ...,
        ge          = 0,
        le          = 9999,
        description = "Days since last purchase"
    )

    recency_score   : int = Field(
        ...,
        ge          = 0,
        le          = 2,
        description = "0=Recent, 1=Lapsing, 2=Churned"
    )

    country         : str = Field(
        ...,
        description = "Customer country"
    )

    favorite_genre  : str = Field(
        ...,
        description = "Customer favorite music genre"
    )

    support_rep_id  : str = Field(
        ...,
        description = "Support representative ID (3, 4, or 5)"
    )

    class Config:
        # Example shown in API docs
        json_schema_extra = {
            "example": {
                "recency_days"   : 120,
                "recency_score"  : 0,
                "country"        : "USA",
                "favorite_genre" : "Rock",
                "support_rep_id" : "3"
            }
        }

# -------------------------------------------------------
# Output schema — what the API returns to the client
# -------------------------------------------------------
class PredictionResponse(BaseModel):
    """
    Prediction result returned to the client.
    """
    predicted_spend  : float = Field(
        description="Predicted customer lifetime value in dollars"
    )
    model            : str = Field(
        description="Model used for prediction"
    )
    confidence       : str = Field(
        description="Average prediction error range"
    )
    input_received   : dict = Field(
        description="Echo of input features for verification"
    )

# -------------------------------------------------------
# Health check schema
# -------------------------------------------------------
class HealthResponse(BaseModel):
    status      : str
    model       : str
    version     : str