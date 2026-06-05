# api/main.py
from fastapi            import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from schema import (CustomerFeatures,
                    PredictionResponse,
                    HealthResponse)
from model  import (predict_customer_spend,
                    model_name,
                    model_mae,
                    feature_cols)

# -------------------------------------------------------
# Create FastAPI app
# -------------------------------------------------------
app = FastAPI(
    title       = "Chinook CLV Prediction API",
    description = """
    Predicts Customer Lifetime Value (CLV) for Chinook
    music store customers.

    ## How to use
    1. Send a POST request to /predict with customer features
    2. Receive predicted spend in dollars

    ## Model
    - Algorithm  : Ridge Regression (L2 regularization)
    - Trained on : 47 customers
    - Test MAE   : $1.82
    - Features   : 11 (recency, country, genre, support rep)
    """,
    version     = "1.0.0",
)

# -------------------------------------------------------
# CORS middleware
# Allows the API to be called from a browser
# or from other services
# -------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # in production restrict this
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# -------------------------------------------------------
# Endpoints
# -------------------------------------------------------

@app.get("/",
         summary="Root — API info")
def root():
    """
    Returns basic API information.
    """
    return {
        "name"        : "Chinook CLV Prediction API",
        "version"     : "1.0.0",
        "model"       : model_name,
        "model_mae"   : f"${model_mae:.2f}",
        "endpoints"   : {
            "health"  : "GET  /health",
            "predict" : "POST /predict",
            "docs"    : "GET  /docs",
        }
    }

@app.get("/health",
         response_model = HealthResponse,
         summary        = "Health check")
def health():
    """
    Health check endpoint.
    Use this to verify the API and model are loaded correctly.
    """
    return HealthResponse(
        status  = "healthy",
        model   = model_name,
        version = "1.0.0"
    )


@app.post("/predict",
          response_model = PredictionResponse,
          summary        = "Predict customer lifetime value")
def predict(features: CustomerFeatures):
    """
    Predicts total spend for a customer based on their features.

    ## Input
    - **recency_days**   : Days since last purchase (0-9999)
    - **recency_score**  : 0=Recent, 1=Lapsing, 2=Churned
    - **country**        : Customer country
    - **favorite_genre** : Customer favorite music genre
    - **support_rep_id** : Support rep ID (3, 4, or 5)

    ## Output
    - **predicted_spend** : Predicted total spend in dollars
    - **model**           : Model used for prediction
    - **confidence**      : Average error range
    - **input_received**  : Echo of your input for verification
    """
    try:
        result = predict_customer_spend(features)
        return PredictionResponse(**result)

    except ValueError as e:
        # Known validation errors — return 422
        raise HTTPException(
            status_code = 422,
            detail      = str(e)
        )
    except Exception as e:
        # Unknown errors — return 500
        raise HTTPException(
            status_code = 500,
            detail      = f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch",
          summary = "Predict for multiple customers")
def predict_batch(customers: list[CustomerFeatures]):
    """
    Predicts spend for multiple customers in one request.
    Maximum 100 customers per request.
    """
    if len(customers) > 100:
        raise HTTPException(
            status_code = 400,
            detail      = "Maximum 100 customers per batch request"
        )

    results = []
    errors  = []

    for i, customer in enumerate(customers):
        try:
            result = predict_customer_spend(customer)
            results.append({
                "index"          : i,
                "predicted_spend": result["predicted_spend"],
                "confidence"     : result["confidence"],
            })
        except Exception as e:
            errors.append({
                "index" : i,
                "error" : str(e)
            })

    return {
        "total_requested" : len(customers),
        "successful"      : len(results),
        "failed"          : len(errors),
        "predictions"     : results,
        "errors"          : errors,
        "model"           : model_name,
    }


@app.get("/model/info",
         summary = "Model information and features")
def model_info():
    """
    Returns detailed information about the model
    including feature names and training metrics.
    """
    return {
        "model"         : model_name,
        "version"       : "1.0.0",
        "target"        : "TotalSpent (dollars)",
        "test_mae"      : f"${model_mae:.2f}",
        "features"      : feature_cols,
        "n_features"    : len(feature_cols),
        "valid_inputs"  : {
            "recency_score" : {
                0: "Recent (bought within last 94 days)",
                1: "Lapsing (95-251 days)",
                2: "Churned (252+ days)"
            },
            "country"       : [
                "USA", "Canada", "France",
                "Brazil", "Germany", "Other"
            ],
            "support_rep_id": ["3", "4", "5"],
        }
    }


# -------------------------------------------------------
# Run server
# -------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = True     # auto-restart on code changes
    )