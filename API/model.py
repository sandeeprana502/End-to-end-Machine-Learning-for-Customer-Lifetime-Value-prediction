# api/model.py
import numpy  as np
import pandas as pd
import joblib
import json
import os
from schema import CustomerFeatures

# -------------------------------------------------------
# Paths — go up one level from api/ to testing/
# -------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models/data")
DATA_DIR    = os.path.join(BASE_DIR, "data")

# -------------------------------------------------------
# Load model artifacts at startup
# These are loaded ONCE when the API starts
# not on every request — much faster
# -------------------------------------------------------
print("Loading model artifacts...")

production_model = joblib.load(
    os.path.join(MODELS_DIR, "production_model.joblib")
)
preprocessor = joblib.load(
    os.path.join(MODELS_DIR, "preprocessor.joblib")
)

with open(os.path.join(MODELS_DIR, "tuning_summary.json")) as f:
    tuning_summary = json.load(f)

with open(os.path.join(DATA_DIR, "feature_meta.json")) as f:
    feature_meta = json.load(f)

feature_cols = feature_meta["feature_cols"]
model_name   = tuning_summary["production_model"]
model_mae    = tuning_summary["tuned_models"][model_name]["test_mae"]

print(f"✓ Model loaded      : {model_name}")
print(f"✓ Features expected : {len(feature_cols)}")
print(f"✓ Model MAE         : ${model_mae:.2f}")

# -------------------------------------------------------
# Known categories from training data
# CRITICAL: must match exactly what was seen in Stage 3
# If a new category arrives we group it as Other
# -------------------------------------------------------
MAJOR_COUNTRIES = ["USA", "Canada", "France",
                   "Brazil", "Germany"]

KNOWN_GENRES    = ["Rock", "Metal"]   # others = base category

KNOWN_REPS      = ["4", "5"]          # rep 3 = base category

# -------------------------------------------------------
# Prediction function
# -------------------------------------------------------
def predict_customer_spend(features: CustomerFeatures) -> dict:
    """
    Takes customer features, encodes them exactly as done
    in Stage 3, scales with Stage 4 preprocessor,
    and returns predicted spend in dollars.
    """

    # -------------------------------------------------------
    # Step 1: Group rare country → Other
    # Must match Stage 3 logic exactly
    # -------------------------------------------------------
    country_grouped = (
        features.country
        if features.country in MAJOR_COUNTRIES
        else "Other"
    )

    # -------------------------------------------------------
    # Step 2: Build OHE columns manually
    # We cannot use pd.get_dummies on a single row
    # because it only creates columns for values present
    # in that row — missing columns would crash the model
    # Instead we manually set each column to 0 or 1
    # -------------------------------------------------------

    # Country OHE (base = Brazil)
    country_canada  = int(country_grouped == "Canada")
    country_france  = int(country_grouped == "France")
    country_germany = int(country_grouped == "Germany")
    country_other   = int(country_grouped == "Other")
    country_usa     = int(country_grouped == "USA")

    # Genre OHE (base = first genre alphabetically)
    genre_metal = int(features.favorite_genre == "Metal")
    genre_rock  = int(features.favorite_genre == "Rock")

    # SupportRepId OHE (base = rep 3)
    rep_4 = int(features.support_rep_id == "4")
    rep_5 = int(features.support_rep_id == "5")

    # -------------------------------------------------------
    # Step 3: Build feature row in exact same column order
    # as training data — order matters for the model
    # -------------------------------------------------------
    feature_row = {
        "RecencyDays"              : features.recency_days,
        "RecencyScore"             : features.recency_score,
        "Country_grouped_Canada"   : country_canada,
        "Country_grouped_France"   : country_france,
        "Country_grouped_Germany"  : country_germany,
        "Country_grouped_Other"    : country_other,
        "Country_grouped_USA"      : country_usa,
        "FavoriteGenre_Metal"      : genre_metal,
        "FavoriteGenre_Rock"       : genre_rock,
        "SupportRepId_4"           : rep_4,
        "SupportRepId_5"           : rep_5,
    }

    # Convert to DataFrame — single row
    df_input = pd.DataFrame([feature_row])

    # -------------------------------------------------------
    # Step 4: Verify all expected columns are present
    # and in correct order
    # -------------------------------------------------------
    missing_cols = set(feature_cols) - set(df_input.columns)
    if missing_cols:
        raise ValueError(
            f"Missing columns after encoding: {missing_cols}"
        )

    # Reorder columns to exactly match training order
    df_input = df_input[feature_cols]

    # -------------------------------------------------------
    # Step 5: Scale using fitted preprocessor
    # Same scaler fitted in Stage 4 — never refit
    # -------------------------------------------------------
    X_scaled = preprocessor.transform(df_input)

    # -------------------------------------------------------
    # Step 6: Predict
    # -------------------------------------------------------
    prediction = production_model.predict(X_scaled)[0]

    # Clip to training range
    prediction_clipped = float(np.clip(prediction, 36.64, 49.62))

    return {
        "predicted_spend" : round(prediction_clipped, 2),
        "model"           : model_name,
        "confidence"      : f"±${model_mae:.2f}",
        "input_received"  : features.model_dump(),
    }