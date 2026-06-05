This project is doing Customer Lifetime Value (CLV) prediction. Here's what it does:

------------------
What it predicts
------------------
Predicts how much money a customer will spend in total (TotalSpent in dollars)
Based on the Chinook music store database (digital music sales)

-----------------------
How it defines CLV
-----------------------
Uses RFM analysis — a classic CLV technique:

Recency → how recently a customer bought
Frequency → how often they buy
Monetary → how much they spend

-----------------------------------
What the pipeline does
StageWhat happensStage 1Extracts customer transaction data from SQLiteStage 2Explores spending patterns, distributionsStage 3Builds 34 RFM-based features per customerStage 4Splits data, scales featuresStage 5Trains 5 models to predict future spendStage 6Tunes the best model (Ridge regression)

-----------------
Results
-----------------
Best model: Ridge Regression
Average prediction error: $13.95 per customer
Spend range predicted: $0.89 to $92.75

---------------------
Business use case
---------------------

Identify high-value customers
Target marketing spend wisely
Predict revenue from existing customers

-------------------------------------------------------
-------------------------------------------------------
Stage 1 — Data Ingestion
  Connected to SQLite database
  Merged 11 tables into master DataFrames
  Built customer-level aggregations

Stage 2 — EDA
  Distributions, correlations, outliers
  Identified data leakage in AvgInvoiceValue
  Made informed decisions on features

Stage 3 — Feature Engineering
  Removed leakage and zero variance features
  Built RecencyScore, country grouping
  One Hot Encoded categoricals
  Saved clean feature matrix X and target y

Stage 4 — Preprocessing
  ColumnTransformer with StandardScaler
  Stratified train/test split
  Saved fitted preprocessor for production

Stage 5 — Model Training
  Trained 5 models with cross validation
  Ridge won with MAE $2.03
  Understood why R² was unreliable here

Stage 6 — Hyperparameter Tuning
  GridSearchCV on Ridge → alpha=10 → MAE $1.82
  RandomizedSearchCV on XGBoost → MAE $1.81
  Saved production model

Stage 7 — FastAPI Endpoint
  /predict    → single customer prediction
  /predict/batch → multiple customers
  /model/info → model details
  /docs       → Swagger UI