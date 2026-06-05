# generate_data.py
# Generates synthetic Chinook-style customer data
# Run from testing/ folder: python generate_data.py

import numpy  as np
import pandas as pd
import random
import os
from datetime import datetime, timedelta

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
N_CUSTOMERS  = 1000
RANDOM_SEED  = 42
OUTPUT_PATH  = os.path.join("data", "synthetic_customers.csv")

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("=" * 55)
print("SYNTHETIC CHINOOK CUSTOMER DATA GENERATOR")
print("=" * 55)
print(f"  Generating {N_CUSTOMERS} customers...")
print()

# -------------------------------------------------------
# Reference data — mirrors real Chinook distributions
# -------------------------------------------------------

# Country distribution — matches real Chinook
COUNTRIES = {
    "USA"            : 0.22,
    "Canada"         : 0.14,
    "France"         : 0.09,
    "Brazil"         : 0.09,
    "Germany"        : 0.07,
    "United Kingdom" : 0.05,
    "Portugal"       : 0.04,
    "Czech Republic" : 0.04,
    "India"          : 0.03,
    "Belgium"        : 0.02,
    "Austria"        : 0.02,
    "Norway"         : 0.02,
    "Denmark"        : 0.02,
    "Ireland"        : 0.02,
    "Hungary"        : 0.02,
    "Finland"        : 0.02,
    "Netherlands"    : 0.02,
    "Italy"          : 0.02,
    "Chile"          : 0.02,
    "Other"          : 0.03,
}

# Genre distribution — Rock dominates like real data
GENRES = {
    "Rock"              : 0.35,
    "Latin"             : 0.12,
    "Metal"             : 0.10,
    "Alternative & Punk": 0.09,
    "Jazz"              : 0.07,
    "Blues"             : 0.06,
    "Classical"         : 0.05,
    "R&B/Soul"          : 0.04,
    "Reggae"            : 0.04,
    "Pop"               : 0.04,
    "Hip Hop/Rap"       : 0.02,
    "Easy Listening"    : 0.02,
}

SUPPORT_REPS   = [3, 4, 5]
MEDIA_TYPES    = ["MPEG audio file",
                  "Protected AAC audio file",
                  "AAC audio file"]

# Date range for purchases
START_DATE = datetime(2020, 1, 1)
END_DATE   = datetime(2024, 12, 31)
DATE_RANGE = (END_DATE - START_DATE).days

# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def weighted_choice(options_dict):
    """Pick a random key weighted by its probability value"""
    keys   = list(options_dict.keys())
    weights= list(options_dict.values())
    return random.choices(keys, weights=weights, k=1)[0]

def random_date(start, range_days):
    """Generate a random date within range"""
    return start + timedelta(days=random.randint(0, range_days))

def generate_invoice_dates(n_invoices, last_date):
    """
    Generate n invoice dates ending at last_date
    Spaced realistically — not perfectly uniform
    """
    dates = sorted([
        last_date - timedelta(
            days=random.randint(0, 365 * 3)
        )
        for _ in range(n_invoices)
    ])
    return dates

def calculate_invoice_total(n_tracks, genre):
    """
    Calculate realistic invoice total based on
    number of tracks and genre
    Some genres have higher priced tracks
    """
    # Base price per track
    base_prices = {
        "Classical"         : 1.29,
        "Jazz"              : 1.19,
        "Rock"              : 0.99,
        "Metal"             : 0.99,
        "Alternative & Punk": 0.99,
        "Latin"             : 0.99,
        "Blues"             : 1.09,
        "R&B/Soul"          : 1.09,
        "Reggae"            : 0.99,
        "Pop"               : 0.99,
        "Hip Hop/Rap"       : 0.99,
        "Easy Listening"    : 1.09,
    }

    base_price = base_prices.get(genre, 0.99)

    # Add slight random variation per track
    total = sum([
        base_price + random.uniform(-0.10, 0.20)
        for _ in range(n_tracks)
    ])

    return round(total, 2)

# -------------------------------------------------------
# Generate customers
# -------------------------------------------------------
customers = []

for customer_id in range(1, N_CUSTOMERS + 1):

    # Basic attributes
    country       = weighted_choice(COUNTRIES)
    favorite_genre= weighted_choice(GENRES)
    support_rep   = random.choice(SUPPORT_REPS)

    # Purchase behavior
    # Use negative binomial for realistic frequency distribution
    # Most customers buy a few times, some buy many times
    n_invoices = max(1, int(np.random.negative_binomial(
        n   = 3,
        p   = 0.3,
    )))
    n_invoices = min(n_invoices, 20)   # cap at 20

    # Last purchase date
    last_purchase = random_date(START_DATE, DATE_RANGE)

    # Generate all invoice dates
    invoice_dates = generate_invoice_dates(
        n_invoices, last_purchase
    )
    first_purchase = invoice_dates[0]

    # Generate tracks per invoice
    # Most invoices have 1-5 tracks
    tracks_per_invoice = [
        max(1, int(np.random.negative_binomial(n=2, p=0.4)))
        for _ in range(n_invoices)
    ]
    tracks_per_invoice = [min(t, 14) for t in tracks_per_invoice]
    total_tracks = sum(tracks_per_invoice)

    # Calculate spend per invoice
    invoice_totals = [
        calculate_invoice_total(t, favorite_genre)
        for t in tracks_per_invoice
    ]
    total_spent     = round(sum(invoice_totals), 2)
    avg_invoice_val = round(np.mean(invoice_totals), 2)

    # Recency
    latest_date  = datetime(2024, 12, 31)
    recency_days = (latest_date - last_purchase).days
    tenure_days  = (last_purchase - first_purchase).days

    # RecencyScore — same bins as real data processing
    if recency_days <= 365:
        recency_score = 0    # Recent
    elif recency_days <= 730:
        recency_score = 1    # Lapsing
    else:
        recency_score = 2    # Churned

    customers.append({
        "CustomerId"     : customer_id,
        "Country"        : country,
        "SupportRepId"   : support_rep,
        "TotalSpent"     : total_spent,
        "NumInvoices"    : n_invoices,
        "NumTracks"      : total_tracks,
        "AvgInvoiceValue": avg_invoice_val,
        "FavoriteGenre"  : favorite_genre,
        "FirstPurchase"  : first_purchase.strftime("%Y-%m-%d"),
        "LastPurchase"   : last_purchase.strftime("%Y-%m-%d"),
        "RecencyDays"    : recency_days,
        "TenureDays"     : tenure_days,
        "RecencyScore"   : recency_score,
    })

# -------------------------------------------------------
# Build DataFrame
# -------------------------------------------------------
df = pd.DataFrame(customers)

# -------------------------------------------------------
# Verify distributions
# -------------------------------------------------------
print("=== Generated Data Summary ===")
print(f"  Rows     : {len(df)}")
print(f"  Columns  : {len(df.columns)}")
print()

print("TotalSpent distribution:")
print(f"  Min    : ${df['TotalSpent'].min():.2f}")
print(f"  Max    : ${df['TotalSpent'].max():.2f}")
print(f"  Mean   : ${df['TotalSpent'].mean():.2f}")
print(f"  Median : ${df['TotalSpent'].median():.2f}")
print(f"  Std    : ${df['TotalSpent'].std():.2f}")
print(f"  Skew   : {df['TotalSpent'].skew():.3f}")
print()

print("NumInvoices distribution:")
print(f"  Min    : {df['NumInvoices'].min()}")
print(f"  Max    : {df['NumInvoices'].max()}")
print(f"  Mean   : {df['NumInvoices'].mean():.1f}")
print(f"  Std    : {df['NumInvoices'].std():.2f}")
print()

print("RecencyScore distribution:")
dist = df["RecencyScore"].value_counts().sort_index()
labels = {0: "Recent", 1: "Lapsing", 2: "Churned"}
for score, count in dist.items():
    pct = count / len(df) * 100
    print(f"  {score} ({labels[score]:<8}) : "
          f"{count:>4} customers ({pct:.1f}%)")
print()

print("Top 5 countries:")
print(df["Country"].value_counts().head())
print()

print("Top 5 genres:")
print(df["FavoriteGenre"].value_counts().head())
print()

print("Support rep distribution:")
print(df["SupportRepId"].value_counts().sort_index())
print()

# -------------------------------------------------------
# Compare with original data
# -------------------------------------------------------
print("=== Comparison with Original Data ===")
print(f"  {'Metric':<20} {'Original':>12} {'Synthetic':>12}")
print(f"  {'-'*46}")
print(f"  {'N customers':<20} {'59':>12} {len(df):>12}")
print(f"  {'Spend min':<20} {'$36.64':>12} "
      f"${df['TotalSpent'].min():>10.2f}")
print(f"  {'Spend max':<20} {'$49.62':>12} "
      f"${df['TotalSpent'].max():>10.2f}")
print(f"  {'Spend std':<20} {'$2.08':>12} "
      f"${df['TotalSpent'].std():>10.2f}")
print(f"  {'Invoice freq':<20} {'6-7':>12} "
      f"{df['NumInvoices'].min()}-"
      f"{df['NumInvoices'].max():>9}")
print()

# -------------------------------------------------------
# Save
# -------------------------------------------------------
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"✓ Saved to {OUTPUT_PATH}")
print()
print("=" * 55)
print("NEXT STEPS")
print("=" * 55)
print("""
  1. Copy synthetic_customers.csv to your notebooks/data/
  2. Open 02_eda.ipynb and load synthetic_customers.csv
  3. Rerun Stage 2 → Stage 7 with new data
  4. Expected improvements:
       R²        : should be positive (0.4-0.7)
       CV RMSE   : wider range → better signal
       MAE       : may increase in dollars but
                   more meaningful as % of spend range
  5. Compare results with original 59-customer run
""")
print("=" * 55)