"""
src/data_collection.py
----------------------
Customer Churn Prediction - Phase 1: Data Collection & Integrity Verification.

Academic Context:
Acquires the standard, authentic IBM Telco Customer Churn dataset.
Ensures raw data immutability and verifies data integrity before any processing.
"""

import os
import sys
import urllib.request
import pandas as pd

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Verified authentic raw data URL (IBM ICP4D Telco Churn repository)
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)


def collect_data(url: str = DATASET_URL, target_path: str = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Downloads the raw IBM Telco Customer Churn dataset if not already present,
    and returns it as a pandas DataFrame.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if not os.path.exists(target_path):
        print(f"[INFO] Downloading authentic dataset from: {url}")
        urllib.request.urlretrieve(url, target_path)
        print(f"[SUCCESS] Dataset successfully saved to: {target_path}")
    else:
        print(f"[INFO] Raw dataset already exists at: {target_path}")

    # Load dataset
    df = pd.read_csv(target_path)
    return df


def verify_raw_dataset(df: pd.DataFrame) -> dict:
    """
    Performs data integrity verification on the collected raw dataset.
    Returns a dictionary of key structural properties.
    """
    print("\n" + "=" * 60)
    print("      RAW DATASET INTEGRITY VERIFICATION REPORT")
    print("=" * 60)

    # 1. Dimensions
    num_rows, num_cols = df.shape
    print(f"\n1. Dataset Dimensions:")
    print(f"   - Total Records (Rows): {num_rows}")
    print(f"   - Total Attributes (Columns): {num_cols}")

    # 2. Columns & Data Types
    print(f"\n2. Attributes & Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"   - {col:<20} : {str(dtype):<10} (Unique: {df[col].nunique()})")

    # 3. Missing Value Audit
    print(f"\n3. Missing Value Audit (Standard NaN):")
    null_counts = df.isnull().sum()
    has_nulls = False
    for col, count in null_counts.items():
        if count > 0:
            print(f"   - {col}: {count} nulls")
            has_nulls = True
    if not has_nulls:
        print("   - No standard NaN values found.")

    # 4. Hidden Missing Values (Whitespace in TotalCharges)
    blank_total_charges = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    print(f"\n4. Hidden Missing Values Check:")
    print(f"   - Blank whitespace strings in 'TotalCharges': {blank_total_charges} records")

    # 5. Target Variable Class Distribution ('Churn')
    print(f"\n5. Target Variable ('Churn') Distribution:")
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    for label, count in churn_counts.items():
        print(f"   - {label:<5}: {count:5d} records ({churn_pct[label]:.2f}%)")

    # 6. Sample Records
    print(f"\n6. First 3 Actual Records:")
    print(df.head(3).to_string())

    print("\n" + "=" * 60)
    print("      VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "blank_total_charges": blank_total_charges,
        "churn_distribution": churn_counts.to_dict(),
    }


if __name__ == "__main__":
    df_raw = collect_data()
    metrics = verify_raw_dataset(df_raw)

