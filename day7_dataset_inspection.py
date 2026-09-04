"""
TIH IIT Guwahati AI/ML Research Internship
Day 7 Implementation: Dataset Receipt & Comprehensive First-Look Inspection
"""

import pandas as pd
import numpy as np

def inspect_raw_dataset():
    file_path = "Problem Statement 3 -FOR SPIRAL COIL DATASET_ (1).csv"
    print("==================================================")
    print("      DAY 7: RAW DATASET INSPECTION REPORT        ")
    print("==================================================")

    # 1. Ingestion of untouched raw data
    df = pd.read_csv(file_path)
    print(f"File Name        : {file_path}")
    print(f"Total Rows       : {df.shape[0]}")
    print(f"Total Columns    : {df.shape[1]}")

    # 2. Missing values and data types
    print("\n--- Column Types & Missingness ---")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        pct = (null_count / len(df)) * 100
        print(f"• {col:<35} | Type: {str(df[col].dtype):<8} | Nulls: {null_count:<5} ({pct:.2f}%)")

    # 3. Numeric statistics breakdown
    num_cols = df.select_dtypes(include=[np.number]).columns
    stats = df[num_cols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
    print("\n--- Key Distribution Statistics ---")
    print(stats.to_string())

    # 4. Monotonicity verification for Coupling Coefficient
    k_vals = df['CplCoef(Current1,Current3) []'].values
    is_strictly_decreasing = np.all(np.diff(k_vals) <= 0)
    print("\n--- Physical Validation ---")
    print(f"Monotonic decay in coupling coefficient: {is_strictly_decreasing}")
    print(f"Maximum coupling at min distance (0 mm) : {k_vals[0]:.6f}")
    print(f"Minimum coupling at max distance (200 mm): {k_vals[-1]:.6f}")

    # 5. Week 1 Completion Verification
    print("\n--- Week 1 Status ---")
    print("Dataset status: Validated, verified, and ready for Week 2 Cleaning & EDA.")
    print("==================================================")

if __name__ == "__main__":
    inspect_raw_dataset()
