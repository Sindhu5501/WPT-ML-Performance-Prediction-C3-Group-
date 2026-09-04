"""
TIH IIT Guwahati AI/ML Research Internship
Day 9 Implementation: Mentor-Approved Data Cleaning & Dataset V1 Generation
"""

import pandas as pd
import numpy as np

def execute_cleaning_plan():
    raw_file = "Problem Statement 3 -FOR SPIRAL COIL DATASET_ (1).csv"
    output_file = "wpt_cleaned_data_v1.csv"

    print("==================================================")
    print("      DAY 9: EXECUTING APPROVED CLEANING PLAN     ")
    print("==================================================")

    # 1. Ingestion
    df = pd.read_csv(raw_file)
    print(f"Loaded raw records: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Strip whitespaces from column headers
    df.columns = df.columns.str.strip()

    # 3. Rename columns to standardized snake_case format
    rename_mapping = {
        'dista [mm]': 'distance_mm',
        'L(Current1,Current1) [uH]': 'L1_uH',
        'L(Current3,Current1) [uH]': 'M13_uH',
        'L(Current3,Current3) [uH]': 'L3_uH',
        'CplCoef(Current1,Current3) []': 'coupling_coeff'
    }
    df = df.rename(columns=rename_mapping)

    # 4. Drop non-numeric metadata column
    if 'Data_Type' in df.columns:
        df = df.drop(columns=['Data_Type'])
        print("Dropped non-numeric metadata column: 'Data_Type'")

    # 5. Integrity Verification
    print("\n--- Cleaned Dataset Verification ---")
    print(f"Remaining Columns: {list(df.columns)}")
    print(f"Total Rows Retained: {len(df)} (100% data preservation)")
    print(f"Null Values Across Features: {df.isnull().sum().sum()}")
    print(f"Duplicate Rows Count: {df.duplicated().sum()}")

    # 6. Physical Sanity Assertions
    assert (df['distance_mm'] >= 0).all(), "Negative distance detected!"
    assert (df['coupling_coeff'] >= 0).all() and (df['coupling_coeff'] <= 1).all(), "Coupling coefficient out of bounds [0, 1]!"
    assert (df['L1_uH'] > 0).all() and (df['M13_uH'] > 0).all() and (df['L3_uH'] > 0).all(), "Non-positive inductance values found!"
    print("Physical Sanity Assertions: ALL PASSED")

    # 7. Save Dataset V1
    df.to_csv(output_file, index=False)
    print(f"\nSaved cleaned dataset: '{output_file}'")
    print("==================================================")

if __name__ == "__main__":
    execute_cleaning_plan()
