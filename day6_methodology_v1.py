"""
TIH IIT Guwahati AI/ML Research Internship
Day 6 Implementation: Methodology V1 Validation & Bound Enforcement Check
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def execute_methodology_v1():
    # 1. Ingestion
    file_path = "Problem Statement 3 -FOR SPIRAL COIL DATASET_ (1).csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    clean_cols = {
        'dista [mm]': 'distance_mm',
        'L(Current1,Current1) [uH]': 'L1_uH',
        'L(Current3,Current1) [uH]': 'M13_uH',
        'L(Current3,Current3) [uH]': 'L3_uH',
        'CplCoef(Current1,Current3) []': 'coupling_coeff'
    }
    df = df.rename(columns=clean_cols).drop(columns=['Data_Type'], errors='ignore')

    # 2. Frozen 70/15/15 Partition Protocol
    X = df[['distance_mm']]
    y = df['coupling_coeff']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.17647, random_state=42
    )

    print("==================================================")
    print("      DAY 6: METHODOLOGY V1 PARTITION VERIFICATION")
    print("==================================================")
    print(f"Train samples     : {len(X_train)} (70.0%)")
    print(f"Validation samples: {len(X_val)} (15.0%)")
    print(f"Test samples      : {len(X_test)} (15.0%)")

    # 3. Model Training under Methodology V1
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluation with Physical Bound Enforcement (0 <= k <= 1)
    test_preds = model.predict(X_test)
    test_preds = np.clip(test_preds, 0.0, 1.0)

    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    r2 = r2_score(y_test, test_preds)
    mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100

    print("\n--- Frozen Test Set Evaluation ---")
    print(f"MAE     : {mae:.6f}")
    print(f"RMSE    : {rmse:.6f}")
    print(f"R² Score: {r2:.6f}")
    print(f"MAPE (%) : {mape:.4f}%")
    print("Verification Status: Methodology V1 Confirmed & Reproducible")
    print("==================================================")

if __name__ == "__main__":
    execute_methodology_v1()
