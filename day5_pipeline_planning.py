import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def setup_pipeline_and_log():
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

    # 2. Strict 70% Train, 15% Validation, 15% Test Splitting
    X = df[['distance_mm']]
    y = df['coupling_coeff']

    # Step A: Split 15% for held-out Test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Step B: Split remaining 85% into Train (70% of total) and Val (15% of total)
    # (0.15 / 0.85 ≈ 0.17647)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.17647, random_state=42
    )

    print("==================================================")
    print("      DAY 5: PIPELINE EXECUTION & SPLIT CHECK     ")
    print("==================================================")
    print(f"Total Records     : {len(df)}")
    print(f"Train Partition   : {len(X_train)} samples ({len(X_train)/len(df):.1%})")
    print(f"Validation Set    : {len(X_val)} samples ({len(X_val)/len(df):.1%})")
    print(f"Test Partition    : {len(X_test)} samples ({len(X_test)/len(df):.1%})\n")

    # 3. Models to Register
    candidate_models = {
        'EXP-01_LinearRegression': LinearRegression(),
        'EXP-02_Ridge_Alpha1.0': Ridge(alpha=1.0),
        'EXP-03_RandomForest_N100': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    logs = []

    for exp_id, model in candidate_models.items():
        # Fit on Train only
        model.fit(X_train, y_train)

        # Evaluate on Validation Set
        val_preds = model.predict(X_val)
        val_r2 = r2_score(y_val, val_preds)
        val_mape = np.mean(np.abs((y_val - val_preds) / y_val)) * 100

        # Evaluate on Test Set
        test_preds = model.predict(X_test)
        test_r2 = r2_score(y_test, test_preds)
        test_mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

        logs.append({
            'Experiment_ID': exp_id,
            'Model': model.__class__.__name__,
            'Val_R2': round(val_r2, 6),
            'Val_MAPE(%)': round(val_mape, 4),
            'Test_R2': round(test_r2, 6),
            'Test_MAPE(%)': round(test_mape, 4),
            'Test_MAE': round(test_mae, 6),
            'Test_RMSE': round(test_rmse, 6)
        })

    # 4. Save and Display Log
    log_df = pd.DataFrame(logs)
    log_file = "experiment_log.csv"
    log_df.to_csv(log_file, index=False)
    print(f"Experiment log saved to: {log_file}\n")
    print(log_df.to_string(index=False))
    print("==================================================")

if __name__ == "__main__":
    setup_pipeline_and_log()
