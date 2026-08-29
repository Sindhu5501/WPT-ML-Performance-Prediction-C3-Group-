"""
TIH IIT Guwahati AI/ML Research Internship
Day 2 Implementation: Baseline Regression Models for WPT Performance Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_experiment():
    file_path = "Problem Statement 3 -FOR SPIRAL COIL DATASET_ (1).csv"
    print(f"Loading dataset: {file_path}")
    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()
    clean_columns = {
        'dista [mm]': 'distance_mm',
        'L(Current1,Current1) [uH]': 'L1_uH',
        'L(Current3,Current1) [uH]': 'M13_uH',
        'L(Current3,Current3) [uH]': 'L3_uH',
        'CplCoef(Current1,Current3) []': 'coupling_coeff'
    }
    df = df.rename(columns=clean_columns)

    X = df[['distance_mm']]
    y = df['coupling_coeff']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression (alpha=1.0)': Ridge(alpha=1.0),
        'Random Forest Regressor (n=100)': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        results.append({
            'Model': name,
            'MAE': round(mae, 6),
            'RMSE': round(rmse, 6),
            'R2 Score': round(r2, 6),
            'MAPE (%)': round(mape, 4)
        })

    results_df = pd.DataFrame(results)
    print("\n=== DAY 2 EVALUATION SUMMARY ===")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    run_experiment()
