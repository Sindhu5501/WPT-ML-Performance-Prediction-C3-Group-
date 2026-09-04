"""
TIH IIT Guwahati AI/ML Research Internship
Day 3 Check: Multi-Target Baseline Regression (L1, L2, M13, Coupling Coefficient)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Dataset
df = pd.read_csv("Problem Statement 3 -FOR SPIRAL COIL DATASET_ (1).csv")
df.columns = df.columns.str.strip()

clean_cols = {
    'dista [mm]': 'distance_mm',
    'L(Current1,Current1) [uH]': 'L1_uH',
    'L(Current3,Current1) [uH]': 'M13_uH',
    'L(Current3,Current3) [uH]': 'L3_uH',
    'CplCoef(Current1,Current3) []': 'coupling_coeff'
}
df = df.rename(columns=clean_cols)

# 2. Features and Targets
X = df[['distance_mm']]
targets = ['coupling_coeff', 'M13_uH', 'L1_uH', 'L3_uH']
Y = df[targets]

# 3. 80/20 Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Multi-target Random Forest
rf_multi = RandomForestRegressor(n_estimators=100, random_state=42)
rf_multi.fit(X_train, Y_train)
preds = rf_multi.predict(X_test)
preds_df = pd.DataFrame(preds, columns=targets, index=Y_test.index)

# 5. Evaluate each target
summary = []
for target in targets:
    actual = Y_test[target]
    pred = preds_df[target]
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    summary.append({
        'Target': target,
        'MAE': round(mae, 6),
        'RMSE': round(rmse, 6),
        'R2 Score': round(r2, 6),
        'MAPE (%)': round(mape, 4)
    })

print("=== DAY 3 MULTI-TARGET BENCHMARK ===")
print(pd.DataFrame(summary).to_string(index=False))
