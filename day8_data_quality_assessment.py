"""
TIH IIT Guwahati AI/ML Research Internship
Day 8 Implementation: Data Quality Assessment, IQR Outlier Evaluation & Cleaning Rules
"""

import pandas as pd
import numpy as np

def assess_data_quality():
    file_path = "Problem Statement 3 -FOR SPIRAL COIL DATASET_ (1).csv"
    df = pd.read_csv(file_path)

    print("================================================================")
    print("        DAY 8: COMPREHENSIVE DATA QUALITY AUDIT REPORT          ")
    print("================================================================\n")

    # 1. Check raw columns and white spaces
    print("--- 1. Column Sanitization Check ---")
    for col in df.columns:
        has_spaces = col.startswith(' ') or col.endswith(' ')
        print(f"• Column: '{col}' -> Whitespace detected: {has_spaces}")

    # 2. Missing & Duplicate Analysis
    print("\n--- 2. Missing Values & Duplicate Records ---")
    print(f"Total Rows: {len(df)}")
    print(f"Total Duplicate Rows: {df.duplicated().sum()}")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        pct = (null_count / len(df)) * 100
        print(f"• {col:<35}: {null_count} nulls ({pct:.2f}%)")

    # 3. Numeric Physical Bounds Validation
    print("\n--- 3. Boundary & Sign Violations ---")
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        negative_count = (df[col] < 0).sum()
        print(f"• {col:<35}: min={df[col].min():.4f}, max={df[col].max():.4f} | Negatives: {negative_count}")

    # 4. Statistical IQR Outlier Detection vs. Physical Reality
    print("\n--- 4. Statistical IQR Outlier Analysis ---")
    outlier_summary = []
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        count = outlier_mask.sum()
        pct = (count / len(df)) * 100

        outlier_summary.append({
            'Feature': col,
            'Q1 (25%)': round(q1, 3),
            'Q3 (75%)': round(q3, 3),
            'Lower Bound': round(lower_bound, 3),
            'Upper Bound': round(upper_bound, 3),
            'IQR Outliers': count,
            'Outlier (%)': round(pct, 2)
        })

    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df.to_string(index=False))

    # 5. Proposed Cleaning Rules Summary
    print("\n--- 5. Approved Cleaning Action Plan ---")
    print("Rule 1: Drop non-numeric metadata column 'Data_Type' (99.98% nulls).")
    print("Rule 2: Retain 100% of numeric rows (IQR outliers are genuine physical near-field values).")
    print("Rule 3: Standardize feature column labels to snake_case without brackets.")
    print("================================================================")

if __name__ == "__main__":
    assess_data_quality()
