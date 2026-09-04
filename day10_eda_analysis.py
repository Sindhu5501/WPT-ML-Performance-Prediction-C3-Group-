"""
TIH IIT Guwahati AI/ML Research Internship
Day 10 Implementation: Exploratory Data Analysis, Non-Linear Dynamics & Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda():
    file_path = "Problem Statement 3 -FOR SPIRAL COIL DATASET_ (1).csv"
    df = pd.read_csv(file_path)

    # 1. Clean column labels
    df.columns = df.columns.str.strip()
    rename_mapping = {
        'dista [mm]': 'distance_mm',
        'L(Current1,Current1) [uH]': 'L1_uH',
        'L(Current3,Current1) [uH]': 'M13_uH',
        'L(Current3,Current3) [uH]': 'L3_uH',
        'CplCoef(Current1,Current3) []': 'coupling_coeff'
    }
    df = df.rename(columns=rename_mapping).drop(columns=['Data_Type'], errors='ignore')

    print("================================================================")
    print("           DAY 10: EXPLORATORY DATA ANALYSIS REPORT             ")
    print("================================================================\n")

    # 2. Pearson vs Spearman Rank Correlation
    pearson_corr = df.corr(method='pearson')
    spearman_corr = df.corr(method='spearman')

    print("--- 1. Pearson Correlation Matrix (Linear Dependency) ---")
    print(pearson_corr.round(4).to_string())

    print("\n--- 2. Spearman Rank Correlation (Monotonic Dependency) ---")
    print(spearman_corr.round(4).to_string())

    # 3. Distribution Moments (Skewness & Kurtosis)
    print("\n--- 3. Distribution Moments ---")
    moments_df = pd.DataFrame({
        'Skewness': df.skew(),
        'Kurtosis': df.kurtosis()
    })
    print(moments_df.round(4).to_string())

    # 4. Generate and Save Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # A: Distance vs Coupling
    axes[0, 0].plot(df['distance_mm'], df['coupling_coeff'], color='crimson', lw=2)
    axes[0, 0].set_title("Coupling Coefficient (k) vs Distance (mm)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Air Gap Distance (mm)")
    axes[0, 0].set_ylabel("Coupling Coefficient (k)")
    axes[0, 0].grid(True, alpha=0.3)

    # B: Distance vs Mutual Inductance
    axes[0, 1].plot(df['distance_mm'], df['M13_uH'], color='navy', lw=2)
    axes[0, 1].set_title("Mutual Inductance (M13) vs Distance (mm)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Air Gap Distance (mm)")
    axes[0, 1].set_ylabel("Mutual Inductance (uH)")
    axes[0, 1].grid(True, alpha=0.3)

    # C: Pearson Heatmap
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".3f", ax=axes[1, 0], cbar=True)
    axes[1, 0].set_title("Pearson Correlation Heatmap", fontsize=11, fontweight='bold')

    # D: Coupling Distribution
    sns.histplot(df['coupling_coeff'], kde=True, color='teal', ax=axes[1, 1], bins=40)
    axes[1, 1].set_title("Distribution of Coupling Coefficient (k)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Coupling Coefficient (k)")

    plt.tight_layout()
    plot_filename = "eda_analysis_day10.png"
    plt.savefig(plot_filename, dpi=150)
    print(f"\n[+] EDA visualization plots generated and saved as: '{plot_filename}'")
    print("================================================================")

if __name__ == "__main__":
    perform_eda()
