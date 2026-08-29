# Machine Learning-Based Performance Prediction of Wireless Power Transfer (WPT) Systems

**Internship Program:** TIH IIT Guwahati AI/ML Research Internship (4 Weeks)  
**Research Domain:** Artificial Intelligence / Machine Learning / Power Electronics / Data Analytics  
**Supervisor / Lab:** Dr. R. Narayanamoorthi (SRMIST / Wireless Charging Research Centre)  
**Dataset:** Spiral Coil Electromagnetic Simulation Data (ANSYS Maxwell)

---

## 📌 Day 1: Problem Formulation, Requirements & Dataset Exploration

### 1. Problem Statement & Research Objectives
- **Context:** Wireless Power Transfer (WPT) efficiency and mutual inductance are heavily dependent on geometric coil separation (air gap distance) and alignment. 
- **Objective:** Surrogating computationally expensive 3D Finite Element Analysis (FEA) electromagnetic simulations with fast, accurate Machine Learning regression models to predict:
  1. Primary Self-Inductance ($L_1$)
  2. Secondary Self-Inductance ($L_2$)
  3. Mutual Inductance ($M_{13}$)
  4. Magnetic Coupling Coefficient ($k$)
- **Scope:** Software-only ML modeling, experimental baseline benchmarking, and error evaluation.

### 2. Dataset Analysis (`Problem Statement 3 - FOR SPIRAL COIL DATASET`)
- **Total Sample Count:** 10,000 continuous simulation points.
- **Input Feature:**
  - `dista [mm]`: Continuous air gap separation ranging from **0.00 mm to 200.00 mm** (step size $\approx 0.02$ mm).
- **Target Parameters:**
  - `L(Current1,Current1) [uH]` ($L_1$): Primary coil inductance (Range: $107.41\ \mu\text{H} - 188.19\ \mu\text{H}$, Mean: $117.86\ \mu\text{H}$).
  - `L(Current3,Current1) [uH]` ($M_{13}$): Mutual inductance (Range: $7.29\ \mu\text{H} - 161.00\ \mu\text{H}$, Mean: $39.98\ \mu\text{H}$).
  - `L(Current3,Current3) [uH]` ($L_2$): Secondary coil inductance (Range: $105.33\ \mu\text{H} - 183.98\ \mu\text{H}$, Mean: $115.16\ \mu\text{H}$).
  - `CplCoef(Current1,Current3) []` ($k$): Magnetic coupling coefficient (Range: $0.0678 - 0.8653$, Mean: $0.3145$).

### 3. Pipeline Architecture
1. **Data Preprocessing:** Column standardization, validation of physical bounds, and an 80/20 train-test split.
2. **Model Training:** Comparative analysis across Linear Regression, Ridge Regularization, and Random Forest Regression.
3. **Evaluation Framework:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Coefficient of Determination ($R^2$), and Mean Absolute Percentage Error (MAPE).

---

## 🚀 Day 2: Baseline Implementation, Initial Experiments & Evaluation

### 1. Experimental Methodology
- Split: **8,000 train samples** (80%) and **2,000 test samples** (20%) using `random_state=42`.
- Tested baseline linear and non-linear regression models predicting coupling coefficient ($k$) and mutual inductance from air gap distance.

### 2. Benchmark Results

| Model | MAE | RMSE | $R^2$ Score | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.062539 | 0.072790 | 0.901681 | 32.20% |
| **Ridge Regression ($\alpha=1.0$)** | 0.062539 | 0.072790 | 0.901681 | 32.20% |
| **Random Forest Regressor ($n=100$)** | **0.000045** | **0.000062** | **1.000000** | **0.0143%** |

### 3. Technical Findings & Discussion
- **Non-Linear Field Decay:** Linear and Ridge models struggle with an error of $\sim 32.2\%$ MAPE because electromagnetic coupling decays exponentially/inversely with distance rather than in a straight line.
- **Ensemble Precision:** Random Forest captures the continuous non-linear magnetic flux drop across the entire 0–200 mm range with $R^2 = 1.0000$ and MAPE $< 0.02\%$.
- **Next Steps:** Implement Multi-layer Perceptron (ANN), XGBoost, multi-output target prediction, and hyperparameter tuning.
