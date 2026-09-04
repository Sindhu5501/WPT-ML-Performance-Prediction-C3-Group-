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


---

## 📋 Day 4: Dataset Design, Schema & Quality Expectations

### 1. Target Schema & Specifications (`Problem Statement 3 - FOR SPIRAL COIL DATASET`)

| Feature / Column | Physical Quantity | Expected Range | Data Type | Null Count | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `dista [mm]` | Air Gap Distance ($z$-axis) | 0.00 – 200.00 mm | `float64` | 0 | Primary Predictor |
| `L(Current1,Current1) [uH]` | Primary Coil Self-Inductance ($L_1$) | 107.41 – 188.19 $\mu\text{H}$ | `float64` | 0 | Secondary Target |
| `L(Current3,Current1) [uH]` | Mutual Inductance ($M_{13}$) | 7.29 – 161.00 $\mu\text{H}$ | `float64` | 0 | Critical Target |
| `L(Current3,Current3) [uH]` | Secondary Coil Self-Inductance ($L_3$) | 105.33 – 183.98 $\mu\text{H}$ | `float64` | 0 | Secondary Target |
| `CplCoef(Current1,Current3) []` | Magnetic Coupling Coefficient ($k$) | 0.0678 – 0.8653 | `float64` | 0 | Primary Target |
| `Data_Type` | Simulation Metadata Tag | `Maxwell_Simulation` | `object` | 9,998 (drop) | Metadata Flag |

### 2. Dataset Risk Checklist & Mitigation Strategy

* **Spatial Autocorrelation Risk:** Distance is uniformly sampled at high frequency ($\Delta z \approx 0.02\text{ mm}$). Naive random splitting can lead to information leakage between adjacent points. 
  * *Mitigation:* Stratified block/range splitting or random holdout evaluation verified against interpolation boundaries.
* **Missing Value & Corruption Risk:** Simulation logs contain unpopulated string headers in `Data_Type` (9,998 NaNs).
  * *Mitigation:* Explicitly drop `Data_Type` before downstream model training.
* **Out-of-Distribution (OOD) Extrapolation:** Tree-based models (Random Forest) cannot extrapolate beyond observed min ($0\text{ mm}$) and max ($200\text{ mm}$) bounds.
  * *Mitigation:* Set operational inference bounds strictly to $[0, 200]\text{ mm}$ and explore physical/parametric baselines for boundary extensions.
 

---

## ⚙️ Day 5: Pipeline Architecture & Experiment Planning

### 1. End-to-End Pipeline Architecture Flowchart

```text
+-----------------------------------------------------------------------------------+
|                        RAW DATA INGESTION & QUALITY AUDIT                         |
|   (ANSYS Maxwell 10k Simulation CSV -> Column Strip -> Drop 'Data_Type' Flag)     |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                      LEAKAGE-SAFE SPLIT STRATEGY (70/15/15)                       |
|   Total: 10,000 samples | Train: 7,000 (70%) | Val: 1,500 (15%) | Test: 1,500 (15%)|
|   Preserved random state seed = 42                                                |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                       FEATURE SCALING & PREPROCESSING                             |
|   Robust / StandardScaler fit strictly on Train (70%), transformed on Val & Test  |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                       MODEL BENCHMARK SUITE (PLANNED)                             |
|   • Baseline: Linear OLS, Ridge Regularization                                    |
|   • Ensembles: Random Forest, XGBoost Regressor                                   |
|   • Neural Architectures: Multi-Layer Perceptron (MLP / PyTorch ANN)              |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                     EVALUATION SUITE & LOGGING FRAMEWORK                          |
|   MAE | MSE | RMSE | R² Score | MAPE (%) tracked across coupling (k) & inductances|
+-----------------------------------------------------------------------------------+
* **Multicollinearity:** $L_1$, $L_3$, and $M_{13}$ are physically coupled via $k = \frac{M}{\sqrt{L_1 L_3}}$.
  * *Mitigation:* Predict $k$ directly from geometry ($z$) or formulate multi-output regression with joint loss objectives.

### 3. Split Strategy Preservation
* **Partition:** 70% Train, 15% Validation, 15% Test.
* **Preservation Rule:** Preserve the continuous monotonic decay curve across training and validation splits while testing interpolation accuracy on unseen sub-intervals.



WPT-ML-Performance-Prediction/
├── README.md                           # Cumulative daily reports & documentation
├── Problem Statement 3 -...csv         # Raw untouched Maxwell simulation dataset
├── experiments/                        # Experiment execution scripts
│   ├── day2_experiment.py              # Baseline 80/20 benchmark
│   ├── day4_data_quality_audit.py      # Schema verification & bound checks
│   └── day5_pipeline_planning.py       # 70/15/15 split pipeline & automated logger
└── logs/
    └── experiment_log.csv              # Machine-readable experiment records





---

## 📑 Day 6: Pre-Dataset Readiness & Methodology V1 (Mentor Review 2)

### 1. Methodology V1 Specification
* **Objective:** Establish a mathematically consistent, leakage-safe pipeline to model non-linear electromagnetic parameters across continuous air gaps ($0 - 200\text{ mm}$).
* **Data Partitioning Protocol (Frozen):**
  * **Train Set:** 7,000 samples (70.0%)
  * **Validation Set:** 1,500 samples (15.0%)
  * **Test Set (Held-out):** 1,500 samples (15.0%)
  * **Partition Rule:** Seed fixed at `random_state=42`. Preprocessing parameters (min/max, mean/std) are calculated strictly on the training partition to eliminate target/distribution leakage.
* **Feature Engineering & Transformation:**
  * Primary predictor: Air-gap distance $z = \text{distance\_mm}$.
  * Physics-based transformation: Inverse-cube candidate features ($1/z^3$, $\ln(z + \epsilon)$) planned for linear regularized baselines to bridge the non-linear magnetic decay gap.
* **Model Candidates for Week 2/3 Expansion:**
  * Baselines: OLS Linear Regression, Ridge ($\alpha=1.0$), Lasso ($\alpha=0.01$).
  * Non-Parametric Ensembles: Random Forest Regressor ($n=100$, max_depth=None), XGBoost / LightGBM.
  * Deep Learning: Multi-Layer Perceptron (MLP) with 3 dense layers, ReLU activations, and Adam optimizer.
* **Evaluation Standards:** All models are ranked against 5 compulsory metrics: MAE, MSE, RMSE, $R^2$, and MAPE.

### 2. Mentor Review 2 Checkpoint & Action Items
* **Split Validation:** Confirmed that a 70/15/15 split maintains identical continuous density across all three partitions.
* **Multi-Target Formulation:** Approved training individual models for direct physical parameters ($k$, $M_{13}$) alongside multi-output baseline regressors.
* **Physical Constraints:** Predicted coupling coefficient values must strictly obey $0.0 \le k \le 1.0$; negative predictions will be clipped during post-processing.




---

## 📊 Day 7: Dataset Receipt & Initial Inspection Report (Week 1 Milestone)

### 1. Raw Dataset Audit & Ingestion Check
* **Archive Integrity:** Original source file `Problem Statement 3 -FOR SPIRAL COIL DATASET_ (1).csv` verified and locked as untouched raw baseline.
* **Volume:** 10,000 continuous simulation records, 6 features.
* **Format:** Comma-Separated Values (CSV), continuous floating-point measurements generated from 3D Finite Element Analysis (ANSYS Maxwell).

### 2. Comprehensive Statistical Summary

| Feature Column | Units | Missing Count | Min | 25% | Median (50%) | 75% | Max | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dista [mm]` | mm | 0 | 0.0000 | 50.0000 | 100.0000 | 150.0000 | 200.0000 | 57.7437 |
| `L(Current1,Current1) [uH]` | $\mu\text{H}$ | 0 | 107.4112 | 108.4942 | 109.3394 | 119.0869 | 188.1925 | 16.7479 |
| `L(Current3,Current1) [uH]` | $\mu\text{H}$ | 0 | 7.2921 | 13.1261 | 25.1705 | 54.6047 | 160.9991 | 36.3868 |
| `L(Current3,Current3) [uH]` | $\mu\text{H}$ | 0 | 105.3333 | 106.9806 | 107.5889 | 114.1761 | 183.9760 | 16.1027 |
| `CplCoef(Current1,Current3) []` | - | 0 | 0.0678 | 0.1220 | 0.2325 | 0.4683 | 0.8653 | 0.2295 |
| `Data_Type` | string | 9,998 | N/A | N/A | N/A | N/A | N/A | N/A |

### 3. First-Look Physical Findings
* **Monotonicity & Continuity:** Mutual Inductance ($M_{13}$) and Coupling Coefficient ($k$) decrease monotonically as distance increases from $0\text{ mm}$ to $200\text{ mm}$.
* **Asymptotic Tail:** Beyond $120\text{ mm}$, coupling drops below $0.15$ and flattens asymptotically towards $0.0678$, where inductive transfer becomes negligible.
* **Secondary Metadata:** The column `Data_Type` contains only 2 non-null simulation markers (`Maxwell_Simulation`) and will be discarded during Week 2 data cleaning without any loss of physical signal.

### 4. Week 1 Gate Sign-Off
* [x] Problem understanding, technical framing, and candidate methods documented.
* [x] Baseline linear, regularized, and ensemble models benchmarked.
* [x] 70/15/15 train-validation-test split protocol frozen.
* [x] Raw dataset inspected and validated for Week 2 preprocessing and EDA.



---

## 🔍 Day 8: Comprehensive Data Quality Assessment & Cleaning Rules (Intern Discussion 4)

### 1. Feature-by-Feature Data Quality Table

| Feature / Column | Expected Physical Range | Observed Range | Null / Missing (%) | Duplicates | IQR Statistical Outliers (%) | Physical Outlier Status | Quality Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dista [mm]` | $[0, 200]\text{ mm}$ | $0.00 - 200.00$ | $0\ (0.00\%)$ | 0 | $0\ (0.00\%)$ | None | **Valid & Clean** |
| `L(Current1,Current1) [uH]` | $> 0\ \mu\text{H}$ | $107.41 - 188.19$ | $0\ (0.00\%)$ | 0 | $1,306\ (13.06\%)$ | False Outliers (Near-field coupling) | **Physically Valid** |
| `L(Current3,Current1) [uH]` | $> 0\ \mu\text{H}$ | $7.29 - 161.00$ | $0\ (0.00\%)$ | 0 | $613\ (6.13\%)$ | False Outliers (Proximity effect) | **Physically Valid** |
| `L(Current3,Current3) [uH]` | $> 0\ \mu\text{H}$ | $105.33 - 183.98$ | $0\ (0.00\%)$ | 0 | $1,562\ (15.62\%)$ | False Outliers (Near-field coupling) | **Physically Valid** |
| `CplCoef(Current1,Current3) []` | $[0, 1]$ | $0.0678 - 0.8653$ | $0\ (0.00\%)$ | 0 | $0\ (0.00\%)$ | None | **Valid & Clean** |
| `Data_Type` | String metadata | `Maxwell_Simulation` | $9,998\ (99.98\%)$ | N/A | N/A | Metadata artifact | **Drop Column** |

### 2. Physical Justification of Apparent Statistical Outliers
* **The "IQR Outlier" Fallacy:** Standard boxplot IQR analysis flags values above $134.98\ \mu\text{H}$ for $L_1$ and $116.82\ \mu\text{H}$ for $M_{13}$ as statistical outliers.
* **Domain Truth:** These higher values occur strictly when distance $z < 25\text{ mm}$. At extremely close proximity, mutual magnetic induction and magnetic flux concentration spike sharply due to magnetic core/ferrite proximity effects. 
* **Cleaning Rule:** **Do not trim or clip these values.** Deleting these data points would discard critical short-distance charging behavior and corrupt continuous physical laws.

### 3. Proposed Cleaning Rules & Ingestion Protocol
1. **Rule 1 (Whitespace Elimination):** Strip all leading and trailing whitespaces from raw column labels.
2. **Rule 2 (Column Renaming):** Map verbose simulation tags to clean standard identifiers (`distance_mm`, `L1_uH`, `M13_uH`, `L3_uH`, `coupling_coeff`).
3. **Rule 3 (Metadata Pruning):** Drop `Data_Type` completely, preserving 100% of physical data without information loss.
4. **Rule 4 (Physical Bounds Assertion):** Assert strictly that $0 \le k \le 1$, $\text{distance} \ge 0$, and all inductances $> 0$.\



---

## 🧹 Day 9: Cleaning Proposal & Mentor Review 3 Checkpoint

### 1. Approved Cleaning Plan Specification
Following Mentor Review 3, the data treatment protocol is finalized as follows:
* **Pruning Metadata:** Column `Data_Type` is dropped due to $99.98\%$ missingness, removing non-informative simulation labels without altering numerical features.
* **Row Preservation Policy:** All $10,000$ simulation rows are retained. Apparent IQR statistical outliers ($13.06\%$ in $L_1$, $6.13\%$ in $M_{13}$, $15.62\%$ in $L_3$) are verified as genuine non-linear magnetic flux spikes occurring at close coil proximities ($z < 25\text{ mm}$).
* **Column Normalization:** Whitespace stripped and variables mapped to standard identifiers:
  * `dista [mm]` $\rightarrow$ `distance_mm`
  * `L(Current1,Current1) [uH]` $\rightarrow$ `L1_uH`
  * `L(Current3,Current1) [uH]` $\rightarrow$ `M13_uH`
  * `L(Current3,Current3) [uH]` $\rightarrow$ `L3_uH`
  * `CplCoef(Current1,Current3) []` $\rightarrow$ `coupling_coeff`

### 2. Mentor Decisions & Sign-Off
* **Cleaning Approval:** Retaining all continuous spatial steps confirmed; traditional outlier trimming rejected on domain-physics grounds.
* **Split Integrity Check:** Leakage-safe 70/15/15 partition with seed 42 reaffirmed.
* **Output Artifact:** The cleaned dataset is exported as `wpt_cleaned_data_v1.csv` for downstream EDA and feature engineering.



---

## 📈 Day 10: Exploratory Data Analysis & Non-Linearity Profiling (Intern Discussion 5)

### 1. Correlation Matrix & Statistical Properties

| Feature Pair | Pearson Correlation ($r$) | Spearman Rank ($\rho$) | Relationship Profile |
| :--- | :---: | :---: | :--- |
| `distance_mm` vs. `coupling_coeff` | **-0.948** | **-1.000** | Strictly monotonic inverse non-linear decay |
| `distance_mm` vs. `M13_uH` | **-0.885** | **-1.000** | Strict monotonic decay with steep near-field drop |
| `distance_mm` vs. `L1_uH` | **-0.749** | **-0.961** | Core proximity effect saturation beyond 35 mm |
| `distance_mm` vs. `L3_uH` | **-0.698** | **-0.690** | Highly non-linear plateau |
| `M13_uH` vs. `coupling_coeff` | **+0.980** | **+1.000** | Physical coupling law: $k \propto M$ |

### 2. Physical & Mathematical Interpretation of Visualizations
* **The Monotonicity Contrast ($r = -0.948$ vs. $\rho = -1.000$):** While the linear Pearson correlation is $-0.948$, the Spearman rank correlation is a perfect **$-1.000$**. This confirms that electromagnetic coupling is strictly monotonic, but non-linear (decaying exponentially / following inverse power laws).
* **Near-Field Proximity Spike ($0 - 25\text{ mm}$):** Mutual inductance drops by over **53%** (from $161.0\ \mu\text{H}$ down to $75.6\ \mu\text{H}$) within the first $25\text{ mm}$ alone. This explains why standard linear regression yielded a $32.2\%$ MAPE, whereas tree-based ensembles handle it smoothly.
* **Far-Field Asymptotic Flattening ($> 120\text{ mm}$):** Above $120\text{ mm}$, coupling drops below $0.15$ and flattens out, showing positive skewness ($+0.85$) where high-density samples reside in the low-coupling regime.
* **Data Leakage & Sparsity Check:** Distance values are uniformly distributed across the domain ($0 - 200\text{ mm}$, Skewness $\approx 0.00$), ensuring no unrepresented gap regions exist in the simulation grid.
