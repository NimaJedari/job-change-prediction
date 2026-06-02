# Job Change Prediction — Data Scientists
### XGBoost classifier predicting whether a data science candidate will switch jobs

---

## Overview

This project builds a full end-to-end machine learning pipeline to predict whether a data science candidate is actively looking to switch jobs — based on their demographics, education, experience, and current employment situation.

The business application is direct: in a recruiting context, identifying high-intent candidates early allows teams to prioritise outreach, personalise messaging, and improve pipeline conversion rates.

**Final model performance:**
| Metric | Score |
|---|---|
| ROC-AUC | 0.807 |
| Switch Recall | 77.1% |
| Switch F1 | 0.63 |
| Overall Accuracy | 77.3% |

---

## Dataset

**HR Analytics: Job Change of Data Scientists**  
Source: [Kaggle](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)

- 19,158 candidate records
- 13 raw features covering demographics, education, experience, and employment
- Binary target: `1` = looking to switch jobs, `0` = not looking
- Class distribution: 75.1% Stay / 24.9% Switch
- Naturally occurring missing values across multiple columns — no artificial messiness injected

---

## Project Structure

```
hr-analytics-xgboost/
│
├── job_change_prediction.py   # Full pipeline script
├── Job_Change_Report.html     # Interactive analysis report
├── requirements.txt           # Dependencies
└── README.md
```

---

## Pipeline

```
Load Raw Data → Assess Missingness → Clean & Impute → Feature Engineering → SMOTE → GridSearchCV → Threshold Tuning → Evaluate
```

---

## Data Cleaning

The dataset contained real, naturally occurring data quality issues that required deliberate, context-aware decisions — not blanket imputation.

| Column | Issue | Strategy | Reasoning |
|---|---|---|---|
| `company_type` | 32.1% missing | Fill → `"Unknown"` | Missingness signals no current employer — a meaningful behavioral category |
| `company_size` | 31.0% missing | Ordinal encode + `0` for missing | `0` = no company, preserves the signal rather than guessing a size |
| `gender` | 23.5% missing | Fill → `"Unknown"` | Non-disclosure is its own category, not a value to impute |
| `major_discipline` | 14.7% missing | Fill → `"No Major"` | Structurally correct for non-degree holders |
| `experience` | String `">20"`, `"<1"` | Parse → numeric, then median for NaN | Real-world data entry format requiring string conversion before imputation |
| `last_new_job` | String `"never"`, `">4"` | Parse → numeric, then median for NaN | Same pattern as experience |
| `education_level` | 2.4% missing | Mode imputation → ordinal encode | Small enough for safe imputation; ordinal encoding preserves natural hierarchy |
| `training_hours` | Right-skewed | log1p transformation | Skewness reduced from 1.819 → −0.346 |
| `enrollee_id`, `city` | No predictive value | Dropped | ID column removed; city replaced by continuous `city_development_index` |

**Key principle:** For the three high-missingness columns, missing data was not random — it reflected real candidate characteristics. Treating missingness as a meaningful signal rather than a gap to fill produced the top features in the final model.

---

## Feature Engineering

Three features were engineered to capture non-linear relationships and interaction effects:

**`is_unemployed`** (binary flag)  
Set to `1` where `company_type == "Unknown"` AND `company_size == 0` simultaneously. Identifies candidates with no current employer. Became the single most important feature at **25.7% model importance**.

**`cdi_bin`** (ordinal bins)  
`city_development_index` cut into 4 bands: `low`, `mid`, `high`, `very_high`. CDI has a non-linear relationship with job switching — binning captures threshold effects that a raw continuous variable flattens. Together the three bin dummies account for ~13% of model importance.

**`exp_x_company_size`** (interaction term)  
`experience × company_size`. A senior engineer at a startup behaves very differently from a junior at a large corporation. The interaction term captures this directly.

**`education_level_ord`** (ordinal encoding)  
Replaced one-hot encoding with a 1–5 ordinal scale (Primary School → PhD). Preserves the natural hierarchy and reduces dimensionality from 5 binary columns to 1 numeric column.

---

## Class Imbalance — SMOTE

The dataset is imbalanced at 75/25. SMOTE (Synthetic Minority Oversampling Technique) was applied to generate synthetic Switch examples in the training set only.

```
Before SMOTE:  Stay 11,504  |  Switch 3,822   (train set)
After SMOTE:   Stay 11,504  |  Switch 11,504  (balanced)
```

> ⚠️ **Critical:** SMOTE was applied after the train/test split. Applying it before would contaminate the test set with synthetic data derived from training observations, producing artificially inflated metrics.

---

## Model — XGBoost + GridSearchCV

XGBoost was chosen for its ability to handle mixed feature types, built-in regularisation, and strong performance on tabular data at this scale.

**GridSearchCV** tested 80 hyperparameter combinations (16 candidates × 5-fold CV) scoring by ROC-AUC:

```python
param_grid = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 4],
    "learning_rate": [0.05, 0.1],
    "subsample":     [0.8, 1.0],
}
```

**Best parameters:**
```
n_estimators: 200 | max_depth: 4 | learning_rate: 0.1 | subsample: 0.8
```

`subsample: 0.8` was selected over 1.0 — using 80% of rows per tree introduces regularising randomness that prevents overfitting on SMOTE-generated synthetic patterns.

---

## Threshold Tuning

The default classification threshold of 0.50 was lowered to **0.39** by sweeping all values from 0.10–0.70 and selecting the threshold that maximised F1 for the Switch class.

This trades a small increase in false positives for a meaningful gain in Switch recall — the correct tradeoff in a recruiting context where missing a high-intent candidate is more costly than a false alarm.

---

## Results

### Model progression

| Version | ROC-AUC | Switch Recall | False Negatives |
|---|---|---|---|
| Baseline (scale_pos_weight) | 0.804 | 0.74 | 254 |
| + SMOTE | 0.804 | 0.74 | 254 |
| + Feature Engineering | **0.807** | **0.771** | **219** |

Feature engineering drove a **3.1 percentage point gain in recall**, catching 35 additional job switchers per 3,832 candidates evaluated. At scale across a 50,000-candidate pipeline, this represents ~460 additional correctly identified high-intent candidates.

### Classification report — best model (threshold = 0.39)

```
              precision    recall  f1-score   support
        Stay       0.91      0.77      0.84      2877
      Switch       0.53      0.77      0.63       955
    accuracy                           0.77      3832
```

### Top features

| Feature | Importance | Type |
|---|---|---|
| `is_unemployed` | 25.7% | Engineered |
| `city_development_index` | 5.9% | Raw |
| `company_size` | 5.2% | Cleaned |
| `cdi_bin_mid` | 5.1% | Engineered |
| `cdi_bin_very_high` | 5.0% | Engineered |
| `major_discipline_STEM` | 4.7% | Raw |
| `last_new_job` | 4.0% | Cleaned |
| `education_level_ord` | 2.7% | Engineered |

---

## Key Findings

- **Unemployment is the strongest predictor** — candidates with no current employer are the highest-intent job seekers in the pipeline
- **City development index is non-linear** — both very high CDI (major tech hubs) and low CDI candidates switch at elevated rates, for different reasons
- **STEM candidates are the most mobile** — broader market demand drives higher switching rates regardless of other factors
- **Recent job changers switch again** — demonstrated switching behavior is the most reliable proxy for active job-seeking intent
- **Treating missingness as signal outperformed imputation** — the three high-missingness columns became top model features only because their missing values were preserved as meaningful categories

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/hr-analytics-xgboost.git
cd hr-analytics-xgboost

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python job_change_prediction.py
```

The dataset is loaded directly from GitHub — no manual download required.

---

## Dependencies

```
pandas
numpy
xgboost
scikit-learn
imbalanced-learn
matplotlib
```

---

## Author

Nima Jedari  
[LinkedIn](https://www.linkedin.com/in/nima-jedari/) · [GitHub](https://github.com/NimaJedari)
