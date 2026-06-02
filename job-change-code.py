import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix, f1_score)
from imblearn.over_sampling import SMOTE

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════
url = "https://raw.githubusercontent.com/anhquan0412/dataset/main/aug_train.csv"
df = pd.read_csv(url)

df.drop(columns=["enrollee_id", "city"], inplace=True)
df["gender"] = df["gender"].fillna("Unknown")
df["enrolled_university"] = df["enrolled_university"].fillna(df["enrolled_university"].mode()[0])
df["major_discipline"] = df["major_discipline"].fillna("No Major")
df["experience"] = df["experience"].replace({">20": "21", "<1": "0"})
df["experience"] = pd.to_numeric(df["experience"], errors="coerce")
df["experience"] = df["experience"].fillna(df["experience"].median())
size_map = {
    "<10": 1, "10/49": 2, "50-99": 3, "100-500": 4,
    "500-999": 5, "1000-4999": 6, "5000-9999": 7, "10000+": 8
}
df["company_size"] = df["company_size"].map(size_map).fillna(0)
df["company_type"] = df["company_type"].fillna("Unknown")
df["last_new_job"] = df["last_new_job"].replace({"never": "0", ">4": "5"})
df["last_new_job"] = pd.to_numeric(df["last_new_job"], errors="coerce")
df["last_new_job"] = df["last_new_job"].fillna(df["last_new_job"].median())
df["training_hours_log"] = np.log1p(df["training_hours"])
df.drop(columns=["training_hours"], inplace=True)

# ── education_level: ordinal encode before dropping ───────────────────────
edu_order = {
    "Primary School": 1, "High School": 2,
    "Graduate": 3, "Masters": 4, "Phd": 5
}
df["education_level"] = df["education_level"].fillna(df["education_level"].mode()[0])
df["education_level_ord"] = df["education_level"].map(edu_order)
df.drop(columns=["education_level"], inplace=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
df["cdi_bin"] = pd.cut(
    df["city_development_index"],
    bins=[0, 0.624, 0.789, 0.920, 1.0],
    labels=["low", "mid", "high", "very_high"]
)
df["exp_x_company_size"] = df["experience"] * df["company_size"]
df["is_unemployed"] = ((df["company_type"] == "Unknown") & (df["company_size"] == 0)).astype(int)

print(f"── Clean shape: {df.shape}")
print(f"── Missing after cleaning: {df.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
df["target"] = df["target"].astype(int)
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["target"])
y = df["target"]

print(f"── Features after encoding: {X.shape[1]}")
print(f"── Target: {y.value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"── After SMOTE: {X_train_sm.shape[0]:,} train rows ✅")

# ══════════════════════════════════════════════════════════════════════════════
# 4. GRIDSEARCHCV — original complexity, engineered features
# ══════════════════════════════════════════════════════════════════════════════
param_grid = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 4],
    "learning_rate": [0.05, 0.1],
    "subsample":     [0.8, 1.0],
}

grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid, scoring="roc_auc", cv=5, n_jobs=1, verbose=1
)
print("\n── Running GridSearchCV (80 fits)...")
grid.fit(X_train_sm, y_train_sm)
print(f"\n── Best params: {grid.best_params_}")
print(f"── Best CV ROC-AUC: {grid.best_score_:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
best_model = grid.best_estimator_
y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# Threshold tuning
results = []
for t in np.arange(0.1, 0.7, 0.01):
    y_pred_t = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t, pos_label=1, zero_division=0)
    recall = (y_pred_t[y_test == 1] == 1).mean()
    results.append({"threshold": round(t, 2), "f1": f1, "recall": recall})

results_df = pd.DataFrame(results)
best_t = results_df.loc[results_df["f1"].idxmax(), "threshold"]
y_pred_tuned = (y_prob >= best_t).astype(int)

print(f"\n── Results (threshold = {best_t}) ────────────────────────────────")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred_tuned, target_names=["Stay", "Switch"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))

# ══════════════════════════════════════════════════════════════════════════════
# 6. COMPARISON VS BASELINE (no feature engineering)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── COMPARISON vs BASELINE (no feature engineering) ───────────────")
print(f"{'Metric':<25} {'Baseline':>12} {'Engineered':>12} {'Delta':>10}")
print("-" * 62)
baseline = {"ROC-AUC": 0.8042, "Switch Recall": 0.74, "Switch F1": 0.62, "Accuracy": 0.7766}
current = {
    "ROC-AUC":       round(roc_auc_score(y_test, y_prob), 4),
    "Switch Recall": round((y_pred_tuned[y_test==1]==1).mean(), 4),
    "Switch F1":     round(f1_score(y_test, y_pred_tuned, pos_label=1), 4),
    "Accuracy":      round(accuracy_score(y_test, y_pred_tuned), 4),
}
for metric in baseline:
    delta = current[metric] - baseline[metric]
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    print(f"{metric:<25} {baseline[metric]:>12} {current[metric]:>12} {arrow} {abs(delta):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
importances = pd.Series(best_model.feature_importances_, index=X.columns)
print("\n── Top 15 Features:")
print(importances.sort_values(ascending=False).head(15).round(4))