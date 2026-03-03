---
name: tabpfn-classify
description: Run a TabPFN classification baseline, generate the first submission, rapidly probe features, then optimize with GBT ensembles, threshold tuning, and calibration. Use after tabpfn-explore has prepared the data and CV folds.
---

> **Core context:** See [tabpfn-core](../tabpfn-core/SKILL.md) for behavior rules, workflow principles, and project conventions.
> **API setup:** See [api-setup.md](../tabpfn-core/references/api-setup.md) — always check budget before any API call.
> **Data prep:** See [data-requirements.md](../tabpfn-core/references/data-requirements.md) for preprocessing rules.

**Prerequisites:** Clean `X_train`, `y_train`, `X_test` DataFrames; saved `folds`; API budget confirmed; task type and evaluation metric identified.

---

## What TabPFN v2.5 Is

A transformer-based **tabular foundation model** using in-context learning. Makes predictions in a single forward pass — no training loop, no hyperparameters to tune. Meta-trained on billions of synthetic tabular datasets.

- Outperforms AutoGluon 1.4 (a 4-hour tuned ensemble) in seconds
- Natively handles missing values, categoricals, outliers, mixed types
- Well-calibrated probability outputs (good for log-loss, AUC)
- Scikit-learn API: `.fit()`, `.predict()`, `.predict_proba()`

---

## 1. Classification Baseline

```python
from tabpfn_client import TabPFNClassifier
import numpy as np

clf = TabPFNClassifier()

oof_preds = np.zeros((len(X_train), n_classes))
test_preds = np.zeros((len(X_test), n_classes))
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    X_tr = X_train.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]

    clf.fit(X_tr, y_tr)

    oof_preds[val_idx] = clf.predict_proba(X_val)
    test_preds += clf.predict_proba(X_test) / len(folds)

    fold_score = compute_metric(y_train.iloc[val_idx], oof_preds[val_idx])
    fold_scores.append(fold_score)
    print(f"  Fold {fold_idx}: {fold_score:.6f}")

cv_mean = np.mean(fold_scores)
cv_std = np.std(fold_scores)
print(f"\nTabPFN Baseline — CV: {cv_mean:.6f} ± {cv_std:.6f}")

# Save for later ensembling — non-negotiable
np.save("oof/tabpfn_baseline_oof.npy", oof_preds)
np.save("oof/tabpfn_baseline_test.npy", test_preds)
```

---

## 2. Generate & Submit

```python
import pandas as pd

submission = pd.read_csv("data/raw/sample_submission.csv")
# submission["target"] = test_preds[:, 1]          # probability submission
# submission["target"] = np.argmax(test_preds, 1)  # label submission
submission.to_csv(f"submissions/tabpfn_baseline_cv{cv_mean:.4f}.csv", index=False)

# Sanity checks
assert submission.shape == pd.read_csv("data/raw/sample_submission.csv").shape
assert submission.isnull().sum().sum() == 0
print("✅ Submission file ready")
```

**This is your anchor point.** Log both CV and LB scores. Their relationship tells you how trustworthy your CV is:
- CV ≈ LB → trust your CV for all future decisions
- CV ≫ LB or CV ≪ LB → investigate before continuing

---

## 3. Log the Experiment

```python
log_experiment(
    exp_id="baseline_001",
    model_type="TabPFN_v2.5",
    feature_set="raw_minimal_preprocessing",
    n_features=X_train.shape[1],
    params="default",
    cv_score=cv_mean,
    cv_std=cv_std,
    lb_score="<fill after submission>",
    notes="First TabPFN classification baseline."
)
```

---

## 4. Rapid Feature Exploration

TabPFN needs no tuning, making it the fastest hypothesis tester in the toolkit. Use 1-fold tests to conserve API credits.

### Feature Ablation
```python
baseline_cv = cv_mean
train_idx, val_idx = folds[0]

for col in X_train.columns:
    X_drop = X_train.drop(columns=[col])
    clf.fit(X_drop.iloc[train_idx], y_train.iloc[train_idx])
    score = compute_metric(
        y_train.iloc[val_idx],
        clf.predict_proba(X_drop.iloc[val_idx])
    )
    delta = score - baseline_cv
    marker = "DROP" if delta > 0.001 else ""
    print(f"  Drop {col:30s} → {score:.6f} (Δ {delta:+.6f}) {marker}")
```

### Feature Engineering Probing
```python
train_idx, val_idx = folds[0]

# Test a candidate feature with 1-fold quick test
X_train_exp = X_train.copy()
X_test_exp = X_test.copy()
X_train_exp["new_feat"] = X_train["a"] / (X_train["b"] + 1e-8)
X_test_exp["new_feat"] = X_test["a"] / (X_test["b"] + 1e-8)

clf.fit(X_train_exp.iloc[train_idx], y_train.iloc[train_idx])
new_score = compute_metric(
    y_train.iloc[val_idx],
    clf.predict_proba(X_train_exp.iloc[val_idx])
)
print(f"New feature impact: {new_score - baseline_cv:+.6f}")
```

---

## 5. Feature Engineering (Optimization Phase)

### High-Value Patterns

**Aggregation features:**
```python
for cat_col in categorical_cols:
    for num_col in numeric_cols:
        for agg in ["mean", "std", "min", "max", "median"]:
            feat_name = f"{cat_col}_{num_col}_{agg}"
            mapping = train.groupby(cat_col)[num_col].agg(agg)
            X_train[feat_name] = X_train[cat_col].map(mapping)
            X_test[feat_name] = X_test[cat_col].map(mapping)
```

**Target encoding (OOF to prevent leakage):**
```python
def target_encode_oof(X_train, y_train, X_test, col, folds):
    X_train[f"{col}_target_enc"] = np.nan
    for train_idx, val_idx in folds:
        mapping = y_train.iloc[train_idx].groupby(X_train[col].iloc[train_idx]).mean()
        X_train.loc[X_train.index[val_idx], f"{col}_target_enc"] = X_train[col].iloc[val_idx].map(mapping)
    global_mapping = y_train.groupby(X_train[col]).mean()
    X_test[f"{col}_target_enc"] = X_test[col].map(global_mapping)
```

**Interaction features:**
```python
X_train["a_div_b"] = X_train["a"] / (X_train["b"] + 1e-8)
X_train["a_minus_b"] = X_train["a"] - X_train["b"]
X_train["a_times_b"] = X_train["a"] * X_train["b"]
```

**Frequency encoding:**
```python
freq = X_train[col].value_counts(normalize=True)
X_train[f"{col}_freq"] = X_train[col].map(freq)
X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0)
```

### Feature Selection
- TabPFN ablation: drop features one-by-one, measure CV impact
- LightGBM importance: rank by `feature_importances_`, drop bottom 20%
- Correlation dedup: for pairs >0.95 correlation, keep one

---

## 6. GBT Models for Ensemble Diversity

TabPFN + GBTs have fundamentally different inductive biases — ideal for ensembling.

```python
import lightgbm as lgb
import optuna

def lgb_objective(trial):
    params = {
        "objective": "binary",    # or "multiclass" with num_class
        "metric": "auc",          # adapt to competition metric
        "verbosity": -1,
        "n_estimators": 10000,
        "learning_rate": trial.suggest_float("lr", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
    scores = []
    for train_idx, val_idx in folds:
        dtrain = lgb.Dataset(X_train.iloc[train_idx], y_train.iloc[train_idx])
        dval = lgb.Dataset(X_train.iloc[val_idx], y_train.iloc[val_idx])
        model = lgb.train(params, dtrain, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        preds = model.predict(X_train.iloc[val_idx])
        scores.append(compute_metric(y_train.iloc[val_idx], preds))
    return np.mean(scores)

# Don't run without user confirmation for large n_trials
study = optuna.create_study(direction="maximize")
study.optimize(lgb_objective, n_trials=100)
```

**Always generate OOF predictions for every GBT model:**
```python
def train_lgb_with_oof(params, X_train, y_train, X_test, folds):
    oof = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    for train_idx, val_idx in folds:
        dtrain = lgb.Dataset(X_train.iloc[train_idx], y_train.iloc[train_idx])
        dval = lgb.Dataset(X_train.iloc[val_idx], y_train.iloc[val_idx])
        model = lgb.train(params, dtrain, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        oof[val_idx] = model.predict(X_train.iloc[val_idx])
        test_preds += model.predict(X_test) / len(folds)
    return oof, test_preds
```

---

## 7. TabPFN-Specific Optimization

### Multiple Subsample Ensembles (for large datasets)
```python
n_bags = 5
bag_test_preds = []
for i in range(n_bags):
    X_sub, y_sub = subsample_for_tabpfn(X_train, y_train, max_samples=10000, random_state=SEED + i)
    clf = TabPFNClassifier()
    clf.fit(X_sub, y_sub)
    bag_test_preds.append(clf.predict_proba(X_test))
tabpfn_bagged = np.mean(bag_test_preds, axis=0)
```

### Feature Subset Diversity
```python
import random
subsets = []
for i in range(3):
    random.seed(SEED + i)
    n_select = int(len(all_features) * 0.7)
    subsets.append(random.sample(all_features, n_select))
# Train TabPFN on each subset → 3 diverse prediction sets for blending
```

---

## 8. Ensembling

```python
from scipy.optimize import minimize

oof_list = [oof_tabpfn, oof_lgbm, oof_xgb, oof_catboost]
test_list = [test_tabpfn, test_lgbm, test_xgb, test_catboost]

def blend_objective(weights):
    w = np.array(weights)
    w = w / w.sum()
    blended = sum(wi * pi for wi, pi in zip(w, oof_list))
    return -compute_metric(y_train, blended)

n = len(oof_list)
result = minimize(blend_objective, x0=[1/n]*n, method="Nelder-Mead")
weights = np.array(result.x); weights = weights / weights.sum()
print(f"Weights: {dict(zip(['TabPFN','LGBM','XGB','CatBoost'], weights.round(3)))}")
print(f"Blend CV: {-result.fun:.6f}")
final_test = sum(w * p for w, p in zip(weights, test_list))
```

**Rules:**
- NEVER blend models validated on different CV splits
- Diversity > individual strength — a weaker model that disagrees adds value
- TabPFN + GBTs is a high-diversity combo (different architectures)

---

## 9. Post-Processing

### Threshold Optimization
```python
# For F1, macro-F1, or threshold-sensitive metrics
thresholds = np.arange(0.1, 0.9, 0.01)
best_thresh, best_score = 0.5, -np.inf
for t in thresholds:
    preds = (oof_proba[:, 1] > t).astype(int)
    score = compute_metric(y_train, preds)
    if score > best_score:
        best_thresh, best_score = t, score
print(f"Optimal threshold: {best_thresh:.2f} → {best_score:.4f}")
```

### Calibration
TabPFN outputs are generally well-calibrated. GBT probabilities may not be. If the metric rewards calibration (log-loss, Brier score), calibrate individual GBT models before ensembling using `CalibratedClassifierCV`.

---

## 10. Common Mistakes to Avoid

- Over-preprocessing for TabPFN (don't impute, scale, encode)
- Only TabPFN without GBT diversity — or only GBTs without TabPFN
- Exceeding API cell limits without checking first
- Burning credits on full-CV experiments during early exploration
- Different CV splits across models (breaks stacking)
- Target leakage in feature engineering
- Submitting without sanity checks (shape, NaNs, range, format)

---

## 11. Final Submission Checklist

1. ✅ CV score logged for every model and ensemble variant
2. ✅ CV-LB relationship consistent
3. ✅ OOF predictions saved for all models
4. ✅ Submission: correct shape, no NaNs, valid ranges, correct columns
5. ✅ Two final picks: one "safe" (best CV), one "risky" (aggressive ensemble)
6. ✅ All experiments in `logs/experiment_log.csv`
