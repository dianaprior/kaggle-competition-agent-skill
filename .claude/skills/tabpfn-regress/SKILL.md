---
name: tabpfn-regress
description: Run a TabPFN regression baseline, generate the first submission, then optimize with GBT ensembles and regression-specific post-processing (clipping, target transforms, rank blending). Use after tabpfn-explore has prepared the data and CV folds.
---

> **Core context:** See [tabpfn-core](../tabpfn-core/SKILL.md) for behavior rules, workflow principles, and project conventions.
> **API setup:** See [api-setup.md](../tabpfn-core/references/api-setup.md) — always check budget before any API call.
> **Data prep:** See [data-requirements.md](../tabpfn-core/references/data-requirements.md) for preprocessing rules.

**Prerequisites:** Clean `X_train`, `y_train`, `X_test` DataFrames; saved `folds`; API budget confirmed; evaluation metric identified.

---

## What TabPFN v2.5 Is

A transformer-based **tabular foundation model** using in-context learning. No training loop, no hyperparameter tuning. Optimal for ≤50K samples, ≤2K features. Scikit-learn API.

---

## 1. Regression Baseline

```python
from tabpfn_client import TabPFNRegressor
import numpy as np

reg = TabPFNRegressor()

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    X_tr = X_train.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]

    reg.fit(X_tr, y_tr)

    oof_preds[val_idx] = reg.predict(X_val)
    test_preds += reg.predict(X_test) / len(folds)

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
submission["target"] = test_preds
submission.to_csv(f"submissions/tabpfn_baseline_cv{cv_mean:.4f}.csv", index=False)

# Sanity checks
assert submission.shape == pd.read_csv("data/raw/sample_submission.csv").shape
assert submission.isnull().sum().sum() == 0
print("✅ Submission file ready")
```

**Anchor point:** Log CV and LB scores. CV ≈ LB means the CV scheme is trustworthy. Divergence requires investigation before continuing.

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
    notes="First TabPFN regression baseline."
)
```

---

## 4. Target Transformation

For skewed regression targets, consider transforming before modeling:

```python
# Log-transform (common for price, count, revenue targets)
y_train_log = np.log1p(y_train)
# Train model on y_train_log, then back-transform predictions:
test_preds_original = np.expm1(test_preds_log)

# Rank-transform (when distribution is highly irregular)
from scipy.stats import rankdata
y_train_rank = rankdata(y_train) / len(y_train)

# Square-root transform (moderate skew)
y_train_sqrt = np.sqrt(y_train)
```

**Rule:** Whatever transformation you apply to `y_train`, apply the inverse to predictions before scoring or submitting.

---

## 5. Rapid Feature Exploration

Use 1-fold tests to conserve API credits.

```python
baseline_cv = cv_mean
train_idx, val_idx = folds[0]

# Test a candidate feature
X_train_exp = X_train.copy()
X_test_exp = X_test.copy()
X_train_exp["new_feat"] = X_train["a"] / (X_train["b"] + 1e-8)
X_test_exp["new_feat"] = X_test["a"] / (X_test["b"] + 1e-8)

reg.fit(X_train_exp.iloc[train_idx], y_train.iloc[train_idx])
new_score = compute_metric(
    y_train.iloc[val_idx],
    reg.predict(X_train_exp.iloc[val_idx])
)
print(f"New feature impact: {new_score - baseline_cv:+.6f}")
```

---

## 6. Feature Engineering (Optimization Phase)

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

**Interaction features:**
```python
X_train["a_div_b"] = X_train["a"] / (X_train["b"] + 1e-8)
X_train["a_minus_b"] = X_train["a"] - X_train["b"]
X_train["log_a"] = np.log1p(X_train["a"].clip(lower=0))
```

**Frequency encoding:**
```python
freq = X_train[col].value_counts(normalize=True)
X_train[f"{col}_freq"] = X_train[col].map(freq)
X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0)
```

---

## 7. GBT Models for Ensemble Diversity

```python
import lightgbm as lgb
import optuna

def lgb_objective(trial):
    params = {
        "objective": "regression",   # or "regression_l1" for MAE
        "metric": "rmse",            # adapt to competition metric
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
    return np.mean(scores)  # direction="minimize" for RMSE/MAE

# Don't run without user confirmation for large n_trials
study = optuna.create_study(direction="minimize")
study.optimize(lgb_objective, n_trials=100)
```

**Generate OOF for every GBT model:**
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

## 8. Ensembling

### Weighted Blending
```python
from scipy.optimize import minimize

oof_list = [oof_tabpfn, oof_lgbm, oof_xgb, oof_catboost]
test_list = [test_tabpfn, test_lgbm, test_xgb, test_catboost]

def blend_objective(weights):
    w = np.array(weights)
    w = w / w.sum()
    blended = sum(wi * pi for wi, pi in zip(w, oof_list))
    return compute_metric(y_train, blended)  # use + for minimization metrics

n = len(oof_list)
result = minimize(blend_objective, x0=[1/n]*n, method="Nelder-Mead")
weights = np.array(result.x); weights = weights / weights.sum()
final_test = sum(w * p for w, p in zip(weights, test_list))
```

### Rank Averaging (robust across different prediction scales)
```python
from scipy.stats import rankdata
rank_preds = [rankdata(p) / len(p) for p in test_list]
final_test = np.mean(rank_preds, axis=0)
```

### Stacking
```python
from sklearn.linear_model import Ridge

meta_train = np.column_stack(oof_list)
meta_test = np.column_stack(test_list)
meta = Ridge(alpha=1.0)
meta.fit(meta_train, y_train)
final_test = meta.predict(meta_test)
```

---

## 9. Post-Processing

### Clipping & Rounding
```python
# Clip to training target range
final_test = np.clip(final_test, y_train.min(), y_train.max())

# Round if target is discrete (integer counts, etc.)
final_test = np.round(final_test).astype(int)
```

### Back-Transform If Using Target Transform
```python
# If you log-transformed: final_test = np.expm1(final_test)
# If you sqrt-transformed: final_test = np.square(final_test)
```

---

## 10. Common Mistakes to Avoid

- Over-preprocessing for TabPFN (don't impute, scale, encode)
- Forgetting to back-transform predictions after target transformation
- Only TabPFN without GBT diversity — or only GBTs without TabPFN
- Exceeding API cell limits without checking first
- Different CV splits across models (breaks stacking)
- Target leakage in feature engineering
- Submitting without sanity checks (shape, NaNs, range, format)

---

## 11. Final Submission Checklist

1. ✅ CV score logged for every model and ensemble variant
2. ✅ CV-LB relationship consistent
3. ✅ OOF predictions saved for all models
4. ✅ Submission: correct shape, no NaNs, values in valid range
5. ✅ Two final picks: one "safe" (best CV), one "risky" (aggressive ensemble)
6. ✅ All experiments in `logs/experiment_log.csv`
