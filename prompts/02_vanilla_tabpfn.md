# Instructions: Vanilla TabPFN Running & Exploration

You are a data scientist running TabPFN v2.5 via the `tabpfn-client` cloud API to establish a strong baseline and rapidly explore the feature/signal landscape. Your goal is to produce a **credible first submission as fast as possible**, then use TabPFN's zero-tuning speed to test hypotheses before investing in heavier models.

**Prerequisites:** You must have completed the EDA & Preprocessing phase. You need:
- Clean `X_train`, `y_train`, `X_test` DataFrames (minimal preprocessing, raw categoricals)
- Saved `folds` (CV split indices)
- Confirmed API budget check
- Competition metric and task type identified

---

## 1. Setup — tabpfn-client

```bash
pip install --upgrade tabpfn-client
```

```python
import tabpfn_client

# Interactive login (first time — opens browser):
# tabpfn_client.init()

# Or set token directly:
tabpfn_client.set_access_token("your_token")

# Verify:
from tabpfn_client import UserDataClient
print(UserDataClient.get_data_summary())
```

**What tabpfn-client is:** A cloud API client. Data is sent to Prior Labs' servers. No local GPU required. Scikit-learn compatible interface.

---

## 2. What TabPFN v2.5 Does

TabPFN v2.5 is a transformer-based **tabular foundation model** using **in-context learning**. It makes predictions in a single forward pass — no training loop, no hyperparameters to tune. It is meta-trained on billions of synthetic tabular datasets.

**Properties that matter for competition use:**
- Matches AutoGluon 1.4 (a 4-hour tuned ensemble) in seconds
- Natively handles missing values, categoricals, outliers, mixed types
- Well-calibrated probability outputs (good for log-loss, AUC)
- Optimal for ≤50K samples, ≤2K features
- Scikit-learn API: `.fit()`, `.predict()`, `.predict_proba()`
- **No hyperparameter tuning needed** for a competitive baseline

---

## 3. Baseline — Classification

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

# Save predictions (critical for later ensembling)
np.save("oof/tabpfn_baseline_oof.npy", oof_preds)
np.save("oof/tabpfn_baseline_test.npy", test_preds)
```

## 4. Baseline — Regression

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

np.save("oof/tabpfn_baseline_oof.npy", oof_preds)
np.save("oof/tabpfn_baseline_test.npy", test_preds)
```

---

## 5. Generate & Submit Baseline

```python
import pandas as pd

submission = pd.read_csv("data/raw/sample_submission.csv")

# Classification — probabilities or labels depending on submission format
# submission["target"] = test_preds[:, 1]          # for probability submission
# submission["target"] = np.argmax(test_preds, 1)  # for label submission

# Regression
# submission["target"] = test_preds

submission.to_csv(f"submissions/tabpfn_baseline_cv{cv_mean:.4f}.csv", index=False)

# Sanity checks before submitting
assert submission.shape == pd.read_csv("data/raw/sample_submission.csv").shape
assert submission.isnull().sum().sum() == 0
print("✅ Submission file ready")
```

**This is your anchor point.** Log the CV score AND the LB score. The relationship between them tells you how trustworthy your CV is:
- CV ≈ LB → Great. Trust your CV for all future decisions.
- CV ≫ LB or CV ≪ LB → Investigate. Possible distribution shift, leakage, or flawed CV scheme.

---

## 6. Log the Experiment

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
    notes="First TabPFN baseline. No feature engineering."
)
```

---

## 7. Rapid Exploration — Leverage TabPFN's Speed

TabPFN needs no tuning. This makes it the **fastest hypothesis tester** in your toolkit. Use it to explore the feature space before investing time in GBT tuning.

### Feature Ablation (find noisy features)
```python
baseline_cv = cv_mean  # from above

for col in X_train.columns:
    X_drop = X_train.drop(columns=[col])
    X_test_drop = X_test.drop(columns=[col])

    # Quick 1-fold test to save API credits
    train_idx, val_idx = folds[0]
    clf.fit(X_drop.iloc[train_idx], y_train.iloc[train_idx])
    score = compute_metric(
        y_train.iloc[val_idx],
        clf.predict_proba(X_drop.iloc[val_idx])
    )
    delta = score - baseline_cv
    marker = "🗑️ DROP" if delta > 0.001 else ""
    print(f"  Drop {col:30s} → {score:.6f} (Δ {delta:+.6f}) {marker}")
```

### Feature Engineering Probing
```python
# Add a candidate feature and instantly test impact
X_train_exp = X_train.copy()
X_test_exp = X_test.copy()

X_train_exp["new_feat"] = X_train["a"] / (X_train["b"] + 1e-8)
X_test_exp["new_feat"] = X_test["a"] / (X_test["b"] + 1e-8)

# Quick 1-fold test
train_idx, val_idx = folds[0]
clf.fit(X_train_exp.iloc[train_idx], y_train.iloc[train_idx])
new_score = compute_metric(
    y_train.iloc[val_idx],
    clf.predict_proba(X_train_exp.iloc[val_idx])
)
print(f"New feature impact: {new_score - baseline_cv:+.6f}")
```

### Batch Feature Testing
```python
candidate_features = {
    "a_div_b": X_train["a"] / (X_train["b"] + 1e-8),
    "a_minus_b": X_train["a"] - X_train["b"],
    "a_times_b": X_train["a"] * X_train["b"],
    "log_a": np.log1p(X_train["a"].clip(lower=0)),
}

results = {}
for feat_name, feat_values in candidate_features.items():
    X_exp = X_train.copy()
    X_exp[feat_name] = feat_values
    # 1-fold quick test...
    results[feat_name] = score_delta

# Keep only features that improve score
good_features = [f for f, delta in results.items() if delta > 0.0005]
print(f"Features to keep: {good_features}")
```

### API Credit Conservation Tips
- Use **1-fold tests** for exploration (1/5 the cost of full CV)
- Run **full 5-fold CV** only for promising changes
- Subsample for very early exploration if dataset is large
- Track your credit usage: `UserDataClient.get_data_summary()`

---

## 8. Output Deliverables

At the end of this phase, the following must exist:

1. **Submitted baseline** with known CV and LB scores
2. **CV-LB relationship** documented (trustworthy or not)
3. **OOF predictions** saved: `oof/tabpfn_baseline_oof.npy`
4. **Test predictions** saved: `oof/tabpfn_baseline_test.npy`
5. **Feature exploration results:** Which features help, which are noise, which new features improve score
6. **Experiment log** entry for the baseline
7. **List of promising directions** for Part 3 optimization
