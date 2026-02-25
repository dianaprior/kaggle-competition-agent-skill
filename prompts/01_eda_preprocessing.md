# Instructions: EDA, Data Understanding & Preprocessing for TabPFN

You are a data scientist performing exploratory data analysis and data preparation for a tabular Kaggle competition. Your output will feed directly into TabPFN v2.5 (via `tabpfn-client`) and gradient boosted tree models. Be thorough but efficient — every finding should connect to a modeling decision.

---

## 1. Competition Reconnaissance

Before touching data, gather and document context:

- Read the competition description, evaluation metric, data description, and rules **thoroughly**.
- Read top-voted discussion posts and public notebooks for domain insights and known pitfalls.
- Identify: task type (binary classification, multi-class, regression), exact evaluation metric, dataset size (rows × columns), whether test labels are hidden, and any temporal/grouped structure.
- Check for leakage vectors: does the test set overlap with training? Are there features derived from the target or from future data?
- Write a concise summary to `notes/competition_overview.md` covering: metric, task type, dataset dimensions, key risks, and initial hypotheses.

---

## 2. EDA Essentials

Profile the data systematically. Focus on findings that change modeling decisions.

```python
import pandas as pd
import numpy as np

train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Total cells (train+test): {(len(train) + len(test)) * train.shape[1]:,}")
print(train.dtypes.value_counts())

# Missing values
missing = train.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# Target distribution
print(train["target"].describe())
print(train["target"].value_counts(normalize=True))

# Cardinality of categoricals
for col in train.select_dtypes(include=["object", "category"]).columns:
    print(f"{col}: {train[col].nunique()} unique values")
```

**Checklist — flag each as ✅ done or ⚠️ issue found:**

| Check | Why It Matters |
|-------|---------------|
| Target leakage (feature ≈ perfect predictor) | Invalidates any model score |
| Train/test distribution shift | Causes CV-LB divergence |
| Duplicate rows | Can inflate CV or cause leakage |
| Class imbalance severity | Affects metric, sampling, thresholds |
| ID columns and constants | Must remove — zero signal, wastes cell budget |
| High-cardinality categoricals (>500 levels) | May need frequency encoding even for TabPFN |
| Feature correlations (>0.95) | Redundancy; keep one, drop the other |

### Adversarial Validation
Train a classifier to distinguish train from test rows. If AUC > 0.8, there is meaningful distribution shift — investigate which features drive the separation and decide whether to address it (e.g., drop leaky features, reweight samples).

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

combined = pd.concat([train.drop(columns=["target"]), test], ignore_index=True)
labels = np.array([0]*len(train) + [1]*len(test))
adv_auc = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    combined.select_dtypes(include="number"), labels, cv=5, scoring="roc_auc"
).mean()
print(f"Adversarial Validation AUC: {adv_auc:.4f}")
```

---

## 3. Data Preprocessing for TabPFN

TabPFN v2.5 requires **minimal preprocessing**. This is one of its biggest advantages. Understand exactly what to do and what NOT to do.

### DO NOT preprocess these (TabPFN handles them natively):
- **Missing values** — Leave NaN. Do NOT impute.
- **Categorical features** — Pass as `object`/`category` dtype. Do NOT one-hot encode or label encode.
- **Outliers** — Leave as-is. Do NOT clip or winsorize.
- **Feature scaling** — Do NOT standardize or normalize.
- **Mixed types** — Numeric + categorical together is fine.

### DO preprocess these:
- **Remove ID columns** with no predictive signal.
- **Remove constant / near-constant columns** (zero variance).
- **Remove confirmed leakage features.**
- **Fix dtypes:** Ensure categoricals are `object`/`category` (not integer-encoded). Ensure numerics are `float`/`int`.
- **Extreme cardinality (>500 unique strings):** Consider adding a frequency-encoded version of the column alongside the original. This gives TabPFN an easier numeric signal without losing the raw categorical.

### Prepare a Clean Dataset
```python
def prepare_for_tabpfn(df, id_cols=None, drop_cols=None):
    """Minimal preprocessing for TabPFN."""
    df = df.copy()
    cols_to_drop = []
    if id_cols:
        cols_to_drop.extend(id_cols)
    if drop_cols:
        cols_to_drop.extend(drop_cols)
    # Drop constants
    for col in df.columns:
        if df[col].nunique() <= 1:
            cols_to_drop.append(col)
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df
```

---

## 4. API Cell Budget Check

The `tabpfn-client` API has hard constraints. Check these BEFORE making any API call.

```python
def check_tabpfn_budget(X_train, X_test):
    n_train, n_test, n_cols = len(X_train), len(X_test), X_train.shape[1]
    total_cells = (n_train + n_test) * n_cols
    print(f"Train rows: {n_train:,}")
    print(f"Test rows:  {n_test:,}")
    print(f"Columns:    {n_cols:,}")
    print(f"Total cells: {total_cells:,} / 20,000,000")
    print(f"Within limit: {'✅' if total_cells < 20_000_000 else '❌ EXCEEDS LIMIT'}")
    if n_train > 50_000:
        print(f"⚠️  Train rows ({n_train:,}) exceed TabPFN's optimal range (≤50K). Consider subsampling.")
    if n_cols > 2_000:
        print(f"⚠️  Columns ({n_cols:,}) exceed TabPFN's optimal range (≤2K). Consider feature selection.")
    return total_cells < 20_000_000
```

| Constraint | Limit |
|-----------|-------|
| Max cells per request | `(train_rows + test_rows) × columns < 20,000,000` |
| Daily credits | 100,000,000 |
| Optimal sample range | ≤ 50,000 train rows |
| Optimal feature range | ≤ 2,000 columns |
| Regression full output | Test samples < 500 if `return_full_output=True` |

---

## 5. Subsampling Strategy (When Data Exceeds Limits)

```python
from sklearn.model_selection import StratifiedShuffleSplit

def subsample_for_tabpfn(X, y, max_samples=10000, random_state=42):
    """Stratified subsample for TabPFN when data exceeds limits."""
    if len(X) <= max_samples:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=random_state)
    idx, _ = next(sss.split(X, y))
    return X.iloc[idx], y.iloc[idx]

# For regression (no stratification):
def subsample_regression(X, y, max_samples=10000, random_state=42):
    if len(X) <= max_samples:
        return X, y
    idx = np.random.RandomState(random_state).choice(len(X), max_samples, replace=False)
    return X.iloc[idx], y.iloc[idx]
```

**If data is much larger than 50K:** Plan to use TabPFN on subsamples for ensemble diversity while GBTs serve as primary models on full data.

**If features exceed 2K:** Use a quick LightGBM feature importance to select top features for TabPFN. Feed full feature set to GBTs.

---

## 6. Cross-Validation Scheme

Establish the CV scheme **before any modeling.** This is the source of truth for all experiments.

```python
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

N_FOLDS = 5
SEED = 42

# Classification
folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X_train, y_train))

# Regression
folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X_train))

# Time-based: use TimeSeriesSplit
# Grouped: use GroupKFold with appropriate group column
```

**Rules:**
- The CV scheme MUST mirror the competition's test distribution.
- Save the `folds` list and reuse the **exact same splits** for every single model. Non-negotiable for fair comparison and stacking.
- Record fold-level scores, not just the mean, to assess variance.

---

## 7. Output Deliverables

At the end of this phase, the following must exist:

1. `notes/competition_overview.md` — competition summary and key findings
2. Clean `X_train`, `y_train`, `X_test` DataFrames ready for TabPFN (minimal preprocessing)
3. `folds` — saved CV split indices, reusable across all models
4. Confirmed API budget check passes (or subsampling plan if it doesn't)
5. List of identified issues: leakage, shift, imbalance, high-cardinality, etc.
6. Feature lists: `id_cols`, `drop_cols`, `categorical_cols`, `numeric_cols`
