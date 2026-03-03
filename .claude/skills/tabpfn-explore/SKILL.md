---
name: tabpfn-explore
description: EDA, data profiling, adversarial validation, preprocessing checks, CV scheme setup, and API budget verification for tabular Kaggle competitions. Run at the start of every new competition before any modeling.
---

> **Core context:** See [tabpfn-core](../tabpfn-core/SKILL.md) for behavior rules, workflow principles, and project conventions.
> **Preprocessing rules:** See [data-requirements.md](../tabpfn-core/references/data-requirements.md) for what TabPFN handles natively vs. what needs preparation.
> **API limits:** See [api-setup.md](../tabpfn-core/references/api-setup.md) for budget checks.

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

---

## 3. Adversarial Validation

Train a classifier to distinguish train from test rows. If AUC > 0.8, there is meaningful distribution shift — investigate which features drive the separation and decide whether to address it.

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
# > 0.8 → meaningful shift, investigate
# 0.6-0.8 → moderate shift, note it
# < 0.6 → train/test are similar, good
```

---

## 4. API Cell Budget Check

Always run before any TabPFN call. See [api-setup.md](../tabpfn-core/references/api-setup.md) for the full reference.

```python
n_train, n_test = len(train), len(test)
n_cols = train.drop(columns=["target"]).shape[1]
total_cells = (n_train + n_test) * n_cols
print(f"Total cells: {total_cells:,} / 20,000,000")
print("✅ OK" if total_cells < 20_000_000 else "EXCEEDS LIMIT — subsample or batching required")
```

---

## 5. Cross-Validation Scheme

Establish the CV scheme **before any modeling.** This is the source of truth for all experiments.

```python
from sklearn.model_selection import StratifiedKFold, KFold

N_FOLDS = 5
SEED = 42

# Classification — stratified to preserve class balance per fold
folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X_train, y_train))

# Regression — standard k-fold
folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X_train))

# Time-based: use TimeSeriesSplit
# Grouped: use GroupKFold with the appropriate group column
```

**Rules:**
- The CV scheme MUST mirror the competition's test distribution.
- Save the `folds` list and reuse the **exact same splits** for every single model. Non-negotiable for fair comparison and stacking.
- Record fold-level scores (not just the mean) to assess variance.

---

## 6. Output Deliverables

At the end of this phase, the following must exist:

1. `notes/competition_overview.md` — competition summary and key findings
2. Clean `X_train`, `y_train`, `X_test` DataFrames ready for TabPFN (see [data-requirements.md](../tabpfn-core/references/data-requirements.md))
3. `folds` — CV split indices reusable across all models
4. API budget check passed (or subsampling plan if not)
5. List of identified issues: leakage, shift, imbalance, high-cardinality, etc.
6. Feature lists: `id_cols`, `drop_cols`, `categorical_cols`, `numeric_cols`
