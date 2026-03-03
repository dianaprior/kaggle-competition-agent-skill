# TabPFN Data Requirements & Preprocessing

TabPFN v2.5 requires **minimal preprocessing**. This is one of its biggest advantages — understand exactly what to do and what NOT to do.

---

## What TabPFN Handles Natively — DO NOT Preprocess

| Aspect | Action |
|--------|--------|
| Missing values | Leave `NaN`. Do **NOT** impute. |
| Categorical features | Pass as `object`/`category` dtype. Do **NOT** one-hot or label encode. |
| Outliers | Leave as-is. Do **NOT** clip or winsorize. |
| Feature scaling | Do **NOT** standardize or normalize. |
| Mixed types | Numeric + categorical together is fine. |

---

## What to Preprocess — DO Prepare These

- **Remove ID columns** with no predictive signal.
- **Remove constant / near-constant columns** (zero variance).
- **Remove confirmed leakage features.**
- **Fix dtypes:** Ensure categoricals are `object`/`category` (not integer-encoded). Ensure numerics are `float`/`int`.
- **Extreme cardinality (>500 unique strings):** Add a frequency-encoded version alongside the original — gives TabPFN an easier numeric signal without losing the raw categorical.

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

## Size & Shape Constraints

| Dataset Size | TabPFN Role | GBT Role |
|-------------|-------------|----------|
| ≤ 5K samples | **Primary** | NA |
| 5K–50K samples | **Primary** | NA |
| 50K–100K | **Primary** | NA |
| > 100K |  **Primary** (subsamples under 100K) | **Primary** (full dataset) |
| > 2K features |  **Primary** (subsamples under 2K features, better to go for 100s features) | **Primary** (full dataset) |

---

## Subsampling When Data Exceeds Limits

```python
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def subsample_for_tabpfn(X, y, max_samples=10000, random_state=42):
    """Stratified subsample for classification when data exceeds limits."""
    if len(X) <= max_samples:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=random_state)
    idx, _ = next(sss.split(X, y))
    return X.iloc[idx], y.iloc[idx]

def subsample_regression(X, y, max_samples=10000, random_state=42):
    """Random subsample for regression when data exceeds limits."""
    if len(X) <= max_samples:
        return X, y
    idx = np.random.RandomState(random_state).choice(len(X), max_samples, replace=False)
    return X.iloc[idx], y.iloc[idx]
```

**If data >> 50K:** Use TabPFN on subsamples for ensemble diversity; GBTs serve as primary models on full data.

**If features > 2K:** Use a quick LightGBM feature importance to select top features for TabPFN. Feed full feature set to GBTs.
