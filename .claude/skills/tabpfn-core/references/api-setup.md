# TabPFN API Setup & Limits

## Installation & Authentication

```bash
pip install --upgrade tabpfn-client
```

```python
import tabpfn_client

# Interactive login (first time — opens browser):
# tabpfn_client.init()

# Or set token directly:
tabpfn_client.set_access_token("your_token")

# Verify and check credit usage:
from tabpfn_client import UserDataClient
print(UserDataClient.get_data_summary())
```

**What tabpfn-client is:** A cloud API client. Data is sent to Prior Labs' servers for inference. No local GPU required. Scikit-learn compatible interface.

---

## API Limits

| Constraint | Limit |
|-----------|-------|
| Max cells per request | `(train_rows + test_rows) × columns < 20,000,000` |
| Daily credits | 100,000,000 (reset 00:00 UTC) |
| Optimal sample range | ≤ 50,000 train rows |
| Optimal feature range | ≤ 2,000 columns |
| Regression full output | Test samples < 500 if `return_full_output=True` |

---

## Budget Check — Always Run Before Any API Call

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

You can also request the limit increase via the form. [Limit request form](https://docs.google.com/forms/d/e/1FAIpQLScrJ17Ozpukwrdlva-cmAEIXbudHmJ55C2SJ8-XB7N0_OcAOw/viewform)

---

## Credit Management Strategy

| Phase | Strategy |
|-------|----------|
| Early exploration | 1-fold tests, subsampled data |
| Feature engineering probing | 1-fold tests on full data |
| Full CV runs | Only for confirmed improvements |
| Final ensembles | Full CV on best feature sets |
| Credit check | `UserDataClient.get_data_summary()` |

- Use **1-fold tests** for exploration (1/5 the cost of full CV)
- Run **full 5-fold CV** only for promising changes
- Subsample for very early exploration if dataset is large
- Warn the user if daily credit burn rate is high
