# Instructions: TabPFN Optimization & Winning Strategies

You are a data scientist in the optimization phase of a tabular Kaggle competition. You have a TabPFN baseline submitted and a working CV pipeline. Now you push for maximum score through feature engineering, model diversity, ensembling, and post-processing.

**Prerequisites:** You must have completed Parts 1 and 2. You need:
- Submitted TabPFN baseline with known CV and LB scores
- Saved OOF and test predictions for the baseline
- Feature exploration results from Part 2
- Working CV pipeline with identical folds across all models
- Credit budget awareness for the tabpfn-client API

---

## 1. Feature Engineering — Where Competitions Are Won

TabPFN eliminates preprocessing busywork but **feature engineering still matters enormously.** The goal: create features that encode signal the model can't easily learn from raw columns.

### High-Value Feature Patterns

**Aggregation features** — Group-by statistics on meaningful categorical groupings:
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
from sklearn.model_selection import KFold

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

**Frequency / count encoding:**
```python
freq = X_train[col].value_counts(normalize=True)
X_train[f"{col}_freq"] = X_train[col].map(freq)
X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0)
```

**Time-based features** (if temporal data): Lags, rolling windows, time-since, cyclical encoding (sin/cos).

**Dimensionality reduction:** PCA or UMAP components as supplementary features when raw feature count is high.

### Feature Selection

After generating candidate features, prune aggressively:

- **TabPFN ablation** (from Part 2): Drop features one-by-one, measure CV impact.
- **LightGBM importance:** Quick LightGBM → rank by `feature_importances_` → drop bottom 20%.
- **Null importance test:** Shuffle the target, retrain, measure feature importance. Real features should score much higher than on shuffled targets.
- **Correlation dedup:** For pairs with correlation >0.95, keep one.

---

## 2. GBT Models for Ensemble Diversity

TabPFN + GBTs have **fundamentally different inductive biases** (transformer vs decision trees). This is ideal for ensembling.

### LightGBM with Optuna HPO
```python
import lightgbm as lgb
import optuna

def lgb_objective(trial):
    params = {
        "objective": "binary",           # adapt: "regression", "multiclass"
        "metric": "auc",                 # adapt to competition metric
        "verbosity": -1,
        "n_estimators": 10000,
        "learning_rate": trial.suggest_float("lr", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
    }

    scores = []
    for train_idx, val_idx in folds:
        dtrain = lgb.Dataset(X_train.iloc[train_idx], y_train.iloc[train_idx])
        dval = lgb.Dataset(X_train.iloc[val_idx], y_train.iloc[val_idx])
        model = lgb.train(
            params, dtrain,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_train.iloc[val_idx])
        scores.append(compute_metric(y_train.iloc[val_idx], preds))
    return np.mean(scores)

study = optuna.create_study(direction="maximize")  # or "minimize" for RMSE etc.
study.optimize(lgb_objective, n_trials=100)
```

### XGBoost & CatBoost
Train with similar Optuna HPO. CatBoost handles categoricals natively via `cat_features`.

### Generate OOF Predictions for Every Model
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

**Always save OOF + test predictions for every model.** You need these for ensembling.

---

## 3. TabPFN-Specific Optimization

### Multiple Subsample Ensembles
If dataset is large, create multiple TabPFN predictions from different stratified subsamples:
```python
from tabpfn_client import TabPFNClassifier

n_bags = 5
max_samples = 10000
bag_test_preds = []

for i in range(n_bags):
    X_sub, y_sub = subsample_for_tabpfn(X_train, y_train,
                                         max_samples=max_samples,
                                         random_state=SEED + i)
    clf = TabPFNClassifier()
    clf.fit(X_sub, y_sub)
    bag_test_preds.append(clf.predict_proba(X_test))

tabpfn_bagged = np.mean(bag_test_preds, axis=0)
```

### Feature Subset Diversity
Multiple TabPFN models on different feature subsets:
```python
import random

subsets = []
for i in range(3):
    random.seed(SEED + i)
    n_select = int(len(all_features) * 0.7)
    subsets.append(random.sample(all_features, n_select))

# Train TabPFN on each subset → 3 diverse prediction sets for blending
```

### TabPFN on Engineered Features vs Raw Features
Run TabPFN separately on raw features and on engineered features. Blend both — they often capture different signals.

---

## 4. Ensembling — The Winning Move

### Strategy 1: Weighted Blending
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

### Strategy 2: Rank Averaging
Robust when models produce predictions on different scales:
```python
from scipy.stats import rankdata

rank_preds = [rankdata(p) / len(p) for p in test_predictions_list]
final_test = np.mean(rank_preds, axis=0)
```

### Strategy 3: Stacking
Meta-learner on OOF predictions:
```python
from sklearn.linear_model import Ridge, LogisticRegression

meta_train = np.column_stack(oof_list)
meta_test = np.column_stack(test_list)

# Regression
meta = Ridge(alpha=1.0)
meta.fit(meta_train, y_train)
final_test = meta.predict(meta_test)

# Classification
meta = LogisticRegression(C=1.0, max_iter=1000)
meta.fit(meta_train, y_train)
final_test = meta.predict_proba(meta_test)
```

### Ensembling Rules
- **NEVER** blend models validated on different CV splits.
- **Diversity > individual strength.** A weaker model that disagrees adds more value.
- **TabPFN + GBTs is a high-diversity combo** — different architectures, different biases.
- More model types = better: consider Ridge/Lasso, KNN, random forests for extra diversity.

---

## 5. Post-Processing

### Threshold Optimization (Classification)
```python
# For F1, macro-F1, or other threshold-sensitive metrics
thresholds = np.arange(0.1, 0.9, 0.01)
best_thresh, best_score = 0.5, -np.inf
for t in thresholds:
    preds = (oof_proba[:, 1] > t).astype(int)
    score = compute_metric(y_train, preds)
    if score > best_score:
        best_thresh, best_score = t, score
print(f"Optimal threshold: {best_thresh:.2f} → {best_score:.4f}")
```

### Clipping & Rounding
```python
final_test = np.clip(final_test, y_train.min(), y_train.max())
if is_discrete:
    final_test = np.round(final_test).astype(int)
```

### Calibration
TabPFN outputs are generally well-calibrated. GBT probabilities may not be. If the metric rewards calibration (log-loss, Brier score):
```python
from sklearn.calibration import CalibratedClassifierCV
# Calibrate individual GBT models before ensembling
```

---

## 6. TabPFN Role by Dataset Size

| Dataset Size | TabPFN Role | GBT Role |
|-------------|-------------|----------|
| ≤ 5K samples | **Primary** | Ensemble diversity |
| 5K–50K samples | **Co-primary** | Co-primary — blend both |
| 50K–100K | Diversity (on subsamples) | **Primary** |
| > 100K | Optional (small subsamples) | **Primary** |
| > 2K features | Top features only | **Primary** (full features) |

---

## 7. API Credit Management

Budget your `tabpfn-client` calls across the competition timeline:

| Phase | Strategy |
|-------|----------|
| Early exploration | 1-fold tests, subsampled data |
| Feature engineering probing | 1-fold tests on full data |
| Full CV runs | Only for confirmed improvements |
| Final ensembles | Full CV on best feature sets |
| Credit check | `UserDataClient.get_data_summary()` |

---

## 8. Common Mistakes to Avoid

- ❌ Over-preprocessing for TabPFN (don't impute, scale, encode)
- ❌ Only TabPFN without GBT diversity
- ❌ Only GBTs without trying TabPFN
- ❌ Exceeding API cell limits without checking first
- ❌ Burning credits on full-CV experiments early
- ❌ Different CV splits across models (breaks stacking)
- ❌ Overfitting to public LB
- ❌ Target leakage in feature engineering
- ❌ Submitting without sanity checks (shape, NaNs, range, format)
- ❌ Skipping experiment logging

---

## 9. Final Submission Checklist

1. ✅ CV score logged for every model and ensemble variant
2. ✅ CV-LB relationship consistent
3. ✅ OOF predictions saved for all models
4. ✅ Submission: correct shape, no NaNs, valid ranges, correct columns
5. ✅ Two final picks: one "safe" (best CV), one "risky" (aggressive ensemble)
6. ✅ All experiments logged in `logs/experiment_log.csv`
