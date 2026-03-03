# Kaggle Competition Skill for Claude

A skill collection for tabular Kaggle competitions using **TabPFN v2.5** — the transformer-based tabular foundation model from Prior Labs. Drop these skills into any competition project and go from data to first submission in under 30 minutes.

---

## What This Is

[TabPFN](https://github.com/PriorLabs/TabPFN) is a pretrained transformer that does in-context learning on tabular data. You pass it your training data, it makes predictions — no training loop, no hyperparameter tuning. Via the `tabpfn-client` repo + cloud API it matchesand outperforms AutoGluon 1.4 (a 4-hour tuned ensemble) in seconds.

This skill collection gives Claude a structured workflow for using TabPFN as a competition baseline tool, then layering GBT ensembles and feature engineering on top of it. The philosophy: **get something on the leaderboard first, optimize second.**

You'll need a `tabpfn-client` API token to use cloud inference - [get it on Prior Labs' UX portal](https://ux.priorlabs.ai/). See `.claude/skills/tabpfn-core/references/api-setup.md` for setup details.

---

## Skills

```
.claude/skills/
├── tabpfn-core/                  # Parent skill — loaded by the others, not invoked directly
│   ├── SKILL.md                  # Identity, behavior rules, workflow, principles
│   └── references/
│       ├── api-setup.md          # tabpfn-client installation, auth, limits, budget checking
│       └── data-requirements.md  # What TabPFN handles natively vs. what needs prep
├── tabpfn-explore/
│   └── SKILL.md                  # EDA, adversarial validation, CV scheme setup
├── tabpfn-classify/
│   └── SKILL.md                  # Classification baseline → GBT ensembling → threshold tuning
└── tabpfn-regress/
    └── SKILL.md                  # Regression baseline → GBT ensembling → clipping/transforms
```

### tabpfn-core

The shared foundation. Defines Claude's role as a pragmatic competition data scientist, the behavior rules (bias toward action, log everything, protect API credits), the four-step workflow, and the core principles. Not user-invocable — it's background context loaded by the three task skills.

### tabpfn-explore

Run this at the start of every competition. Covers:
- Competition reconnaissance (metric, task type, dataset dimensions, leakage risks)
- EDA essentials (missing values, class imbalance, high-cardinality categoricals)
- Adversarial validation to detect train/test distribution shift
- API cell budget check
- CV scheme setup (StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit)

Produces the clean DataFrames and `folds` object that the other two skills depend on.

### tabpfn-classify

For binary and multi-class classification. Covers:
- TabPFN baseline with full CV and OOF predictions saved
- Submission generation with sanity checks
- Rapid feature exploration (ablation and probing, 1-fold tests to conserve credits)
- Feature engineering patterns (aggregations, target encoding, interactions, frequency encoding)
- TabPFN-specific techniques (subsample bagging, feature subset diversity)
- Ensemble blending with optimized weights
- Threshold optimization and calibration

### tabpfn-regress

For regression tasks. Same structure as tabpfn-classify with regression-specific additions:
- Target transformation (log, sqrt, rank) with back-transform at predict time
- Rank averaging for blending across different prediction scales
- Clipping and rounding post-processing

---

## Installation

Copy the `.claude/` directory into your competition project root:

```bash
cp -r .claude/ /path/to/your/competition/
```

Or install as personal skills (available in all projects):

```bash
cp -r .claude/skills/ ~/.claude/skills/
```

---

## Usage

Once installed, invoke a skill with `/` in Claude Code:

| Situation | Skill to invoke |
|-----------|----------------|
| New competition, need to understand the data | `/tabpfn-explore` |
| Classification task, need a baseline | `/tabpfn-classify` |
| Regression task, need a baseline | `/tabpfn-regress` |

**Typical session flow:**

```
1. Drop your data into data/raw/
2. /tabpfn-explore  →  understand the data, establish CV folds
3. /tabpfn-classify  →  baseline on the board in <30 minutes
4. Iterate: feature engineering, GBT ensembling, blending
```

---

## API Limits (tabpfn-client)

| Constraint | Limit |
|-----------|-------|
| Max cells per request | `(train_rows + test_rows) × columns < 20,000,000` |
| Daily credits | 100,000,000 (reset 00:00 UTC) |
| Optimal range | ≤ 50K samples, ≤ 2K features |

The skills always check this before any API call and will warn you if you're approaching limits. If you'd like to extend the limits, reach out to Prior Labs via our [limit request form](https://docs.google.com/forms/d/e/1FAIpQLScrJ17Ozpukwrdlva-cmAEIXbudHmJ55C2SJ8-XB7N0_OcAOw/viewform)

---

## Requirements

```
tabpfn-client>=0.1
lightgbm
xgboost
catboost
optuna
scikit-learn
pandas
numpy
scipy
```
