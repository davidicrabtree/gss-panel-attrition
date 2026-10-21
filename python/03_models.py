"""Predict who stayed in the panel: logistic regression against penalized regression, a random
forest, and gradient boosting. This script fits the models; 04_compare_models.py evaluates them.

Models (outcome = reinterviewed in 2020, among the 4,351 eligible):
  logit_literature  - logistic regression on the literature predictors (features.literature),
                      the specification attrition weights usually use
  logit_pca         - the same, plus principal components of the other questionnaire items
                      (the unsupervised step; the number of components is tuned)
  penalized_logit   - lasso (l1_ratio 1) or ridge (l1_ratio 0) logistic regression on all items
                      (penalty type and strength
                      are tuned)
  random_forest     - on all items
  boosting          - histogram gradient boosting on all items

Design:
  - Five-fold cross-validation, stratified by outcome and baseline year. Every respondent gets a
    prediction from a model that never saw them; step 4 builds weights from these out-of-fold
    ("cross-fitted") propensities, so no one's weight comes from a model fit to their own outcome.
  - Tuning happens inside each training fold (nested cross-validation), scored by log loss because
    weights need well-calibrated probabilities, not just a good ranking.
  - All preprocessing (encoding, standardization, PCA) is fit inside the training folds.
  - Cohort transfer: train on the 2018 cohort and predict the 2016 cohort, and the reverse.
  - Models are fit without survey weights: the question is who among the sampled responded.

Runs in 15-30 minutes on 8 cores.

Outputs:
  data/processed/propensities.parquet - per respondent and model: out-of-fold probability (p_)
                                        and probability from the model trained on the other
                                        cohort (transfer_)
  output/tables/model_tuning.csv      - hyperparameters chosen in each outer fold
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from common import PROCESSED, TABLES, load_baseline
from features import ItemEncoder, literature, questionnaire

SEED = 2020

baseline = load_baseline()
y = baseline["reinterviewed"].to_numpy()
lit, items = literature(baseline), questionnaire(baseline)
X = pd.concat([lit, items], axis=1)
LIT, ITEMS = list(lit.columns), list(items.columns)

# Nearly unpenalized: the literature model is meant to be ordinary logistic regression. A tiny
# penalty only keeps estimates finite for rare categories.
plain_logit = LogisticRegression(C=1e4, max_iter=5000)

# The literature predictors are imputed, as Morgan and Lee did, rather than given missing-value
# indicators. A few predictors are missing for a handful of people who almost all left; with
# indicators, a nearly unpenalized logit separates them, predicts probabilities near zero, and one
# retained respondent's weight reaches 179 times the median. Penalized and tree models keep the
# indicators, since their regularization handles rare flags.
literature_encoder = ItemEncoder(missing_indicators=False)

MODELS = {
    "logit_literature": (
        LIT,
        Pipeline([("encode", literature_encoder), ("model", plain_logit)]),
        {}),
    "logit_pca": (
        LIT + ITEMS,
        Pipeline([
            ("features", ColumnTransformer([
                ("literature", literature_encoder, LIT),
                ("components", Pipeline([("encode", ItemEncoder(missing_indicators=False)),
                                         ("pca", PCA(random_state=SEED))]), ITEMS),
            ])),
            ("model", plain_logit)]),
        {"features__components__pca__n_components": [5, 10, 20, 40]}),
    "penalized_logit": (
        LIT + ITEMS,
        # liblinear penalizes the intercept; a large intercept_scaling makes that penalty negligible
        Pipeline([("encode", ItemEncoder()),
                  ("model", LogisticRegression(solver="liblinear", intercept_scaling=100,
                                               random_state=SEED))]),
        {"model__l1_ratio": [0, 1], "model__C": list(np.logspace(-3, -0.5, 6))}),
    "random_forest": (
        LIT + ITEMS,
        Pipeline([("encode", ItemEncoder()),
                  ("model", RandomForestClassifier(n_estimators=500, random_state=SEED))]),
        {"model__min_samples_leaf": [5, 20, 50], "model__max_features": ["sqrt", 0.1]}),
    "boosting": (
        LIT + ITEMS,
        Pipeline([("encode", ItemEncoder()),
                  ("model", HistGradientBoostingClassifier(
                      learning_rate=0.05, min_samples_leaf=40, l2_regularization=1.0,
                      early_stopping=False, random_state=SEED))]),
        {"model__max_leaf_nodes": [4, 8, 16], "model__max_iter": [100, 300]}),
}


def tuned(name: str):
    columns, pipeline, grid = MODELS[name]
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipeline, grid or {"model__C": [plain_logit.C]}, cv=inner,
                          scoring="neg_log_loss", n_jobs=-1)
    return columns, search


# --- Cross-fitted predictions -------------------------------------------------------------------

warnings.filterwarnings("ignore", category=UserWarning)  # liblinear convergence chatter in tuning
outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
strata = baseline["reinterviewed"].astype(str) + baseline["baseline_year"].astype(str)
oof = pd.DataFrame(index=baseline.index, columns=list(MODELS), dtype=float)
tuning = []
for fold, (train, test) in enumerate(outer.split(X, strata), start=1):
    for name in MODELS:
        columns, search = tuned(name)
        search.fit(X.iloc[train][columns], y[train])
        oof.iloc[test, oof.columns.get_loc(name)] = search.predict_proba(X.iloc[test][columns])[:, 1]
        tuning.append({"fold": fold, "model": name, **{k.split("__")[-1]: v
                                                      for k, v in search.best_params_.items()}})
        print(f"fold {fold} {name}: {search.best_params_}")

# --- Cohort transfer: train on one baseline year, predict the other ----------------------------

transfer = pd.DataFrame(index=baseline.index, columns=list(MODELS), dtype=float)
for train_year in (2018, 2016):
    train = (baseline["baseline_year"] == train_year).to_numpy()
    for name in MODELS:
        columns, search = tuned(name)
        search.fit(X.loc[train, columns], y[train])
        transfer.loc[~train, name] = search.predict_proba(X.loc[~train, columns])[:, 1]

propensities = pd.concat([baseline[["yearid", "baseline_year", "reinterviewed"]],
                          oof.add_prefix("p_"), transfer.add_prefix("transfer_")], axis=1)
propensities.to_parquet(PROCESSED / "propensities.parquet", index=False)
pd.DataFrame(tuning).to_csv(TABLES / "model_tuning.csv", index=False)
