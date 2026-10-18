"""Compare the attrition models fit in 03_models.py, using their saved predictions.

Every metric uses predictions for respondents the model never saw: out-of-fold predictions from
five-fold cross-validation, and cohort-transfer predictions from a model trained on the other
baseline year.

Metrics:
  AUC                - how well the model ranks stayers above leavers (0.5 = chance)
  Brier score        - mean squared error of the predicted probability (lower is better)
  log loss           - the scoring rule the models were tuned on (lower is better)
  calibration slope  - from regressing the outcome on the log-odds of the prediction: 1 is ideal;
                       below 1, predictions are too spread out (overfit); above 1, too compressed.
                       This matters for weights, which are inverse probabilities.

Differences from the literature logit come with 95% intervals from 1,000 bootstrap resamples of
respondents (paired: each resample scores every model on the same people).

Outputs:
  output/tables/model_comparison.csv  - metrics for every model
  output/tables/model_differences.csv - each model vs logit_literature, bootstrap 95% intervals
  output/figures/calibration.png      - predicted vs observed reinterview rates by decile
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from common import (BLUE, FIGURES, MODEL_LABELS, TABLES, load_propensities, save_figure,
                    set_plot_style)

SEED = 2020
N_BOOT = 1000

propensities = load_propensities()
y = propensities["reinterviewed"].to_numpy()
MODELS = list(MODEL_LABELS)
BASELINE_MODEL = MODELS[0]


def calibration_slope(y_true, p) -> float:
    fit = sm.Logit(y_true, sm.add_constant(np.log(p / (1 - p)))).fit(disp=0)
    return fit.params[1]


def metrics(y_true, p) -> dict:
    return {"auc": roc_auc_score(y_true, p), "brier": brier_score_loss(y_true, p),
            "log_loss": log_loss(y_true, p), "calibration_slope": calibration_slope(y_true, p),
            "mean_predicted": p.mean(), "observed": y_true.mean()}


# --- Comparison table ---------------------------------------------------------------------------

rows = []
for name in MODELS:
    row = {"model": name, "label": MODEL_LABELS[name],
           **metrics(y, propensities[f"p_{name}"].to_numpy())}
    for year in (2016, 2018):
        cohort = (propensities["baseline_year"] == year).to_numpy()
        transfer = metrics(y[cohort], propensities.loc[cohort, f"transfer_{name}"].to_numpy())
        row[f"auc_{year}_trained_on_other"] = transfer["auc"]
        row[f"brier_{year}_trained_on_other"] = transfer["brier"]
    rows.append(row)
comparison = pd.DataFrame(rows)
comparison.round(4).to_csv(TABLES / "model_comparison.csv", index=False)
print(comparison.drop(columns="label").round(3).to_string(index=False))

# --- Paired bootstrap of differences from the literature logit ----------------------------------

P = propensities[[f"p_{name}" for name in MODELS]].to_numpy()
rng = np.random.default_rng(SEED)
auc_draws, brier_draws = [], []
for _ in range(N_BOOT):
    i = rng.integers(0, len(y), len(y))
    auc = np.array([roc_auc_score(y[i], P[i, j]) for j in range(len(MODELS))])
    brier = np.array([brier_score_loss(y[i], P[i, j]) for j in range(len(MODELS))])
    auc_draws.append(auc[1:] - auc[0])
    brier_draws.append(brier[1:] - brier[0])
auc_draws, brier_draws = np.array(auc_draws), np.array(brier_draws)

point = comparison.set_index("model")
differences = pd.DataFrame({
    "model": MODELS[1:],
    "auc_difference": point.loc[MODELS[1:], "auc"].to_numpy() - point.loc[BASELINE_MODEL, "auc"],
    "auc_ci_low": np.percentile(auc_draws, 2.5, axis=0),
    "auc_ci_high": np.percentile(auc_draws, 97.5, axis=0),
    "brier_difference": point.loc[MODELS[1:], "brier"].to_numpy() - point.loc[BASELINE_MODEL, "brier"],
    "brier_ci_low": np.percentile(brier_draws, 2.5, axis=0),
    "brier_ci_high": np.percentile(brier_draws, 97.5, axis=0),
})
differences.round(4).to_csv(TABLES / "model_differences.csv", index=False)
print(f"\nDifference from {BASELINE_MODEL} (bootstrap 95% intervals)")
print(differences.round(4).to_string(index=False))

# --- Figure: calibration ------------------------------------------------------------------------

set_plot_style()
fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 3.3), sharex=True, sharey=True)
for ax, name in zip(axes, MODELS):
    p = propensities[f"p_{name}"]
    deciles = pd.DataFrame({"p": p, "y": y}).groupby(pd.qcut(p, 10, labels=False)).mean()
    ax.plot([0.15, 0.75], [0.15, 0.75], color="0.6", linewidth=1, linestyle="--")
    ax.scatter(deciles["p"], deciles["y"], color=BLUE, s=28, zorder=3, edgecolor="white",
               linewidth=1.2)
    ax.text(0.04, 0.97, MODEL_LABELS[name].replace(", ", ",\n").replace(" + ", "\n+ "),
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="0.2")
    ax.text(0.96, 0.04, f"slope {point.loc[name, 'calibration_slope']:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="0.35")
    ax.grid(axis="y", visible=True, color="0.92")
    ax.set_xlim(0.15, 0.75)
    ax.set_ylim(0.15, 0.75)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
axes[0].set_ylabel("Observed reinterview rate", color="0.3")
fig.supxlabel("Predicted reinterview probability (out of fold)", fontsize=10, color="0.3")
save_figure(fig, FIGURES / "calibration.png",
            "Boosting's probabilities are the best calibrated",
            "Respondents grouped into tenths by predicted probability; on the dashed line, "
            "predictions match observed rates")
plt.close(fig)
