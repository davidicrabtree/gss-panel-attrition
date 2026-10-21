"""The bias test: do attrition weights recover what the whole eligible sample said at baseline?

For each of the 15 benchmark indicators (features.BENCHMARKS, held out of every model), the target
is its baseline mean among all 4,351 eligible respondents, weighted by d. Each weighting scheme
estimates the same quantity from the 1,823 reinterviewed respondents alone. The gap between the
two is the attrition bias the scheme leaves behind. This test is possible only because people who
left the panel still have their baseline answers in the file.

Summary: the mean absolute gap across the 15 benchmarks, in percentage points. Gaps would not be
zero even if attrition were completely random, because the reinterviewed are a 42% subsample. The
"noise floor" simulates that case: random subsamples of 1,823 with baseline weights.

Uncertainty: 1,000 bootstrap resamples of sampling units, recomputing the targets, raking, and
propensity classes each time. Model probabilities are held fixed (refitting the models 1,000 times
is not feasible), so the intervals understate uncertainty somewhat. Runs in 5-10 minutes.

Outputs:
  output/tables/bias_summary.csv     - mean absolute gap and effective sample size per scheme
  output/tables/bias_comparisons.csv - differences between schemes, with 95% intervals
  output/tables/benchmark_gaps.csv   - gap for every benchmark and scheme, with 95% intervals
  output/figures/bias_summary.png    - mean absolute gap and effective sample size by scheme
  output/figures/benchmark_gaps.png  - each benchmark, before and after the best scheme
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from common import (BLUE, FIGURES, GRAY, TABLES, cluster_id, grouped_rows, label_rows,
                    load_baseline, load_propensities, save_figure, set_plot_style)
from features import BENCHMARKS, benchmarks
from weighting import (SHORT_MODEL_NAMES, all_schemes, effective_sample_size, raking_margins,
                       scheme_label, scheme_method, scheme_model)

SEED = 2020
N_BOOT = 1000

baseline = load_baseline()
propensities = load_propensities()
assert baseline["yearid"].equals(propensities["yearid"])
frame = pd.concat([baseline, propensities.filter(like="p_")], axis=1)
margins = raking_margins(baseline)
values = benchmarks(baseline)


def gaps(frame: pd.DataFrame, margins: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """Scheme x benchmark gaps (percentage points) between the weighted reinterviewed estimate
    and the baseline-weighted mean of the whole sample. Respondents not asked a benchmark item
    (split ballots) are left out of both."""
    weights = all_schemes(frame, margins)
    base = frame["base_weight"].to_numpy()
    asked = values.notna().to_numpy().astype(float)
    v = values.fillna(0).to_numpy()
    target = (base @ (v * asked)) / (base @ asked)
    W = weights.to_numpy()
    estimate = (W.T @ (v * asked)) / (W.T @ asked)
    return pd.DataFrame(100 * (estimate - target), index=weights.columns, columns=values.columns)


observed = gaps(frame, margins, values)
schemes = list(observed.index)
mean_abs = observed.abs().mean(axis=1)

# --- Cluster bootstrap --------------------------------------------------------------------------

rng = np.random.default_rng(SEED)
cluster = cluster_id(baseline).to_numpy()
rows_of = {c: np.flatnonzero(cluster == c) for c in np.unique(cluster)}
units = np.array(list(rows_of))
draws = []
for _ in range(N_BOOT):
    rows = np.concatenate([rows_of[c] for c in rng.choice(units, len(units))])
    draws.append(gaps(frame.iloc[rows].reset_index(drop=True),
                      margins.iloc[rows].reset_index(drop=True),
                      values.iloc[rows].reset_index(drop=True)).to_numpy())
draws = np.array(draws)                       # replicate x scheme x benchmark
mean_abs_draws = np.abs(draws).mean(axis=2)  # replicate x scheme

# --- Noise floor: what random attrition would leave ---------------------------------------------

n, n_kept = len(baseline), int(baseline["reinterviewed"].sum())
base = baseline["base_weight"].to_numpy()
asked = values.notna().to_numpy().astype(float)
v = values.fillna(0).to_numpy()
target = (base @ (v * asked)) / (base @ asked)
floor = []
for _ in range(N_BOOT):
    kept = np.zeros(n)
    kept[rng.choice(n, n_kept, replace=False)] = 1
    estimate = ((base * kept) @ (v * asked)) / ((base * kept) @ asked)
    floor.append(np.abs(100 * (estimate - target)).mean())
floor = np.array(floor)

# --- Tables -------------------------------------------------------------------------------------

weights = all_schemes(frame, margins)
retained = baseline["reinterviewed"] == 1
summary = pd.DataFrame({
    "scheme": schemes,
    "label": [scheme_label(s) for s in schemes],
    "mean_abs_gap": mean_abs.to_numpy(),
    "ci_low": np.percentile(mean_abs_draws, 2.5, axis=0),
    "ci_high": np.percentile(mean_abs_draws, 97.5, axis=0),
    "reduction_vs_base_pct": 100 * (1 - mean_abs.to_numpy() / mean_abs["base"]),
    "effective_n": [effective_sample_size(weights.loc[retained, s]) for s in schemes],
})
floor_row = pd.DataFrame([{"scheme": "random_attrition", "label": "Random attrition (simulated)",
                           "mean_abs_gap": floor.mean(), "ci_low": np.percentile(floor, 2.5),
                           "ci_high": np.percentile(floor, 97.5)}])
summary = pd.concat([summary, floor_row], ignore_index=True)
summary.round(3).to_csv(TABLES / "bias_summary.csv", index=False)

# Each scheme against four references: no attrition adjustment, NORC's weight, the standard
# demographic alternative, and the literature model
REFERENCES = ["base", "norc", "raking", "ipw_logit_literature"]
rows = []
for reference in REFERENCES:
    j = schemes.index(reference)
    for i, scheme in enumerate(schemes):
        if scheme != reference:
            diff = mean_abs_draws[:, i] - mean_abs_draws[:, j]
            rows.append({"scheme": scheme, "reference": reference,
                         "difference": mean_abs.iloc[i] - mean_abs.iloc[j],
                         "ci_low": np.percentile(diff, 2.5), "ci_high": np.percentile(diff, 97.5)})
comparisons = pd.DataFrame(rows)
comparisons.round(3).to_csv(TABLES / "bias_comparisons.csv", index=False)

gap_table = pd.DataFrame({
    "scheme": np.repeat(schemes, len(values.columns)),
    "benchmark": np.tile(values.columns, len(schemes)),
    "gap": observed.to_numpy().ravel(),
    "ci_low": np.percentile(draws, 2.5, axis=0).ravel(),
    "ci_high": np.percentile(draws, 97.5, axis=0).ravel(),
})
gap_table["target_pct"] = gap_table["benchmark"].map(pd.Series(100 * target, index=values.columns))
gap_table.round(3).to_csv(TABLES / "benchmark_gaps.csv", index=False)

pd.set_option("display.width", 250)
print(summary.drop(columns="scheme").round(2).to_string(index=False))
print("\nDifferences in mean absolute gap (bootstrap 95% intervals)")
print(comparisons.round(2).to_string(index=False))

# --- Figures ------------------------------------------------------------------------------------

set_plot_style()

# Figure 1: mean absolute gap and effective sample size, one row per scheme
plot = summary[summary["scheme"] != "random_attrition"].assign(
    group=lambda d: [f"Weights from the {SHORT_MODEL_NAMES[m]} model" if (m := scheme_model(s))
                     else "Without a model" for s in d["scheme"]],
    item=lambda d: d["scheme"].map(scheme_method))
rows = grouped_rows(plot, group="group", item="item")
data = rows[~rows["header"]]
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 0.25 * len(rows) + 1.2), sharey=True,
                                  gridspec_kw={"width_ratios": [2, 1]})
y = label_rows(left, rows)
yd = y[~rows["header"].to_numpy()]
left.axvline(floor.mean(), color="0.5", linestyle="--", linewidth=1)
left.text(floor.mean(), len(rows) - 0.3, f" Random attrition: {floor.mean():.1f}", ha="left",
          va="bottom", fontsize=9, color="0.35")
left.hlines(yd, data["ci_low"], data["ci_high"], color=BLUE, linewidth=2)
left.scatter(data["mean_abs_gap"], yd, color=BLUE, s=40, zorder=3, edgecolor="white", linewidth=1.5)
left.set_xlim(0, 3.9)
left.set_xlabel("Mean absolute gap across 15 benchmarks (percentage points)", color="0.3")
right.scatter(data["effective_n"], yd, color=BLUE, s=40, zorder=3, edgecolor="white", linewidth=1.5)
right.set_xlim(1050, 1900)
right.set_xticks([1200, 1500, 1800])
right.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
right.set_xlabel("Effective sample size (of 1,823)", color="0.3")
right.grid(axis="y", visible=False)
cut = summary.set_index("scheme")["reduction_vs_base_pct"]
model_cut = cut[[s for s in schemes if scheme_model(s)]]
save_figure(fig, FIGURES / "bias_summary.png",
            f"Model-based weights cut the attrition gap by {model_cut.min():.0f}-{model_cut.max():.0f}%; "
            f"NORC's panel weight by {cut['norc']:.0f}%",
            "Gap between the weighted 2020 panel and the full baseline sample on 15 held-out "
            "benchmarks, with 95% intervals")
plt.close(fig)

# Figure 2: each benchmark with no attrition adjustment and with the scheme that has the
# smallest mean gap
best = mean_abs.drop(["unweighted", "base"]).idxmin()
order = observed.loc["base"].abs().sort_values(ascending=False).index
yb = np.arange(len(order))[::-1]
fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(order) + 1.3))
ax.set_yticks(yb, [BENCHMARKS[b][0].split(" (")[0] for b in order])
ax.grid(axis="y", visible=False)
ax.axvline(0, color="0.4", linewidth=1)
for scheme, shift, style in (
        ("base", 0.15, {"facecolor": "white", "edgecolor": GRAY, "s": 36}),
        (best, -0.15, {"color": BLUE, "edgecolor": "white", "s": 40})):
    g = gap_table[gap_table["scheme"] == scheme].set_index("benchmark").loc[order]
    line_color = GRAY if scheme == "base" else BLUE
    ax.hlines(yb + shift, g["ci_low"], g["ci_high"], color=line_color, linewidth=2)
    ax.scatter(g["gap"], yb + shift, zorder=3, linewidth=1.5, **style,
               label="No attrition adjustment" if scheme == "base" else "Best weights")
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}" if v else "0")
ax.set_xlabel("Weighted 2020 panel minus full baseline sample (percentage points)", color="0.3")
save_figure(fig, FIGURES / "benchmark_gaps.png",
            "Who stayed: more voters, more trusting, more liberal",
            "Baseline answers of 2020 panel members minus those of all eligible respondents, "
            f"with 95% intervals.\nBest weights: the scheme with the smallest mean gap "
            f"({scheme_label(best).lower()}).")
plt.close(fig)
