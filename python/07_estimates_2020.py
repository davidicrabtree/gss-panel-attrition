"""What the weights change: 2020 estimates and baseline-to-2020 change under each scheme.

The benchmark indicators from the bias test were asked again in 2020, so the same 15 measures can
be estimated for 2020 (levels) and as within-person change (each respondent's 2020 answer minus
their baseline answer, averaged). Voting is a level only: the 2020 wave asked about the 2016
election, which is not the baseline question for the 2016 cohort.

There is no true 2020 value to compare against. What this step shows is how far each scheme
moves the estimates relative to NORC's released weight; the bias test (06_bias_test.py) is the
evidence on which schemes move them in the right direction. Changes also mix real change with
the 2020 switch from face-to-face to web and phone interviews; the mode effect is common to every
weighting scheme, so it does not affect the comparison between schemes.

Uncertainty: the same cluster bootstrap as the bias test (1,000 resamples of sampling units, with
raking and propensity classes rebuilt each time). Runs in about 5 minutes.

Outputs:
  output/tables/estimates_2020.csv - level and change estimates by scheme, and each scheme's
                                     shift from NORC's weight, with 95% intervals
  output/figures/shifts_2020.png   - shift from NORC's weight for the scheme that did best in
                                     the bias test (chosen on baseline benchmarks, not on 2020)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (BLUE, FIGURES, PROCESSED, TABLES, cluster_id, load_baseline,
                    load_propensities, save_figure, set_plot_style)
from features import BENCHMARKS, benchmarks
from weighting import all_schemes, raking_margins, scheme_label

SEED = 2020
N_BOOT = 1000
SCHEMES = ["unweighted", "base", "norc", "raking", "ipw_logit_literature",
           "ipw_penalized_logit", "ipw_boosting"]
REFERENCE = "norc"
BEST = "ipw_penalized_logit"  # smallest mean gap in the bias test (output/tables/bias_summary.csv)

baseline = load_baseline()
propensities = load_propensities()
assert baseline["yearid"].equals(propensities["yearid"])
frame = pd.concat([baseline, propensities.filter(like="p_")], axis=1)
margins = raking_margins(baseline)

# 2020 answers lined up with the baseline rows (missing for people who left). In 2020 everyone
# was asked VOTE16, which is the variable the voting rule reads for the 2018 cohort.
wave2 = pd.read_parquet(PROCESSED / "wave2.parquet")
wave2 = baseline[["yearid"]].merge(wave2, on="yearid", how="left", validate="1:1")
after = benchmarks(wave2.assign(baseline_year=2018))
before = benchmarks(baseline)
change = (after - before).drop(columns="voted")


def estimates(frame: pd.DataFrame, margins: pd.DataFrame, after: pd.DataFrame,
              change: pd.DataFrame) -> pd.DataFrame:
    """Weighted means (percentage points) of 2020 levels and within-person changes, by scheme."""
    W = all_schemes(frame, margins)[SCHEMES].to_numpy()
    out = {}
    for measure, values in (("level", after), ("change", change)):
        known = values.notna().to_numpy().astype(float)
        v = values.fillna(0).to_numpy()
        means = (W.T @ (v * known)) / (W.T @ known)
        for j, name in enumerate(values.columns):
            out[(measure, name)] = 100 * means[:, j]
    return pd.DataFrame(out, index=SCHEMES)


observed = estimates(frame, margins, after, change)

rng = np.random.default_rng(SEED)
cluster = cluster_id(baseline).to_numpy()
rows_of = {c: np.flatnonzero(cluster == c) for c in np.unique(cluster)}
units = np.array(list(rows_of))
draws = []
for _ in range(N_BOOT):
    rows = np.concatenate([rows_of[c] for c in rng.choice(units, len(units))])
    draws.append(estimates(*(d.iloc[rows].reset_index(drop=True)
                             for d in (frame, margins, after, change))).to_numpy())
draws = np.array(draws)  # replicate x scheme x (measure, outcome)
ref = SCHEMES.index(REFERENCE)
shift_draws = draws - draws[:, [ref], :]

table = []
for i, scheme in enumerate(SCHEMES):
    for k, (measure, outcome) in enumerate(observed.columns):
        table.append({
            "outcome": outcome, "label": BENCHMARKS[outcome][0], "measure": measure,
            "scheme": scheme, "scheme_label": scheme_label(scheme),
            "estimate": observed.iloc[i, k],
            "ci_low": np.percentile(draws[:, i, k], 2.5),
            "ci_high": np.percentile(draws[:, i, k], 97.5),
            "shift_vs_norc": observed.iloc[i, k] - observed.iloc[ref, k],
            "shift_ci_low": np.percentile(shift_draws[:, i, k], 2.5),
            "shift_ci_high": np.percentile(shift_draws[:, i, k], 97.5),
        })
table = pd.DataFrame(table)
table.round(3).to_csv(TABLES / "estimates_2020.csv", index=False)

pd.set_option("display.width", 250)
for measure in ("level", "change"):
    print(f"\n2020 {measure} (percentage points)")
    print(observed[measure].T.round(1).to_string())
    shifts = table[(table["measure"] == measure) & (table["scheme"] != REFERENCE)]
    print(f"\nMean absolute shift from NORC's weight, {measure}:")
    print(shifts.groupby("scheme", sort=False)["shift_vs_norc"].apply(lambda s: s.abs().mean())
          .round(2).to_string())

# --- Figure: shift from NORC's weight, levels and changes ---------------------------------------

set_plot_style()
best = table[table["scheme"] == BEST]
order = (best[best["measure"] == "level"].set_index("outcome")["shift_vs_norc"]
         .sort_values().index)
y = np.arange(len(order))[::-1]
fig, axes = plt.subplots(1, 2, figsize=(9, 0.32 * len(order) + 1.4), sharey=True, sharex=True)
for ax, measure, title in zip(axes, ("level", "change"),
                              ("Shift in the 2020 estimate", "Shift in change since baseline")):
    d = best[best["measure"] == measure].set_index("outcome").reindex(order)
    ax.axvline(0, color="0.4", linewidth=1)
    ax.hlines(y, d["shift_ci_low"], d["shift_ci_high"], color=BLUE, linewidth=2)
    ax.scatter(d["shift_vs_norc"], y, color=BLUE, s=40, zorder=3, edgecolor="white", linewidth=1.5)
    ax.set_xlabel(title, color="0.3")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}" if v else "0")
axes[0].set_yticks(y, [BENCHMARKS[o][0].split(" (")[0] for o in order])
axes[1].text(0, y[list(order).index("voted")], "not measured the same way", ha="center",
             va="center", fontsize=8.5, color="0.5",
             bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
save_figure(fig, FIGURES / "shifts_2020.png",
            "The best weights move 2020 levels by up to 3.6 points but not change estimates",
            f"{scheme_label(BEST).capitalize()} weights (best in the bias test) minus NORC's "
            "released weight,\npercentage points, with 95% intervals")
plt.close(fig)
