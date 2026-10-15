"""Who left the panel: weighted 2020 reinterview rates by baseline characteristics, and attrition
gaps compared with the 2006-2014 GSS panels.

Rates use the baseline weight (WTSSNR times the 2016 subsampling factor), so they describe the
population the 2016 and 2018 cross-sections represent. Standard errors are clustered by sampling
unit (see common.weighted_shares).

The comparison uses Morgan and Lee (2020, GSS Methodological Report 132), Table 3, column 3:
bivariate differences in the probability of attrition among base-year respondents who stayed
eligible, 2006-2014 panels pooled (N = 5,732). For a single categorical predictor, a bivariate
logit is saturated, so its average probability difference equals the difference in weighted
attrition rates computed here. Their attrition is leaving within four years (two reinterviews,
face to face); here it is missing one reinterview after two to four years, by web or phone,
during the pandemic.

Outputs (output/tables/):
  retention_by_group.csv       - weighted reinterview rate for each baseline group
  attrition_vs_2006_panels.csv - attrition gaps here and in the 2006-2014 panels, and the change
Outputs (output/figures/):
  retention_by_group.png       - reinterview rates for selected characteristics
  attrition_vs_2006_panels.png - attrition gaps in both panels
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import (BLUE, FIGURES, TABLES, characteristics, cluster_id, load_baseline,
                    save_figure, set_plot_style, weighted_shares)

Z = 1.96

baseline = load_baseline()
groups = characteristics(baseline)
retained = baseline["reinterviewed"]
weight = baseline["base_weight"]
cluster = cluster_id(baseline)

# --- Reinterview rate by group ----------------------------------------------------------------

overall = np.average(retained, weights=weight)
rows = []
for name, group in groups.items():
    fit = weighted_shares(retained, group, weight, cluster)
    n = group.value_counts()
    for category, rate in fit.params.items():
        rows.append({
            "characteristic": name, "group": category, "n": int(n[category]),
            "reinterviewed_pct": 100 * rate,
            "ci_low": 100 * (rate - Z * fit.bse[category]),
            "ci_high": 100 * (rate + Z * fit.bse[category]),
        })
rates = pd.DataFrame(rows)
rates.round(1).to_csv(TABLES / "retention_by_group.csv", index=False)
print(f"Overall weighted reinterview rate: {100 * overall:.1f}% (N = {len(baseline):,})\n")
print(rates.round(1).to_string(index=False))

# --- Attrition gaps compared with the 2006-2014 panels -----------------------------------------

# (characteristic, group, reference group, Morgan and Lee gap, its standard error), in
# percentage points of attrition
MORGAN_LEE = [
    ("Sex", "Female", "Male", -2.0, 1.5),
    ("Race and ethnicity", "Black", "White", 0.1, 2.2),
    ("Race and ethnicity", "Hispanic", "White", 13.5, 2.4),
    ("Education", "Less than high school", "High school", 13.2, 2.3),
    ("Education", "Junior college", "High school", 0.1, 2.7),
    ("Education", "Bachelor's", "High school", -1.1, 2.0),
    ("Education", "Graduate", "High school", -3.9, 2.5),
    ("Marital status", "Widowed", "Married", 6.2, 3.0),
    ("Marital status", "Divorced", "Married", -3.4, 2.1),
    ("Marital status", "Separated", "Married", 8.7, 4.9),
    ("Marital status", "Never married", "Married", 3.5, 1.8),
    ("Housing", "Rents", "Owns", 7.2, 1.6),
    ("Housing", "Other", "Owns", 14.5, 7.4),
    ("Citizenship", "Citizen, born abroad", "Born in the U.S.", 1.8, 2.9),  # their "naturalized"
    ("Citizenship", "Not a citizen", "Born in the U.S.", 31.4, 3.0),
    ("Interview language", "Spanish", "English", 25.1, 3.6),
    ("Interview mode", "Phone", "In person", 3.7, 2.5),
    ("Interviewer-rated attitude", "Cooperative", "Friendly and interested", 6.4, 2.1),
    # Their "restless, impatient" row; hostile respondents (+46.4 there) are too few here to separate
    ("Interviewer-rated attitude", "Restless or hostile", "Friendly and interested",
     17.3, 3.9),
    ("Interviewer-rated understanding", "Fair", "Good", 11.6, 2.3),
    ("Interviewer-rated understanding", "Poor", "Good", 20.2, 5.4),
]


def gap_row(name, category, reference, gap, se, theirs, their_se) -> dict:
    """Both gaps, and the change between panels. The panels are independent samples, so the
    standard error of the change combines the two standard errors."""
    change, change_se = 100 * gap - theirs, np.hypot(100 * se, their_se)
    return {"characteristic": name, "group": category, "reference": reference,
            "gap_2016_2020": 100 * gap, "ci_low": 100 * (gap - Z * se), "ci_high": 100 * (gap + Z * se),
            "gap_2006_2014": theirs, "ci_low_2006_2014": theirs - Z * their_se,
            "ci_high_2006_2014": theirs + Z * their_se,
            "change": change, "change_ci_low": change - Z * change_se,
            "change_ci_high": change + Z * change_se}


dropped = 1 - retained
rows = []
for name, category, reference, theirs, their_se in MORGAN_LEE:
    fit = weighted_shares(dropped, groups[name], weight, cluster, reference=reference)
    rows.append(gap_row(name, category, reference, fit.params[category], fit.bse[category],
                        theirs, their_se))

# Vocabulary score enters as a count: attrition difference per word correct
asked = baseline["wordsum"].notna()
fit = sm.WLS(dropped[asked], sm.add_constant(baseline.loc[asked, "wordsum"]),
             weights=weight[asked]).fit(cov_type="cluster", cov_kwds={"groups": cluster[asked]})
rows.append(gap_row("Vocabulary test", "Per word correct (of 10)", "", fit.params["wordsum"],
                    fit.bse["wordsum"], -3.3, 0.4))

comparison = pd.DataFrame(rows)
comparison.round(1).to_csv(TABLES / "attrition_vs_2006_panels.csv", index=False)
print("\nAttrition gap vs reference group, percentage points")
print(comparison.round(1).to_string(index=False))

# --- Figures ----------------------------------------------------------------------------------

set_plot_style()
FIGURES.mkdir(parents=True, exist_ok=True)
GRAY = "0.6"


def grouped_rows(table: pd.DataFrame, header=lambda name: name) -> pd.DataFrame:
    """Insert a header row before each characteristic; returns rows in top-to-bottom order."""
    out = []
    for name, block in table.groupby("characteristic", sort=False):
        out.append({"label": header(name), "header": True})
        out += [{**row, "label": row["group"], "header": False} for row in block.to_dict("records")]
    return pd.DataFrame(out)


def label_rows(ax, rows: pd.DataFrame) -> np.ndarray:
    """Rows run top to bottom. Groups label the y axis; each header starts at the figure's left
    edge on its own empty row (a white box keeps reference lines from crossing it)."""
    y = np.arange(len(rows))[::-1]
    ax.set_yticks(y, ["" if h else lab for lab, h in zip(rows["label"], rows["header"])])
    for yi, lab, h in zip(y, rows["label"], rows["header"]):
        if h:
            ax.annotate(lab, xy=(0.02, yi), xycoords=("figure fraction", "data"), ha="left",
                        va="center", fontweight="bold", fontsize=10, color="0.15", zorder=5,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    return y


# Figure 1: reinterview rates for the characteristics with the largest and most policy-relevant
# differences, chosen to show both socioeconomic and interview-experience gradients
SHOWN = ["Education", "Household income", "Voted in last presidential election", "Political views",
         "Race and ethnicity", "Age", "Interviewer-rated attitude",
         "Don't know / no answer rate"]
rows = grouped_rows(rates[rates["characteristic"].isin(SHOWN)]
                    .assign(order=lambda d: d["characteristic"].map(SHOWN.index))
                    .sort_values("order", kind="stable"))
fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(rows) + 0.9))
y = label_rows(ax, rows)
data = rows[~rows["header"]]
yd = y[~rows["header"].to_numpy()]
ax.axvline(100 * overall, color="0.5", linestyle="--", linewidth=1)
ax.text(100 * overall, len(rows) - 0.3, f" All eligible: {100 * overall:.1f}%", ha="left",
        va="bottom", fontsize=9, color="0.35")
ax.hlines(yd, data["ci_low"], data["ci_high"], color=BLUE, linewidth=2)
ax.scatter(data["reinterviewed_pct"], yd, color=BLUE, s=40, zorder=3, edgecolor="white",
           linewidth=1.5)
ax.set_xlim(15, 65)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
save_figure(fig, FIGURES / "retention_by_group.png",
            "Who came back in 2020",
            "Share of eligible 2016 and 2018 GSS respondents reinterviewed in 2020, weighted, "
            "with 95% confidence intervals")
plt.close(fig)

# Figure 2: attrition gaps in both panels (vocabulary is per word, so it stays in the table)
shown = comparison[comparison["characteristic"] != "Vocabulary test"]
references = shown.groupby("characteristic", sort=False)["reference"].first()
rows = grouped_rows(shown, header=lambda name: f"{name} (vs {references[name]})")
fig, ax = plt.subplots(figsize=(7.5, 0.26 * len(rows) + 1.1))
y = label_rows(ax, rows)
data = rows[~rows["header"]]
yd = y[~rows["header"].to_numpy()]
ax.axvline(0, color="0.4", linewidth=1)
offset = 0.17
ax.hlines(yd + offset, data["ci_low_2006_2014"], data["ci_high_2006_2014"], color=GRAY, linewidth=2)
ax.scatter(data["gap_2006_2014"], yd + offset, s=36, zorder=3, facecolor="white", edgecolor=GRAY,
           linewidth=1.5, label="2006-2014 panels (Morgan and Lee 2020)")
ax.hlines(yd - offset, data["ci_low"], data["ci_high"], color=BLUE, linewidth=2)
ax.scatter(data["gap_2016_2020"], yd - offset, s=40, zorder=3, color=BLUE, edgecolor="white",
           linewidth=1.5, label="2016-2020 panel")
ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}" if v else "0")
ax.set_xlabel("Difference in attrition rate from the reference group (percentage points)",
              color="0.3")
ax.legend(loc="upper right", frameon=False, fontsize=9)
save_figure(fig, FIGURES / "attrition_vs_2006_panels.png",
            "Dropout now follows education more and citizenship less",
            "Attrition gaps in the 2006-2014 GSS panels and the 2016-2020 panel, "
            "with 95% confidence intervals")
plt.close(fig)
