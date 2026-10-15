"""Build the analysis files from the 2016-2020 GSS Panel and check them against the codebook.

Everyone in the panel file was first interviewed in the 2016 or 2018 GSS cross-section (the
baseline). All 2018 respondents and a random 74.85% of 2016 respondents were invited to a 2020
reinterview. The analysis sample is the 4,351 invited respondents who were still eligible in
2020. People who died, moved abroad, or entered an institution (143) are left out: they left the
population rather than declining to respond, which is the target Morgan and Lee (2020, GSS
Methodological Report 132) recommend for attrition weights.

Outputs (data/processed/):
  baseline.parquet - one row per eligible invited respondent: baseline answers and weights,
                     item-nonresponse counts, and whether they were reinterviewed
  wave2.parquet    - 2020 answers of the 1,823 who were reinterviewed
Output (output/tables/):
  codebook_checks.csv - figures reported in the panel codebook next to the values in this file
"""

import warnings

import numpy as np
import pandas as pd

from common import (DATA_FILE, DECEASED, NOT_REINTERVIEWED, NOT_SELECTED, OUT_OF_SCOPE,
                    PROCESSED, REINTERVIEWED, SUFFIX, TABLES, WAVE2)

# A few string values are stored as latin-1; none of them are used here.
warnings.filterwarnings("ignore", category=UnicodeWarning)

panel = pd.read_stata(DATA_FILE, convert_categoricals=False)
assert len(panel) == 5215 and panel["yearid"].is_unique

# Each baseline variable appears twice: _1a (2016 interview) and _1b (2018 interview).
names = [c.removesuffix(SUFFIX[2016]) for c in panel.columns if c.endswith(SUFFIX[2016])]
assert all(f"{v}{SUFFIX[2018]}" in panel for v in names)
from_2016 = panel["samptype"] == 2016


def harmonize(frame: pd.DataFrame) -> pd.DataFrame:
    """One column per baseline variable, taken from each respondent's own baseline year."""
    return pd.DataFrame({
        v: frame[f"{v}{SUFFIX[2016]}"].where(from_2016, frame[f"{v}{SUFFIX[2018]}"])
        for v in names
    })


base = harmonize(panel)

# Item nonresponse at baseline. Stata's extended missing codes separate questions not asked
# (.i: skip patterns and split ballots) from "don't know" (.d) and no answer (.n). The values
# above treat all three as missing, so the codes are read separately here and counted.
codes = harmonize(pd.read_stata(
    DATA_FILE, convert_categoricals=False, convert_missing=True,
    columns=[f"{v}{s}" for v in names for s in SUFFIX.values()],
)).map(lambda value: getattr(value, "string", None))
items = pd.DataFrame({
    "n_dont_know": (codes == ".d").sum(axis=1),
    "n_no_answer": (codes == ".n").sum(axis=1),
    "n_asked": (base.notna() | codes.isin([".d", ".n"])).sum(axis=1),
})

# Only part of the 2016 sample was invited back, so each invited 2016 respondent stands in for
# 2,867 / 2,146 baseline respondents. The baseline weight (WTSSNR, scaled to sum to each year's
# sample size) times this factor makes the invited sample represent both cross-sections.
n_baseline = panel["samptype"].value_counts()
n_invited_2016 = int((from_2016 & (panel["panstat"] != NOT_SELECTED)).sum())
subsample_factor = np.where(from_2016, n_baseline[2016] / n_invited_2016, 1.0)

status = pd.DataFrame({
    "yearid": panel["yearid"],
    "baseline_year": panel["samptype"],
    "panstat": panel["panstat"],
    "reinterviewed": (panel["panstat"] == REINTERVIEWED).astype(int),
    "base_weight": base["wtssnr"] * subsample_factor,
    "wtssnr_2": panel["wtssnr_2"],  # NORC's 2020 panel weight (reinterviewed only)
})

# --- Checks against the 2016-2020 GSS Panel Codebook (Release 1a) ---------------------------

checks = []


def check(name: str, codebook, ours, note: str = "") -> None:
    checks.append({"check": name, "codebook": codebook, "this_file": ours, "note": note})


def pct(x) -> str:
    return f"{100 * x:.1f}"


def weighted_retention(frame: pd.DataFrame) -> float:
    return np.average(frame["reinterviewed"], weights=frame["base_weight"])


# Sample sizes and panel status (codebook Table 2): must match exactly
codebook_panstat = {
    REINTERVIEWED: (809, 1014),
    NOT_SELECTED: (721, 0),
    NOT_REINTERVIEWED: (1261, 1267),
    DECEASED: (61, 45),
    OUT_OF_SCOPE: (15, 22),
}
counts = pd.crosstab(panel["panstat"], panel["samptype"])
for code, expected in codebook_panstat.items():
    observed = tuple(int(counts.loc[code, year]) for year in (2016, 2018))
    assert observed == expected, (code, observed, expected)
    check(f"PANSTAT {code}, 2016 / 2018", "{:,} / {:,}".format(*expected),
          "{:,} / {:,}".format(*observed))

assert (n_baseline[2016], n_baseline[2018]) == (2867, 2348)
check("Baseline respondents, 2016 / 2018", "2,867 / 2,348",
      f"{n_baseline[2016]:,} / {n_baseline[2018]:,}")
assert n_invited_2016 == 2146
check("2016 respondents invited to 2020", "2,146 (74.85%)",
      f"{n_invited_2016:,} ({100 * n_invited_2016 / n_baseline[2016]:.2f}%)")

# NORC's 2020 weight is scaled so each cohort sums to 911.5 (half of 1,823)
reinterviewed = status[status["reinterviewed"] == 1]
assert status["wtssnr_2"].notna().equals(status["reinterviewed"] == 1)
sums = reinterviewed.groupby("baseline_year")["wtssnr_2"].sum()
assert np.allclose(sums, 911.5)
check("WTSSNR_2 sum, 2016 / 2018 cohort", "911.5 / 911.5", f"{sums[2016]:.1f} / {sums[2018]:.1f}")

# Weighted retention: the cohort rates match when out-of-scope and deceased respondents are
# excluded; the overall rate matches only when they count as not retained.
eligible_mask = status["panstat"].isin([REINTERVIEWED, NOT_REINTERVIEWED])
eligible = status[eligible_mask]
invited = status[status["panstat"] != NOT_SELECTED]
for year, published in ((2016, 38.8), (2018, 44.9)):
    ours = weighted_retention(eligible[eligible["baseline_year"] == year])
    assert abs(100 * ours - published) < 0.15
    check(f"Weighted retention, {year} cohort (%)", f"{published}", pct(ours),
          "Eligible respondents, baseline WTSSNR")
check("Weighted retention, overall (%)", "40.4", pct(weighted_retention(eligible)),
      f"Eligible respondents; {pct(weighted_retention(invited))} if out-of-scope and deceased "
      "respondents count as not retained")

# Completion mode in 2020: codebook says 70.8% web, 29.2% phone
web = reinterviewed["yearid"].map(panel.set_index("yearid")["mode_2"]) == 4
check("2020 interviews completed on the web (%)", "70.8", pct(web.mean()),
      f"Not reproduced; {pct(np.average(web, weights=reinterviewed['wtssnr_2']))} weighted by WTSSNR_2")

TABLES.mkdir(parents=True, exist_ok=True)
checks = pd.DataFrame(checks)
checks.to_csv(TABLES / "codebook_checks.csv", index=False)
print(checks.to_string(index=False))

# --- Analysis files --------------------------------------------------------------------------

PROCESSED.mkdir(parents=True, exist_ok=True)
baseline = pd.concat([status, items, base], axis=1)[eligible_mask].reset_index(drop=True)
assert len(baseline) == 4351 and baseline["reinterviewed"].sum() == 1823
baseline.to_parquet(PROCESSED / "baseline.parquet", index=False)

wave2_columns = [c for c in panel.columns if c.endswith(WAVE2)]
wave2 = (panel.loc[panel["panstat"] == REINTERVIEWED, ["yearid", *wave2_columns]]
         .rename(columns=lambda c: c.removesuffix(WAVE2))
         .reset_index(drop=True))
wave2.to_parquet(PROCESSED / "wave2.parquet", index=False)

print(f"\nbaseline.parquet: {len(baseline):,} eligible respondents, "
      f"{baseline['reinterviewed'].sum():,} reinterviewed, {len(names):,} baseline variables")
print(f"wave2.parquet: {len(wave2):,} respondents, {len(wave2_columns):,} variables")
