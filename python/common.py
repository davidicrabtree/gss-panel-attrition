"""Shared paths, constants, baseline characteristics, and helpers for the GSS panel attrition
analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"

# 2016-2020 GSS Panel, Release 1a (April 2022), Stata format, from the GSS "Get the Data" page
DATA_URL = ("https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/"
            "GSS_2020_panel_stata_1a.zip")
DATA_ZIP_SHA256 = "d1ae45c793a4fc3f8bbf4af65b11762c8d814106cecd11f244a2831d2272dc6d"
DATA_FILE = RAW / "gss2020panel_r1a.dta"

# Variable suffixes in the panel file: the baseline interview came from either the 2016 or the
# 2018 cross-section (wave 1a or 1b); the reinterview was in 2020 (wave 2).
SUFFIX = {2016: "_1a", 2018: "_1b"}
WAVE2 = "_2"

# PANSTAT codes (codebook Table 2)
REINTERVIEWED = 1
NOT_SELECTED = 0
NOT_REINTERVIEWED = 2
OUT_OF_SCOPE = 3
DECEASED = 33

BLUE = "#2A6F97"


def load_baseline() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "baseline.parquet")


def cluster_id(df: pd.DataFrame) -> pd.Series:
    """Sampling unit for standard errors. Variance strata are numbered separately for 2016
    (3201-3266) and 2018 (3301-3378), with two units each, so stratum and unit identify a cluster."""
    return df["vstrat"].astype(int) * 10 + df["vpsu"].astype(int)


def _categorize(values: pd.Series, mapping: dict) -> pd.Series:
    """Map codes to labels; labels keep the order of `mapping`, codes not listed become missing."""
    labels = list(dict.fromkeys(mapping.values()))
    return pd.Categorical(values.map(mapping), categories=labels, ordered=True)


def characteristics(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline characteristics as labeled categories, in display order.

    Groupings follow Morgan and Lee (2020) where they overlap, so attrition gaps can be compared
    with the 2006-2014 panels. Respondents not asked an item (split ballots) are missing.
    """
    born_abroad = df["born"] == 2
    citizenship = pd.Series(np.select(
        [df["born"] == 1, born_abroad & df["uscitzn"].isin([1, 3, 4]), born_abroad & (df["uscitzn"] == 2)],
        ["Born in the U.S.", "Citizen, born abroad", "Not a citizen"], default=None),
        index=df.index)
    race = pd.Series(np.select(
        [df["hispanic"] > 1, df["race"] == 1, df["race"] == 2, df["race"] == 3],
        ["Hispanic", "White", "Black", "Other"], default=None), index=df.index)
    race = race.where(df["hispanic"].notna() & df["race"].notna())
    # Voted in the most recent presidential election before the baseline interview
    voted = df["vote12"].where(df["baseline_year"] == 2016, df["vote16"])
    item_nonresponse = (df["n_dont_know"] + df["n_no_answer"]) / df["n_asked"]

    return pd.DataFrame({
        "Baseline interview": _categorize(df["baseline_year"], {2016: "2016", 2018: "2018"}),
        "Age": pd.cut(df["age"], [17, 29, 44, 64, 89], labels=["18-29", "30-44", "45-64", "65+"]),
        "Sex": _categorize(df["sex"], {1: "Male", 2: "Female"}),
        "Race and ethnicity": pd.Categorical(race, ["White", "Black", "Hispanic", "Other"], ordered=True),
        "Education": _categorize(df["degree"], {0: "Less than high school", 1: "High school",
                                                2: "Junior college", 3: "Bachelor's", 4: "Graduate"}),
        "Household income": pd.qcut(df["coninc"], 3, labels=["Lowest third", "Middle third", "Highest third"]),
        "Marital status": _categorize(df["marital"], {1: "Married", 2: "Widowed", 3: "Divorced",
                                                      4: "Separated", 5: "Never married"}),
        "Housing": _categorize(df["dwelown"], {1: "Owns", 2: "Rents", 3: "Other"}),
        "Citizenship": pd.Categorical(citizenship, ["Born in the U.S.", "Citizen, born abroad",
                                                    "Not a citizen"], ordered=True),
        "Region": _categorize(df["region"], {1: "Northeast", 2: "Northeast", 3: "Midwest", 4: "Midwest",
                                             5: "South", 6: "South", 7: "South", 8: "West", 9: "West"}),
        "Party identification": _categorize(df["partyid"], {0: "Democrat", 1: "Democrat",
                                                            2: "Independent", 3: "Independent",
                                                            4: "Independent", 5: "Republican",
                                                            6: "Republican", 7: "Other party"}),
        "Political views": _categorize(df["polviews"], {1: "Liberal", 2: "Liberal", 3: "Liberal",
                                                        4: "Moderate", 5: "Conservative",
                                                        6: "Conservative", 7: "Conservative"}),
        "Religious attendance": _categorize(df["attend"], {0: "Never or almost never", 1: "Never or almost never",
                                                           2: "Yearly to monthly", 3: "Yearly to monthly",
                                                           4: "Yearly to monthly", 5: "Yearly to monthly",
                                                           6: "Weekly or more", 7: "Weekly or more",
                                                           8: "Weekly or more"}),
        "Voted in last presidential election": _categorize(voted, {1: "Voted", 2: "Did not vote",
                                                                    3: "Ineligible"}),
        "Vocabulary test (words of 10)": pd.cut(df["wordsum"], [-1, 4, 6, 10], labels=["0-4", "5-6", "7-10"]),
        "Interview language": _categorize(df["spaneng"], {1: "English", 2: "Spanish"}),
        "Interview mode": _categorize(df["mode"], {1: "In person", 2: "Phone"}),
        # PHASE 2: an initial nonrespondent, kept in the random subsample for extra follow-up
        "Baseline fieldwork phase": _categorize(df["phase"], {1: "Initial", 2: "Nonresponse follow-up"}),
        # Interviewer ratings; only 6 respondents were rated hostile, so they join "restless"
        "Interviewer-rated attitude": _categorize(df["coop"], {
            1: "Friendly and interested", 2: "Cooperative", 3: "Restless or hostile", 4: "Restless or hostile"}),
        "Interviewer-rated understanding": _categorize(df["comprend"], {
            1: "Good", 2: "Fair", 3: "Poor"}),
        "Don't know / no answer rate": pd.qcut(item_nonresponse, 4, labels=[
            "Lowest quarter", "Second quarter", "Third quarter", "Highest quarter"]),
    })


def weighted_shares(y: pd.Series, group: pd.Series, weight: pd.Series, cluster: pd.Series,
                    reference: str | None = None):
    """Weighted mean of a 0/1 outcome by group, with standard errors clustered by sampling unit.

    Weighted least squares on a full set of group indicators (no constant) returns each group's
    weighted mean as a coefficient. With `reference`, the model has a constant and drops that
    group, so the coefficients are differences from it. Clustered (sandwich) standard errors are
    the Taylor-linearization standard errors used for survey means, ignoring stratification.
    """
    keep = group.notna()
    X = pd.get_dummies(group[keep], dtype=float)
    X = X.loc[:, X.sum() > 0]
    if reference is not None:
        X = sm.add_constant(X.drop(columns=reference))
    fit = sm.WLS(y[keep].astype(float), X, weights=weight[keep]).fit(
        cov_type="cluster", cov_kwds={"groups": cluster[keep]})
    return fit


def set_plot_style() -> None:
    """Minimal style shared by all figures."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Helvetica",
        "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "axes.grid": True, "axes.grid.axis": "x", "grid.color": "0.92",
        "axes.axisbelow": True,
        "xtick.color": "0.3", "ytick.color": "0.3",
        "xtick.major.size": 0, "ytick.major.size": 0,
    })


def save_figure(fig, path: Path, title: str, subtitle: str | None = None) -> None:
    """Lay out a left-aligned title and subtitle above the plot, then save."""
    height = fig.get_figheight()
    line = lambda size: size * 1.25 / 72  # line height in inches
    fig.tight_layout()

    top_pad, gap = 0.15, 0.08
    fig.text(0.02, 1 - top_pad / height, title, ha="left", va="top", fontsize=13)
    header = top_pad + line(13) * (title.count("\n") + 1)
    if subtitle:
        fig.text(0.02, 1 - (header + gap) / height, subtitle, ha="left", va="top",
                 fontsize=10, color="0.25")
        header += gap + line(10) * (subtitle.count("\n") + 1)
    fig.subplots_adjust(top=1 - (header + 0.12) / height)
    fig.savefig(path, dpi=300, facecolor="white")


# --- Attrition models ---------------------------------------------------------------------------

MODEL_LABELS = {
    "logit_literature": "Logistic regression, literature predictors",
    "logit_pca": "Logistic regression + questionnaire components",
    "penalized_logit": "Penalized logistic regression, all items",
    "random_forest": "Random forest, all items",
    "boosting": "Gradient boosting, all items",
}


def load_propensities() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "propensities.parquet")
