"""Attrition weighting schemes.

Every scheme starts from the baseline weight d (WTSSNR times the 2016 subsampling factor), which
makes the eligible sample represent the 2016 and 2018 cross-sections. Each function takes the
eligible sample (all 4,351, or a bootstrap resample of it) and returns a weight for every row:
the attrition-adjusted weight for reinterviewed respondents and 0 for everyone else.

  base      d alone: no attrition adjustment
  norc      NORC's WTSSNR_2 (area-level response-rate adjustment), rescaled so each cohort's
            share of the total matches its share under d. NORC mixes the cohorts 50/50; the
            target here is the pooled cross-sections, and this keeps that difference from
            counting against NORC's weight.
  raking    d adjusted by iterative proportional fitting until reinterviewed respondents match
            the whole eligible sample on age, sex, race and ethnicity, education, region, and
            baseline year
  ipw       d / p, with p the out-of-fold reinterview probability from a model
  ipw_trim  the same, with the adjustment 1 / p capped at its 99th percentile among the
            reinterviewed
  classes   d / r, with r the weighted reinterview rate in the respondent's fifth of the
            predicted probabilities (the weighting-class method)
"""

import numpy as np
import pandas as pd

from common import MODEL_LABELS, characteristics

RAKING_MARGINS = ["Age", "Sex", "Race and ethnicity", "Education", "Region", "Baseline interview"]


def raking_margins(baseline: pd.DataFrame) -> pd.DataFrame:
    """Raking categories; the few missing values take the most common category."""
    groups = characteristics(baseline)[RAKING_MARGINS].astype(object)
    return groups.fillna(groups.mode().iloc[0])


def rake(margins: pd.DataFrame, base: pd.Series, retained: pd.Series,
         tol: float = 1e-8, max_iter: int = 200) -> pd.Series:
    """Iterative proportional fitting: scale reinterviewed respondents' weights one margin at a
    time until their weighted totals match the eligible sample's on every margin."""
    targets = {m: base.groupby(margins[m]).sum() for m in margins}
    w = base.where(retained, 0.0)
    for _ in range(max_iter):
        for m in margins:
            current = w.groupby(margins[m]).sum()
            w = w * margins[m].map(targets[m] / current).astype(float)
        gap = max((w.groupby(margins[m]).sum() / targets[m] - 1).abs().max() for m in margins)
        if gap < tol:
            return w
    raise RuntimeError(f"Raking did not converge (largest margin gap {gap:.2e})")


def norc(frame: pd.DataFrame, base: pd.Series) -> pd.Series:
    w = frame["wtssnr_2"].fillna(0.0)
    cohort = frame["baseline_year"]
    share_target = base.groupby(cohort).sum() / base.sum()
    share_norc = w.groupby(cohort).sum() / w.sum()
    return w * cohort.map(share_target / share_norc)


def ipw(p: pd.Series, base: pd.Series, retained: pd.Series, trim: bool = False) -> pd.Series:
    adjustment = 1 / p
    if trim:
        adjustment = adjustment.clip(upper=adjustment[retained].quantile(0.99))
    return (base * adjustment).where(retained, 0.0)


def classes(p: pd.Series, base: pd.Series, retained: pd.Series, n_classes: int = 5) -> pd.Series:
    group = pd.qcut(p, n_classes, labels=False)
    rate = base.where(retained, 0.0).groupby(group).sum() / base.groupby(group).sum()
    return (base / group.map(rate)).where(retained, 0.0)


def all_schemes(frame: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    """Every scheme's weights for one sample. `frame` holds the baseline file's columns plus the
    model probabilities (p_<model>); `margins` holds the raking categories for the same rows."""
    base = frame["base_weight"]
    retained = frame["reinterviewed"] == 1
    schemes = {
        "unweighted": retained.astype(float),
        "base": base.where(retained, 0.0),
        "norc": norc(frame, base),
        "raking": rake(margins, base, retained),
    }
    for model in MODEL_LABELS:
        p = frame[f"p_{model}"]
        schemes[f"ipw_{model}"] = ipw(p, base, retained)
        schemes[f"ipw_trim_{model}"] = ipw(p, base, retained, trim=True)
        schemes[f"classes_{model}"] = classes(p, base, retained)
    return pd.DataFrame(schemes)


def effective_sample_size(w: pd.Series) -> float:
    """Kish's effective sample size: the number of equally weighted respondents that would give
    the same precision for a mean."""
    w = w[w > 0]
    return w.sum() ** 2 / (w ** 2).sum()


def r_indicator(p: pd.Series, base: pd.Series) -> float:
    """Representativity indicator (Schouten, Cobben, and Bethlehem 2009): 1 minus twice the
    weighted standard deviation of response propensities. 1 means everyone was equally likely
    to respond. It depends on the propensity model: a model that finds no differences reports
    perfect representativeness."""
    mean = np.average(p, weights=base)
    return 1 - 2 * np.sqrt(np.average((p - mean) ** 2, weights=base))


SHORT_MODEL_NAMES = {
    "logit_literature": "literature logit",
    "logit_pca": "logit + components",
    "penalized_logit": "lasso/ridge",
    "random_forest": "random forest",
    "boosting": "boosting",
}
METHOD_NAMES = {"ipw": "Inverse probability", "ipw_trim": "Inverse probability, trimmed",
                "classes": "Propensity classes"}
FIXED_LABELS = {"unweighted": "Unweighted", "base": "Baseline weight only",
                "norc": "NORC panel weight (WTSSNR_2)", "raking": "Raking on demographics"}


def scheme_model(name: str) -> str | None:
    return next((m for m in SHORT_MODEL_NAMES if name.endswith(m)), None)


def scheme_method(name: str) -> str:
    """'Raking on demographics', 'Propensity classes', ...: the scheme without its model."""
    model = scheme_model(name)
    return METHOD_NAMES[name.removesuffix("_" + model)] if model else FIXED_LABELS[name]


def scheme_label(name: str) -> str:
    model = scheme_model(name)
    return f"{scheme_method(name)}, {SHORT_MODEL_NAMES[model]}" if model else FIXED_LABELS[name]
