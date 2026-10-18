"""Feature sets for the attrition models, the held-out benchmark variables, and the encoder that
turns GSS items into model inputs.

Three feature sets:
  literature    - about 20 predictors modeled on Morgan and Lee (2020): demographics, vocabulary,
                  interviewer traits, and the base-year interview experience
  questionnaire - every other baseline item asked in both 2016 and 2018 (at least a quarter of each
                  cohort answered or was asked it), plus counts of don't-know and no-answer responses
  benchmarks    - 15 indicators held out of every model, so step 4 can test whether attrition
                  weights recover their baseline values. Items from the same question battery
                  are held out too, so no model sees a near-copy of a benchmark.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from common import characteristics

# --- Benchmarks (held out of every model) -----------------------------------------------------


def _voted(df):
    voted = df["vote12"].where(df["baseline_year"] == 2016, df["vote16"])
    return (voted == 1).astype(float).where(voted.notna())


def _share(column, codes):
    return lambda df: df[column].isin(codes).astype(float).where(df[column].notna())


BENCHMARKS = {
    "democrat": ("Democrat, strong or not very strong (PARTYID 0-1)", _share("partyid", [0, 1])),
    "republican": ("Republican, strong or not very strong (PARTYID 5-6)", _share("partyid", [5, 6])),
    "liberal": ("Liberal (POLVIEWS 1-3)", _share("polviews", [1, 2, 3])),
    "conservative": ("Conservative (POLVIEWS 5-7)", _share("polviews", [5, 6, 7])),
    "voted": ("Voted in last presidential election", _voted),
    "very_happy": ("Very happy (HAPPY)", _share("happy", [1])),
    "good_health": ("Health excellent or good (HEALTH)", _share("health", [1, 2])),
    "trusts_people": ("Most people can be trusted (TRUST)", _share("trust", [1])),
    "weekly_attendance": ("Attends religious services weekly or more (ATTEND)", _share("attend", [6, 7, 8])),
    "death_penalty": ("Favors death penalty for murder (CAPPUN)", _share("cappun", [1])),
    "gun_permits": ("Favors gun permits (GUNLAW)", _share("gunlaw", [1])),
    "legal_marijuana": ("Marijuana should be legal (GRASS)", _share("grass", [1])),
    "abortion_any_reason": ("Abortion for any reason (ABANY)", _share("abany", [1])),
    "confidence_science": ("Great deal of confidence in science (CONSCI)", _share("consci", [1])),
    "afraid_at_night": ("Afraid to walk alone at night (FEAR)", _share("fear", [1])),
}

HELD_OUT = {
    "partyid", "polviews", "happy", "health", "hlthstrt", "attend", "cappun", "gunlaw", "grass",
    "fear",
    # voting family
    "vote12", "vote16", "pres12", "pres16", "if12who", "if16who",
    # abortion battery
    "abany", "abdefect", "abhlth", "abnomore", "abpoor", "abrape", "absingle",
    # trust battery
    "trust", "fair", "helpful",
    # confidence battery
    "conarmy", "conbus", "conclerg", "coneduc", "confed", "confinan", "conjudge", "conlabor",
    "conlegis", "conmedic", "conpress", "consci", "contv",
}


def benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: rule(df) for name, (_, rule) in BENCHMARKS.items()})


# --- Variables never used as features ----------------------------------------------------------

EXCLUDED = {
    # design, weights, identifiers, and randomization (ballot and form are randomly assigned)
    "wtssall", "wtssnr", "wtss", "vstrat", "vpsu", "year", "id", "ballot", "form", "formwt",
    "version", "issp", "random", "sample", "sampcode", "oversamp", "hefinfo", "intid",
    # interviewer ratings of the respondent's appearance, skin tone, and health
    "ratetone", "rlooks", "rgroomed", "rweight", "rhlthend",
    # nominal codes with hundreds of values (occupation, industry, ancestry, major, language)
    "isco08", "isco88", "occ10", "indus10", "paind10", "paisco08", "paisco88", "paocc10",
    "maind10", "maisco08", "maisco88", "maocc10", "spind10", "spisco08", "spocc10", "major1",
    "ethnic", "eth1", "eth2", "eth3", "othlang1",
    # astrological sign
    "zodiac",
    # duplicates of other variables (birth year = year - age; income in other units; household
    # roster entries that repeat the household counts)
    "cohort", "income", "income16", "rincome", "rincom16", "realinc", "realrinc",
    "worda", "wordb", "wordc", "wordd", "worde", "wordf", "wordg", "wordh", "wordi", "wordj",
}
ROSTER_PREFIXES = ("mar", "gender", "old", "relate", "relhh", "relhhd", "relsp")
ROSTER = {f"{p}{i}" for p in ROSTER_PREFIXES for i in range(1, 15)}

# Baseline variables the literature set already covers (directly or through characteristics())
LITERATURE_SOURCES = {
    "age", "sex", "race", "hispanic", "degree", "coninc", "marital", "dwelown", "born", "uscitzn",
    "region", "wordsum", "intage", "intsex", "intethn", "intyrs", "lngthinv", "dateintv", "mode",
    "feeused", "coop", "comprend", "spaneng",
}


def literature(df: pd.DataFrame) -> pd.DataFrame:
    """Predictors modeled on Morgan and Lee's 21, with step 2's groupings for categories."""
    groups = characteristics(df)
    date = pd.to_datetime(
        (df["baseline_year"].astype("int64") * 10000 + df["dateintv"].astype("int64")).astype(str), format="%Y%m%d")
    return pd.DataFrame({
        "baseline_year": df["baseline_year"].astype(str),
        "age": df["age"],
        "age_squared": df["age"] ** 2,
        "sex": groups["Sex"],
        "race_ethnicity": groups["Race and ethnicity"],
        "education": groups["Education"],
        "log_income": np.log(df["coninc"]),
        "marital": groups["Marital status"],
        "housing": groups["Housing"],
        "citizenship": groups["Citizenship"],
        "region": df["region"].astype("Int64").astype(str).replace("<NA>", np.nan),
        "vocabulary": df["wordsum"],
        "interviewer_age": df["intage"],
        "interviewer_female": df["intsex"].map({1: 0.0, 2: 1.0}),
        "interviewer_race": df["intethn"].astype("Int64").astype(str).replace("<NA>", np.nan),
        "interviewer_years": df["intyrs"],
        "interview_minutes": df["lngthinv"],
        "fieldwork_day": date.dt.dayofyear,
        "phone_interview": df["mode"].map({1: 0.0, 2: 1.0}),
        "incentive": df["feeused"].map({1: "money", 2: "other", 3: "none"}),
        "attitude": groups["Interviewer-rated attitude"],
        "understanding": groups["Interviewer-rated understanding"],
        "spanish_interview": df["spaneng"].map({1: 0.0, 2: 1.0}),
    }).astype({c: object for c in ["sex", "race_ethnicity", "education", "marital", "housing",
                                   "citizenship", "attitude", "understanding"]})


def questionnaire(df: pd.DataFrame) -> pd.DataFrame:
    """All other usable baseline items, coded as in the GSS, plus item-nonresponse rates."""
    first_variable = df.columns.get_loc("n_asked") + 1
    candidates = df.columns[first_variable:]
    asked_both_years = df.groupby("baseline_year")[list(candidates)].apply(
        lambda d: d.notna().mean()).min() >= 0.25
    keep = [v for v in candidates
            if asked_both_years[v] and v not in EXCLUDED | ROSTER | HELD_OUT | LITERATURE_SOURCES
            and df[v].nunique() > 1 and pd.api.types.is_numeric_dtype(df[v])]
    items = df[keep].copy()
    items["dont_know_rate"] = df["n_dont_know"] / df["n_asked"]
    items["no_answer_rate"] = df["n_no_answer"] / df["n_asked"]
    return items


# --- Encoder -------------------------------------------------------------------------------------


class ItemEncoder(BaseEstimator, TransformerMixin):
    """Encode survey items for models that need complete numeric input.

    Items with at most `max_levels` distinct values become one indicator per value (ordinal and
    nominal alike, so no model assumes equal spacing); other items stay numeric. Every column is
    then standardized using the respondents who were asked the item, and anyone not asked gets the
    mean (0). With `missing_indicators`, one extra 0/1 column per item records that it was missing,
    because who was asked or answered can itself predict attrition.
    """

    def __init__(self, max_levels: int = 10, missing_indicators: bool = True):
        self.max_levels = max_levels
        self.missing_indicators = missing_indicators

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.levels_ = {c: sorted(X[c].dropna().unique()) for c in X.columns
                        if X[c].nunique() <= self.max_levels
                        or not pd.api.types.is_numeric_dtype(X[c])}
        self.with_missing_ = [c for c in X.columns if X[c].isna().any()]
        expanded = self._expand(X)
        self.mean_ = expanded.mean()
        self.scale_ = expanded.std().replace(0, 1).fillna(1)
        self.feature_names_ = list(expanded.columns) + (
            [f"{c}__missing" for c in self.with_missing_] if self.missing_indicators else [])
        return self

    def _expand(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for c in X.columns:
            if c in self.levels_:
                values = X[c]
                dummies = pd.DataFrame(
                    {f"{c}={level}": (values == level).astype(float) for level in self.levels_[c]},
                    index=X.index)
                parts.append(dummies.where(values.notna()))
            else:
                parts.append(X[[c]].astype(float))
        return pd.concat(parts, axis=1)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        encoded = ((self._expand(X) - self.mean_) / self.scale_).fillna(0.0)
        if self.missing_indicators:
            flags = X[self.with_missing_].isna().astype(float).add_suffix("__missing")
            encoded = pd.concat([encoded, flags], axis=1)
        return encoded.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)
