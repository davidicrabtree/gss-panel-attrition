"""Build every attrition weighting scheme (see weighting.py) and measure what each costs.

Cost is precision: the more weights vary, the fewer equally weighted respondents they are worth.
The R-indicator summarizes how unequal reinterview propensities were, by each model's estimate.

Outputs:
  data/processed/weights.parquet  - weights for the 1,823 reinterviewed respondents, every scheme
  output/tables/weight_summary.csv - effective sample size, weight spread, R-indicator
"""

import pandas as pd

from common import PROCESSED, TABLES, load_baseline, load_propensities
from weighting import (all_schemes, effective_sample_size, r_indicator, raking_margins,
                       scheme_label, scheme_model)

baseline = load_baseline()
propensities = load_propensities()
assert baseline["yearid"].equals(propensities["yearid"])
frame = pd.concat([baseline, propensities.filter(like="p_")], axis=1)

weights = all_schemes(frame, raking_margins(baseline))
retained = baseline["reinterviewed"] == 1
assert (weights[~retained] == 0).all().all() and (weights[retained] > 0).all().all()

out = pd.concat([baseline.loc[retained, ["yearid"]], weights[retained]], axis=1)
out.to_parquet(PROCESSED / "weights.parquet", index=False)

rows = []
for name in weights:
    w = weights.loc[retained, name]
    ratio = w / w.median()
    model = scheme_model(name)
    rows.append({
        "scheme": name, "label": scheme_label(name),
        "effective_n": effective_sample_size(w),
        "design_effect": retained.sum() / effective_sample_size(w),
        "max_over_median": ratio.max(), "min_over_median": ratio.min(),
        "r_indicator": r_indicator(frame[f"p_{model}"], baseline["base_weight"]) if model else None,
    })
summary = pd.DataFrame(rows)
summary.round(3).to_csv(TABLES / "weight_summary.csv", index=False)
print(summary.drop(columns="scheme").round(2).to_string(index=False))
