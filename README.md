# Who Leaves a Survey Panel? Machine-Learning Attrition Weights for the 2016–2020 General Social Survey Panel

In 2020 the General Social Survey (GSS), run by NORC at the University of Chicago, tried to reinterview people it had first interviewed in 2016 or 2018. Only 42% of eligible respondents took part. NORC's released panel weight adjusts for that loss only by geographic area, and the model-based attrition weights promised in the panel codebook are not in the current release. This project builds them, comparing logistic regression with penalized regression, random forests, and gradient boosting, and tests them against 15 baseline measures held out of every model. That test is possible because people who left the panel still have their baseline answers in the data.

**Read the write-up:** [report.html](report.html) · [report.pdf](report.pdf)

## Findings

**Weights from a model that uses the whole questionnaire cut the attrition bias roughly in half; NORC's released weight cuts it by about a tenth. The models that best predicted who would leave were not the ones that removed the most bias.**

1. The panel kept more voters, more trusting people, and more liberals. At baseline, those who stayed were 7.0 points more likely than all eligible respondents to say most people can be trusted and 6.7 points more likely to have voted. Dropout tracked education far more steeply than in the GSS's 2006–2014 panels (graduate degree vs. high school: 15.3 points less likely to drop out, against 3.9 earlier).
2. Across the 15 held-out benchmarks, NORC's weight closes 11% of the gap (not distinguishable from zero), raking on demographics 29%, and inverse-probability weights from the five attrition models 37–53%. The best, a penalized logistic regression on the full questionnaire, beats raking by 0.67 points of average gap (95% interval 0.14 to 0.95) at a small cost in effective sample size.
3. Random forests and boosting ranked dropout best (AUC 0.62 vs. 0.60), but models that see the respondents' attitudes, either directly or through principal components of the questionnaire, removed more bias. Weights work when the attrition model captures what drives the outcomes, so predictive accuracy is the wrong yardstick for choosing one.
4. Better weights lower 2020 estimates of trust in people (by 3.6 points), confidence in science (3.1), and turnout (2.6) relative to NORC's weight, but leave within-person change from baseline to 2020 essentially unchanged.

## Project structure

```
python/
  00_download_data.py   Download the panel file from NORC (checked against a SHA-256 hash)
  01_prepare_data.py    Build the analysis files and check them against the codebook
  02_who_left.py        Reinterview rates by baseline characteristics; comparison with 2006-2014
  03_models.py          Fit five attrition models with nested cross-validation (15-30 minutes)
  04_compare_models.py  Discrimination, calibration, and cohort transfer of the five models
  05_weights.py         Build 19 weighting schemes; effective sample sizes and R-indicators
  06_bias_test.py       Benchmark bias test with a cluster bootstrap (5-10 minutes)
  07_estimates_2020.py  2020 estimates and baseline-to-2020 change under each weighting scheme
  weighting.py          Weighting schemes: NORC, raking, inverse probability, propensity classes
  features.py           Feature sets, held-out benchmark variables, and the item encoder
  common.py             Paths, baseline characteristics, weighted shares, figure style
output/                 Tables and figures
report.qmd              Write-up (Quarto)
```

## Reproducing the analysis

The data are the public 2016–2020 GSS Panel (Release 1a, April 2022), free from NORC with no registration. The download script fetches them; they are not stored in this repository.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run_all.sh
```

The full run takes 25-45 minutes on an 8-core laptop, most of it nested cross-validation.

## References

Davern, M., Bautista, R., Freese, J., Morgan, S. L., and Smith, T. W. (2021). General Social Survey 2016–2020 Panel, Release 1a [data file and codebook]. NORC at the University of Chicago. <https://gss.norc.org>

Frankel, L. L., and Hillygus, D. S. (2014). Looking beyond demographics: Panel attrition in the ANES and GSS. *Political Analysis*, 22(3), 336–353.

Little, R. J. A., and Vartivarian, S. (2005). Does weighting for nonresponse increase the variance of survey means? *Survey Methodology*, 31(2), 161–168.

Morgan, S. L., and Lee, J. (2020). Six Alternative Weights that Adjust for Attrition in the 2006–2014 General Social Survey Panels. GSS Methodological Report No. 132. NORC at the University of Chicago.
