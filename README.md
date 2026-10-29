# Machine-Learning Attrition Weights for the 2016–2020 General Social Survey Panel

In 2020 the General Social Survey (GSS), run by NORC at the University of Chicago, tried to reinterview people it had first interviewed in 2016 or 2018. Only 42% of eligible respondents took part. NORC's released panel weight adjusts for that loss only by geographic area. In this project, I evaluate various methods for generating attrition weights and compare them to the weights NORC provided.

I compare attrition weights from the standard logistic regression used in earlier GSS work to machine-learning methods: penalized regression, random forests, and gradient boosting. I find penalized logistic regression produces weights that remove the most bias. Because the people who left the panel still have their full baseline interview in the data, the right answer is known for any question asked before anyone left, so I can check whether each set of weights makes the people who stayed look like everyone who started. Panel studies have long used this kind of check (Fitzgerald, Gottschalk, and Moffitt 1998); to my knowledge it has not been applied to this panel's weights. These findings stand to benefit GSS users whose questions involve answers that were affected by the panel's attrition patterns.

**Read the write-up:** [report.html](https://davidicrabtree.github.io/gss-panel-attrition/report.html) · [report.pdf](https://davidicrabtree.github.io/gss-panel-attrition/report.pdf)

## Findings

**Key Finding: Weights from a machine learning model that uses the whole questionnaire cut the attrition bias roughly in half; NORC's released weight cuts it by about a tenth.**

1. **Model-based weights beat both NORC's weight and demographic raking.** Across 15 benchmark measures held out of every model, NORC's weight closes 11% of the gap, raking on demographics 29%, and weights from the five attrition models 37%–53%.
2. **The panel kept more voters, more trusting people, and more liberals.** At baseline, the people who stayed were 7.0 points more likely than all eligible respondents to say most people can be trusted, 6.7 points more likely to have voted, and 3.9 points more likely to call themselves liberal. Dropout tracked education much more steeply than in the GSS's 2006–2014 panels.
3. **The best predictor is not the best weight.** Random forests and gradient boosting ranked dropout best but removed less bias than penalized logistic regression. Choosing an attrition model by predictive accuracy alone would pick the wrong one.
4. **Better weights change 2020 estimates but not estimates of change.** Relative to NORC's weight, they lower the 2020 share who say most people can be trusted by 3.6 points and the share with a great deal of confidence in science by 3.1 points. Within-person change from baseline to 2020 barely moves under any weighting.

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

Fitzgerald, J., Gottschalk, P., and Moffitt, R. (1998). An analysis of sample attrition in panel data: The Michigan Panel Study of Income Dynamics. *Journal of Human Resources*, 33(2), 251–299.

Frankel, L. L., and Hillygus, D. S. (2014). Looking beyond demographics: Panel attrition in the ANES and GSS. *Political Analysis*, 22(3), 336–353.

Little, R. J. A., and Vartivarian, S. (2005). Does weighting for nonresponse increase the variance of survey means? *Survey Methodology*, 31(2), 161–168.

Mercer, A., Lau, A., and Kennedy, C. (2018). For Weighting Online Opt-In Samples, What Matters Most? Pew Research Center. <https://www.pewresearch.org/methods/2018/01/26/for-weighting-online-opt-in-samples-what-matters-most/>

Morgan, S. L., and Lee, J. (2020). Six Alternative Weights that Adjust for Attrition in the 2006–2014 General Social Survey Panels. GSS Methodological Report No. 132. NORC at the University of Chicago.
