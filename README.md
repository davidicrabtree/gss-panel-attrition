# Who Leaves a Survey Panel? Modeling Attrition in the 2016–2020 General Social Survey Panel

*Work in progress.*

In 2020, the General Social Survey (GSS) tried to reinterview people it had first interviewed in 2016 or 2018. Only 42% of eligible respondents took part. NORC's released panel weight adjusts for that attrition only by geographic area, and the person-level, model-based attrition weights promised in the panel codebook are not in the current release. This project compares logistic regression with machine-learning models for predicting who left, builds attrition weights from each, and tests whether they remove bias, using the fact that the baseline answers of people who left are still in the data.

## Project structure

```
python/
  00_download_data.py   Download the panel file from NORC (checked against a SHA-256 hash)
  01_prepare_data.py    Build the analysis files and check them against the codebook
  02_who_left.py        Reinterview rates by baseline characteristics; comparison with 2006-2014
  03_models.py          Fit five attrition models with nested cross-validation (about 15 minutes)
  04_compare_models.py  Discrimination, calibration, and cohort transfer of the five models
  features.py           Feature sets, held-out benchmark variables, and the item encoder
  common.py             Paths, baseline characteristics, weighted shares, figure style
output/                 Tables and figures
```

## Reproducing the analysis

The data are the public 2016–2020 GSS Panel (Release 1a, April 2022), free from NORC with no registration. The download script fetches them; they are not stored in this repository.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run_all.sh
```

## References

Davern, M., Bautista, R., Freese, J., Morgan, S. L., and Smith, T. W. (2021). General Social Survey 2016–2020 Panel, Release 1a [data file and codebook]. NORC at the University of Chicago. <https://gss.norc.org>

Morgan, S. L., and Lee, J. (2020). Six Alternative Weights that Adjust for Attrition in the 2006–2014 General Social Survey Panels. GSS Methodological Report No. 132. NORC.
