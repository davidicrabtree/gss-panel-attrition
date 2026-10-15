"""Shared paths and constants for the GSS panel attrition analysis."""

from pathlib import Path

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
