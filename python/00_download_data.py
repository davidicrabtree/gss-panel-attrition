"""Download the 2016-2020 GSS Panel (Release 1a) from NORC and unpack the Stata file.

The GSS public-use files are free and need no registration. The archive is checked against
the SHA-256 hash of the copy used for this analysis, so a silent re-release by NORC would
stop the pipeline instead of quietly changing the results.

Output: data/raw/gss2020panel_r1a.dta (the archive's codebook PDF is unpacked alongside it)
"""

import hashlib
import urllib.request
import zipfile

from common import DATA_FILE, DATA_URL, DATA_ZIP_SHA256, RAW

if DATA_FILE.exists():
    print(f"{DATA_FILE.name} already present; skipping download")
    raise SystemExit

RAW.mkdir(parents=True, exist_ok=True)
archive = RAW / DATA_URL.rsplit("/", 1)[-1]
print(f"Downloading {DATA_URL}")
urllib.request.urlretrieve(DATA_URL, archive)

digest = hashlib.sha256(archive.read_bytes()).hexdigest()
if digest != DATA_ZIP_SHA256:
    raise SystemExit(f"Unexpected SHA-256 for {archive.name}: {digest}\n"
                     "NORC may have released a new version; check it before updating the hash.")

with zipfile.ZipFile(archive) as z:
    z.extractall(RAW)
archive.unlink()
print(f"Saved {DATA_FILE.relative_to(RAW.parents[1])}")
