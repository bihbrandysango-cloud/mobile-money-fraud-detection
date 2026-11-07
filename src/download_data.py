"""Download the PaySim data from Kaggle's public link and unzip it into data/raw/.

No Kaggle account is needed. The zip is 186 MB; an interrupted download
resumes where it stopped. Skips everything if the CSV is already there.

Usage: python src/download_data.py
"""
import pathlib
import sys
import time
import zipfile

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
URL = "https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1"
ZIP = RAW / "paysim_archive.zip"
EXPECTED_ROWS = 6_362_620


def download(tries=8):
    for attempt in range(1, tries + 1):
        have = ZIP.stat().st_size if ZIP.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(URL, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:  # already complete
                    return
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored the range: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length", 0))
                with open(ZIP, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        print(f"\r  {have / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            print()
            if ZIP.stat().st_size >= total:
                return
        except requests.RequestException as e:
            print(f"\n  attempt {attempt} failed ({e.__class__.__name__}), retrying")
        time.sleep(10)
    sys.exit("Download failed. Check the internet connection and run again; it resumes.")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if list(RAW.glob("PS_*.csv")):
        print("PaySim CSV already in data/raw, nothing to do")
        return
    print("Downloading PaySim (186 MB) from Kaggle")
    download()
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        z.extract(names[0], RAW)
    csv = RAW / names[0]
    with open(csv, "rb") as f:
        rows = sum(1 for _ in f) - 1
    if rows != EXPECTED_ROWS:
        sys.exit(f"{csv.name} has {rows:,} rows, expected {EXPECTED_ROWS:,}")
    ZIP.unlink()
    print(f"Saved {csv.relative_to(ROOT)} ({rows:,} rows)")


if __name__ == "__main__":
    main()
