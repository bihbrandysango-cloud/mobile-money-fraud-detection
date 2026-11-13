"""Run the whole project in order, from download to browser export and tests.

Usage:
  python run_all.py                 # every step (about 40-60 minutes)
  python run_all.py --from robust   # start at a step, e.g. after changing the model
  python run_all.py --list          # show the steps

Stops at the first step that fails.
"""
import argparse
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent

STEPS = [
    ("download", "src/download_data.py", "Get PaySim from Kaggle"),
    ("prepare", "src/prepare.py", "CSV to Parquet, request-time features"),
    ("flawed", "src/flawed_replication.py", "Reproduce the '99% accuracy'"),
    ("baseline", "src/baseline.py", "Time split, three feature sets"),
    ("robust", "src/robust.py", "New alarm, four tests, app files"),
    ("compare", "src/compare.py", "Four alarms over five seeds, money"),
    ("charts", "src/charts.py", "README charts"),
    ("explain", "src/explain.py", "SHAP summary chart"),
    ("false_alarms", "src/false_alarms.py", "Who is wrongly flagged, calibration chart"),
    ("export", "src/export_js.py", "Browser version of the alarm"),
    ("test", "-m pytest -q tests", "Stories, browser parity, export up to date"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=[s[0] for s in STEPS], default="download")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for name, _, what in STEPS:
            print(f"  {name:9} {what}")
        return

    first = [s[0] for s in STEPS].index(args.start)
    for name, target, what in STEPS[first:]:
        print(f"\n=== {name}: {what}", flush=True)
        started = time.time()
        cmd = [sys.executable, "-W", "ignore", *target.split()]
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            sys.exit(f"Step '{name}' failed. Fix it, then run: python run_all.py --from {name}")
        print(f"--- {name} done in {time.time() - started:.0f}s", flush=True)
    print("\nAll steps finished.")


if __name__ == "__main__":
    main()
