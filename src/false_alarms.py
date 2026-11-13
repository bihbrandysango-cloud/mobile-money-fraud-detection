"""Who the deployed alarm wrongly flags, and whether its risk scores can be trusted.

Uses the model behind the website (reports/model_robust.txt) on days 22-31,
in two versions of the test: thefts as simulated, and the hardest case
(thieves take 60-95%, amounts don't match, mule already holds money; seed 0
as in src/compare.py).

1. False alarms within a 300-reviews-a-day budget: how many, and what the
   wrongly flagged honest transactions look like.
2. Calibration: group transactions by the score the alarm gave them and
   compare with how often they really were thefts.

Usage: python src/false_alarms.py
Output: reports/false_alarms.csv, reports/calibration.csv, reports/chart_calibration.png
"""
import pathlib

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline import VALID_END, load
from compare import make_tests
from robust import REVIEWS_PER_DAY, ROBUST, add_pair_match

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BINS = [0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0001]
LABELS = ["under 1%", "1-5%", "5-20%", "20-50%", "50-80%", "80-95%", "95% and over"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def describe_false_alarms(name, t, score):
    days = (t.step.max() - t.step.min() + 1) / 24
    reviews = int(REVIEWS_PER_DAY * days)
    top = t.iloc[np.argsort(-score)[:reviews]]
    fp = top[top.is_fraud == 0]
    return {
        "test": name, "reviews": reviews, "thefts_caught": int(top.is_fraud.sum()),
        "honest_flagged": len(fp), "honest_share": round(len(fp) / reviews, 3),
        "honest_flagged_per_day": round(len(fp) / days),
        "fp_transfers": round((fp.is_transfer == 1).mean(), 3),
        "fp_send_90pct_or_more": round((fp.amount_to_balance >= 0.9).mean(), 3),
        "fp_night_0_to_5h": round((fp.hour_of_day < 6).mean(), 3),
        "fp_new_receiver": round((fp.dest_prior_in_count == 0).mean(), 3),
        "fp_median_amount": round(fp.amount.median()),
        "honest_median_amount_all": round(t[t.is_fraud == 0].amount.median()),
    }


def calibration(name, t, score):
    d = pd.DataFrame({"score": score, "is_fraud": t.is_fraud.values})
    d["band"] = pd.cut(d.score, BINS, labels=LABELS, right=False)
    c = d.groupby("band", observed=False).agg(transactions=("score", "size"), mean_score=("score", "mean"),
                                              theft_rate=("is_fraud", "mean")).reset_index()
    return c.assign(test=name)


def chart(cal):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, ax = plt.subplots(figsize=(7.5, 5.4), dpi=150)
    ax.plot([0, 100], [0, 100], color=INK_2, linewidth=1, linestyle="--", label="Perfectly honest scores")
    for (name, label, color) in [("as_simulated", "Thieves as simulated", "#2a78d6"),
                                 ("hardest", "Smarter thieves (hardest test)", "#eb6834")]:
        c = cal[(cal.test == name) & (cal.transactions >= 30)]
        ax.plot(c.mean_score * 100, c.theft_rate * 100, color=color, linewidth=2, marker="o",
                markersize=6, label=label)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Risk score the alarm gave")
    ax.set_ylabel("Share that really were thefts")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("How far can you trust the risk score?", loc="left", fontsize=12.5, fontweight="bold", pad=24)
    ax.text(0, 1.025, "Days 22-31, bands with at least 30 transactions. On the dashed line, a 70% score means 70% are thefts.",
            transform=ax.transAxes, color=INK_2, fontsize=8.8)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(REPORTS / "chart_calibration.png")
    plt.close(fig)


def main():
    booster = lgb.Booster(model_file=str(REPORTS / "model_robust.txt"))
    d = add_pair_match(load())
    test = d[d.step > VALID_END]
    tests = {"as_simulated": test,
             "hardest": make_tests(test, np.random.default_rng(100))["partial_split_funded_mule"]}
    rows, cals = [], []
    for name, t in tests.items():
        score = booster.predict(t[ROBUST])
        rows.append(describe_false_alarms(name, t, score))
        cals.append(calibration(name, t, score))
    fa = pd.DataFrame(rows)
    cal = pd.concat(cals)
    fa.to_csv(REPORTS / "false_alarms.csv", index=False)
    cal.to_csv(REPORTS / "calibration.csv", index=False)
    chart(cal)
    pd.set_option("display.width", 200)
    print(fa.T.to_string())
    print(cal.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
