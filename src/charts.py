"""README charts from src/compare.py results.

Usage: python src/charts.py
Output: reports/chart_caught.png, reports/chart_pr_curve.png
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# Reference palette (dataviz skill), categorical slots 1-4 in fixed order, light surface.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ALARMS = [("rule", "One-line rule", "#2a78d6"), ("logistic", "Simple model", "#eb6834"),
          ("old", "Old alarm", "#1baf7a"), ("new", "New alarm", "#eda100")]
TESTS = [("as_simulated", "Thieves take\neverything"), ("partial", "Take 60-95%,\nchain kept"),
         ("partial_split", "Take 60-95%,\nchain broken"),
         ("partial_split_funded_mule", "Same, and mule\nalready has money")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                     "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def caught_chart(summary):
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=150)
    width, gap = 0.19, 0.02
    x = np.arange(len(TESTS))
    for i, (key, label, color) in enumerate(ALARMS):
        rows = summary[summary.alarm == key].set_index("test").loc[[t for t, _ in TESTS]]
        pos = x + (i - 1.5) * (width + gap)
        mean = rows.recall_mean.values * 100
        err = np.vstack([mean - rows.recall_min.values * 100, rows.recall_max.values * 100 - mean])
        ax.bar(pos, mean, width, color=color, label=label, zorder=2)
        ax.errorbar(pos, mean, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2, zorder=3)
        for p, m, top in zip(pos, mean, rows.recall_max.values * 100):
            ax.text(p, max(m, top) + 2, f"{m:.0f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, [t for _, t in TESTS])
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_ylabel("Thefts caught with 300 reviews a day")
    ax.set_title("Only the new alarm keeps catching thieves who change their habits", loc="left",
                 fontsize=13, fontweight="bold", pad=28)
    ax.text(0, 1.03, "Last 10 days of PaySim. Bars: average of 5 runs; lines: lowest to highest run.",
            transform=ax.transAxes, color=INK_2, fontsize=9.5)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9.5)
    fig.tight_layout()
    fig.savefig(REPORTS / "chart_caught.png")
    plt.close(fig)


def pr_chart(scores):
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    for key, label, color in ALARMS:
        s = scores[scores.alarm == key]
        precision, recall, _ = precision_recall_curve(s.is_fraud, s.score)
        ax.plot(recall * 100, precision * 100, color=color, linewidth=2, label=label)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Share of thefts caught")
    ax.set_ylabel("Share of alerts that are real thefts")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Hardest test: catching more thefts without drowning in false alarms", loc="left",
                 fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.025, "Thieves take 60-95%, break the chain and use a funded mule (run 1 of 5).",
            transform=ax.transAxes, color=INK_2, fontsize=9.5)
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(REPORTS / "chart_pr_curve.png")
    plt.close(fig)


if __name__ == "__main__":
    caught_chart(pd.read_csv(REPORTS / "compare_summary.csv"))
    pr_chart(pd.read_parquet(REPORTS / "pr_scores_seed0.parquet"))
    print("saved chart_caught.png and chart_pr_curve.png")
