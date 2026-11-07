"""Chart: which clues the robust alarm uses, and in which direction.

Uses days 22-31 (test days): every fraud plus 20,000 honest transactions.

Usage: python src/explain.py
Output: reports/shap_summary.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from baseline import VALID_END, load
from robust import add_pair_match
from scoring import FEATURES, REPORTS, _explainer

LABELS = {
    "is_transfer": "Transfer (vs cash-out)", "amount": "Amount", "log_amount": "Amount (log)",
    "orig_bal_before": "Sender's balance", "amount_to_balance": "Share of balance sent",
    "empties_account": "Takes whole balance", "hour_of_day": "Hour of day",
    "dest_bal_before": "Receiver's balance", "dest_prior_in_count": "Receiver: payments received before",
    "dest_prior_in_amount": "Receiver: money received before", "dest_prior_out_count": "Receiver: payments sent before",
    "dest_hours_since_last": "Receiver: hours since last payment", "pair_match": "Money-chain match",
}


def main():
    d = add_pair_match(load())
    test = d[d.step > VALID_END]
    sample = pd.concat([test[test.is_fraud == 1], test[test.is_fraud == 0].sample(20_000, random_state=0)])
    X = sample[FEATURES]
    values = _explainer.shap_values(X)
    shap.summary_plot(values, X.rename(columns=LABELS), show=False, max_display=13, plot_size=(9, 6))
    plt.title("What pushes the fraud alarm up (right) or down (left)\nEach dot is one transaction, colour = clue value",
              fontsize=11)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=150)
    print(f"saved {REPORTS / 'shap_summary.png'}")


if __name__ == "__main__":
    main()
