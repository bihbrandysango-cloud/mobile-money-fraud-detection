"""Honest baseline: train on the past, test on the future, judge by cost.

Split (by time, never at random):
  train  days 1-18  (steps 1-432)
  valid  days 19-21 (steps 433-504)  - used to stop training and pick the threshold
  test   days 22-31 (steps 505-743)  - touched once, at the end

Three feature sets, to show where the "accuracy" comes from:
  strict    - only what the operator knows about the sender and the request
  realistic - strict + receiver's balance and history before this transaction
  leaky     - realistic + balances recorded AFTER the transaction

Only TRANSFER and CASH_OUT are scored: PaySim has no fraud in other types.

Cost of a decision:
  missed fraud = the stolen amount
  alert        = REVIEW_COST (an agent's call to the customer); assumption, see notes

Usage: python src/baseline.py
Output: reports/baseline_results.csv, reports/test_scores_<set>.parquet
"""
import pathlib

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = (ROOT / "data" / "paysim.parquet").as_posix()
REPORTS = ROOT / "reports"
TRAIN_END, VALID_END = 432, 504
REVIEW_COST = 1_000  # same currency units as `amount`

STRICT = ["is_transfer", "amount", "log_amount", "orig_bal_before", "amount_to_balance",
          "empties_account", "hour_of_day"]
REALISTIC = STRICT + ["dest_bal_before", "dest_prior_in_count", "dest_prior_in_amount",
                      "dest_prior_out_count", "dest_hours_since_last"]
LEAKY = REALISTIC + ["orig_bal_after", "dest_bal_after", "orig_balance_error", "dest_balance_error"]
FEATURE_SETS = {"strict": STRICT, "realistic": REALISTIC, "leaky": LEAKY}


def load():
    # Receiver history uses only earlier hours (step < current step), so nothing
    # from the future leaks in. Fraud labels are never used as features.
    return duckdb.sql(f"""
        WITH t AS (SELECT * FROM '{DATA}'),
        incoming AS (
            SELECT txn_id,
                count(*) OVER w - 1 AS same_hour_and_before,
                count(*) OVER (PARTITION BY dest ORDER BY step RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                    AS dest_prior_in_count,
                coalesce(sum(amount) OVER (PARTITION BY dest ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS dest_prior_in_amount,
                step - max(step) OVER (PARTITION BY dest ORDER BY step
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS dest_hours_since_last
            FROM t WINDOW w AS (PARTITION BY dest ORDER BY step)
        ),
        outgoing AS (  -- has the receiver itself sent money before?
            SELECT b.txn_id, count(a.txn_id) AS dest_prior_out_count
            FROM t b LEFT JOIN t a ON a.orig = b.dest AND a.step < b.step
            WHERE b.type IN ('TRANSFER', 'CASH_OUT')
            GROUP BY b.txn_id
        )
        SELECT t.txn_id, t.step, t.type, t.is_fraud,
            (t.type = 'TRANSFER')::INT AS is_transfer,
            t.amount, ln(1 + t.amount) AS log_amount, t.orig_bal_before,
            coalesce(t.amount_to_balance, -1) AS amount_to_balance,
            t.empties_account::INT AS empties_account, t.hour_of_day,
            t.dest_bal_before, i.dest_prior_in_count, i.dest_prior_in_amount,
            o.dest_prior_out_count, coalesce(i.dest_hours_since_last, -1) AS dest_hours_since_last,
            t.orig_bal_after, t.dest_bal_after,
            t.orig_bal_before - t.amount - t.orig_bal_after AS orig_balance_error,
            t.dest_bal_before + t.amount - t.dest_bal_after AS dest_balance_error
        FROM t JOIN incoming i USING (txn_id) JOIN outgoing o USING (txn_id)
        WHERE t.type IN ('TRANSFER', 'CASH_OUT')
    """).df()


def best_threshold(y, score, amount):
    """Threshold with the lowest total cost on the validation days."""
    best = (np.inf, 0.5)
    for th in np.unique(np.quantile(score, np.linspace(0.5, 1, 400))):
        flag = score >= th
        cost = amount[(y == 1) & ~flag].sum() + REVIEW_COST * flag.sum()
        best = min(best, (cost, th))
    return best[1]


def evaluate(name, y, score, amount, th, days):
    flag = score >= th
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fraud_amount = amount[y == 1].sum()
    missed_amount = amount[(y == 1) & ~flag].sum()
    return {
        "features": name,
        "pr_auc": round(average_precision_score(y, score), 3),
        "threshold": round(float(th), 4),
        "alerts": tp + fp,
        "alerts_per_day": round((tp + fp) / days),
        "fraud_caught": tp,
        "fraud_total": int(y.sum()),
        "recall": round(tp / y.sum(), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "innocent_flagged": fp,
        "share_of_stolen_money_stopped": round(1 - missed_amount / fraud_amount, 3),
        "total_cost": round(missed_amount + REVIEW_COST * (tp + fp)),
        "cost_if_no_model": round(fraud_amount),
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    d = load()
    train, valid, test = d[d.step <= TRAIN_END], d[(d.step > TRAIN_END) & (d.step <= VALID_END)], d[d.step > VALID_END]
    print(f"train {len(train):,} ({train.is_fraud.sum()} fraud) | valid {len(valid):,} "
          f"({valid.is_fraud.sum()}) | test {len(test):,} ({test.is_fraud.sum()})")
    test_days = (test.step.max() - test.step.min() + 1) / 24

    rows = []
    for name, feats in FEATURE_SETS.items():
        model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                                   min_child_samples=50, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, verbose=-1, random_state=0)
        model.fit(train[feats], train.is_fraud, eval_set=[(valid[feats], valid.is_fraud)],
                  eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
        th = best_threshold(valid.is_fraud.values, model.predict_proba(valid[feats])[:, 1], valid.amount.values)
        score = model.predict_proba(test[feats])[:, 1]
        rows.append(evaluate(name, test.is_fraud.values, score, test.amount.values, th, test_days))
        test[["txn_id", "step", "type", "amount", "is_fraud"]].assign(score=score).to_parquet(
            REPORTS / f"test_scores_{name}.parquet", index=False)
        model.booster_.save_model(str(REPORTS / f"model_{name}.txt"))

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "baseline_results.csv", index=False)
    print(res.T.to_string())


if __name__ == "__main__":
    main()
