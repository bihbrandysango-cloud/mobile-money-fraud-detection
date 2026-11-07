"""A fraud model that does not fall apart when fraudsters change habits.

The baseline learned one shortcut: PaySim fraud sends the whole balance.
Here we
  1. add a "money chain" feature: an opposite-type transaction (TRANSFER vs
     CASH_OUT) of the exact same amount in the past hour, the mule pattern;
  2. train on extra copies of the fraud cases in which the fraudster takes only
     50-95% of the balance, so the model cannot lean on "whole balance";
  3. test both models on three versions of days 22-31:
       as_simulated  - fraud as PaySim made it
       partial       - every fraud pair takes 60-95% of the balance
       partial_split - as partial, and the two halves of a pair use different
                       amounts, so the money-chain clue disappears too
       partial_split_funded_mule - as partial_split, and the receiving
                       account already holds money (PaySim mules start at zero)

Score: an operator can review a fixed number of alerts a day. We report how
much fraud (count and money) lands inside that daily review budget.

Usage: python src/robust.py
Output: reports/robust_results.csv, reports/model_robust.txt
"""
import json
import pathlib

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from baseline import REALISTIC, STRICT, TRAIN_END, VALID_END, load

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ROBUST = REALISTIC + ["pair_match"]
REVIEWS_PER_DAY = 300


def add_pair_match(d):
    """1 if an earlier TRANSFER/CASH_OUT of the opposite type, same amount, happened this hour or the last."""
    t = d[["txn_id", "step", "type", "amount"]]
    hits = duckdb.sql("""
        SELECT DISTINCT b.txn_id FROM t b JOIN t a
          ON a.amount = b.amount AND a.type <> b.type AND a.txn_id < b.txn_id
         AND b.step - a.step BETWEEN 0 AND 1
    """).df()
    return d.assign(pair_match=d.txn_id.isin(hits.txn_id).astype(int))


def take_part(d, rng, low, high, keep_pairs):
    """Fraud rows take only a fraction of the sender's balance.

    keep_pairs=True gives both halves of a pair (same step, same amount) the
    same new amount, so the money chain stays visible.
    """
    d = d.copy()
    f = d.is_fraud == 1
    if keep_pairs:
        pair_key = d.loc[f, "step"].astype(str) + "_" + d.loc[f, "amount"].round(2).astype(str)
        frac_by_pair = {k: rng.uniform(low, high) for k in pair_key.unique()}
        frac = pair_key.map(frac_by_pair).values
        # Both halves move the same money: a share of the first sender's balance.
        base = d.loc[f].groupby(pair_key)["orig_bal_before"].transform("first").values
        new_amount = np.where(base > 0, base * frac, d.loc[f, "amount"].values * frac)
    else:
        frac = rng.uniform(low, high, f.sum())
        new_amount = np.where(d.loc[f, "orig_bal_before"] > 0,
                              d.loc[f, "orig_bal_before"] * frac, d.loc[f, "amount"] * frac)
    d.loc[f, "amount"] = new_amount
    d.loc[f, "log_amount"] = np.log1p(new_amount)
    bal = d.loc[f, "orig_bal_before"]
    d.loc[f, "amount_to_balance"] = np.where(bal > 0, new_amount / bal.where(bal > 0, 1), -1)
    d.loc[f, "empties_account"] = ((bal > 0) & (new_amount >= bal)).astype(d["empties_account"].dtype)
    return d


def fit(train, valid, feats):
    model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    model.fit(train[feats], train.is_fraud, eval_set=[(valid[feats], valid.is_fraud)],
              eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def score_budget(model_name, test_name, model, feats, test):
    s = model.predict_proba(test[feats])[:, 1]
    y = test.is_fraud.values
    days = (test.step.max() - test.step.min() + 1) / 24
    budget = int(REVIEWS_PER_DAY * days)
    top = np.argsort(-s)[:budget]
    return {
        "model": model_name, "test": test_name,
        "pr_auc": round(average_precision_score(y, s), 3),
        "reviews": budget,
        "fraud_caught": int(y[top].sum()), "fraud_total": int(y.sum()),
        "recall": round(y[top].sum() / y.sum(), 3),
        "money_stopped": round(test.amount.values[top][y[top] == 1].sum() / test.amount.values[y == 1].sum(), 3),
        "innocent_reviewed": int(budget - y[top].sum()),
    }


def main():
    rng = np.random.default_rng(0)
    d = add_pair_match(load())
    train, valid, test = (d[d.step <= TRAIN_END], d[(d.step > TRAIN_END) & (d.step <= VALID_END)],
                          d[d.step > VALID_END])

    # Old alarm: the strict baseline, trained on fraud as simulated.
    old = fit(train, valid, STRICT)

    # New alarm: more features, and extra training fraud that leaves money behind.
    def augment(part):
        fraud = part[part.is_fraud == 1]
        extra = take_part(fraud, rng, 0.5, 0.95, keep_pairs=True)
        extra_split = take_part(fraud, rng, 0.5, 0.95, keep_pairs=False).assign(pair_match=0)
        return pd.concat([part, extra, extra_split], ignore_index=True)
    new = fit(augment(train), augment(valid), ROBUST)

    test_rng = np.random.default_rng(1)  # own seed, so test cases do not depend on training draws
    tests = {
        "as_simulated": test,
        "partial": take_part(test, test_rng, 0.6, 0.95, keep_pairs=True),
        "partial_split": take_part(test, test_rng, 0.6, 0.95, keep_pairs=False),
    }
    tests["partial_split"] = add_pair_match(tests["partial_split"])
    # Hardest case: also send to mule accounts that already hold money. PaySim mules
    # almost always start at zero, so borrow balances from honest receivers.
    funded = tests["partial_split"].copy()
    f = funded.is_fraud == 1
    honest_dest_bal = funded.loc[~f & (funded.dest_bal_before > 0), "dest_bal_before"].values
    funded.loc[f, "dest_bal_before"] = test_rng.choice(honest_dest_bal, f.sum())
    tests["partial_split_funded_mule"] = funded

    rows = []
    for test_name, t in tests.items():
        rows.append(score_budget("old (strict)", test_name, old, STRICT, t))
        rows.append(score_budget("new (robust)", test_name, new, ROBUST, t))
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "robust_results.csv", index=False)
    new.booster_.save_model(str(REPORTS / "model_robust.txt"))

    # Alert level for the app: the score of the last alert a 300-a-day team would
    # review on the validation days (never the test days).

    v_scores = np.sort(new.predict_proba(valid[ROBUST])[:, 1])[::-1]
    v_days = (valid.step.max() - valid.step.min() + 1) / 24
    threshold = float(v_scores[int(REVIEWS_PER_DAY * v_days) - 1])
    (REPORTS / "alert_threshold.json").write_text(json.dumps(
        {"threshold": threshold, "reviews_per_day": REVIEWS_PER_DAY}, indent=2))

    # Example transactions for the app, from the test days only.
    examples = []
    labels = {"as_simulated": "Thief takes everything",
              "partial_split_funded_mule": "Smarter thief (part of balance, no chain, funded mule)"}
    for test_name, label in labels.items():
        t = tests[test_name].assign(score=new.predict_proba(tests[test_name][ROBUST])[:, 1])
        fraud, honest = t[t.is_fraud == 1], t[t.is_fraud == 0]
        picks = [fraud.sample(4, random_state=1).assign(case=f"{label}: fraud"),
                 honest.nlargest(200, "score").sample(2, random_state=1).assign(case="Honest but unusual"),
                 honest.sample(4, random_state=1).assign(case="Ordinary honest transaction")]
        examples += picks
    pd.concat(examples).drop(columns="score").drop_duplicates(["txn_id", "case"]).to_parquet(
        REPORTS / "app_examples.parquet", index=False)
    print(f"Alert level {threshold:.4f} saved, with example transactions for the app")
    print(res.to_string(index=False))
    imp = pd.Series(new.booster_.feature_importance("gain"), index=ROBUST).sort_values(ascending=False)
    print("\nWhat the new alarm relies on (share of total gain):")
    print((imp / imp.sum()).round(3).to_string())


if __name__ == "__main__":
    main()
