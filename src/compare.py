"""Compare four alarms over five random seeds, in thefts caught and in money.

Alarms
  rule        - one line: flag when the amount is the whole balance (biggest first)
  logistic    - logistic regression on the strict clues
  old         - LightGBM on the strict clues (src/baseline.py)
  new         - LightGBM robust model (src/robust.py): extra clues + practice on smarter thieves

Tests (days 22-31), as in src/robust.py
  as_simulated, partial, partial_split, partial_split_funded_mule

Each seed changes the model's random start, the practice copies and the
test changes, so the spread shows how much a result depends on luck.

Money: a fraud team reviews 300 alerts a day. A caught theft keeps its amount
in the customer's account; each review costs REVIEW_COST. Amounts are PaySim's
own units, shown as FCFA on the website.

Usage: python src/compare.py
Output: reports/compare_runs.csv (one row per seed x alarm x test),
        reports/compare_summary.csv, reports/pr_scores_seed0.parquet
"""
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import REVIEW_COST, STRICT, TRAIN_END, VALID_END, load
from robust import REVIEWS_PER_DAY, ROBUST, add_pair_match, take_part

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SEEDS = [0, 1, 2, 3, 4]
TESTS = ["as_simulated", "partial", "partial_split", "partial_split_funded_mule"]


def fit_lgb(train, valid, feats, seed):
    model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                               random_state=seed)
    model.fit(train[feats], train.is_fraud, eval_set=[(valid[feats], valid.is_fraud)],
              eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
    return lambda d: model.predict_proba(d[feats])[:, 1]


def fit_logistic(train, seed):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    model.fit(train[STRICT], train.is_fraud)
    return lambda d: model.predict_proba(d[STRICT])[:, 1]


def rule(d):
    # Exactly the whole balance sent (the PaySim fraud habit): flag it.
    # Among flagged ones, review the biggest first.
    exact = (d.orig_bal_before.values > 0) & np.isclose(d.amount.values, d.orig_bal_before.values)
    return exact + d.amount.values / 1e12


def make_tests(test, rng):
    t = {"as_simulated": test,
         "partial": take_part(test, rng, 0.6, 0.95, keep_pairs=True),
         "partial_split": add_pair_match(take_part(test, rng, 0.6, 0.95, keep_pairs=False))}
    funded = t["partial_split"].copy()
    f = funded.is_fraud == 1
    honest_bal = funded.loc[~f & (funded.dest_bal_before > 0), "dest_bal_before"].values
    funded.loc[f, "dest_bal_before"] = rng.choice(honest_bal, f.sum())
    t["partial_split_funded_mule"] = funded
    return t


def evaluate(score, t):
    y, amount = t.is_fraud.values, t.amount.values
    days = (t.step.max() - t.step.min() + 1) / 24
    reviews = int(REVIEWS_PER_DAY * days)
    top = np.argsort(-score)[:reviews]
    caught = y[top] == 1
    stolen = amount[y == 1].sum()
    kept = amount[top][caught].sum()
    return {"pr_auc": average_precision_score(y, score), "recall": caught.sum() / y.sum(),
            "fraud_caught": int(caught.sum()), "fraud_total": int(y.sum()), "reviews": reviews,
            "stolen_total": stolen, "money_kept": kept, "review_cost": reviews * REVIEW_COST,
            "net_saved": kept - reviews * REVIEW_COST}


def main():
    d = add_pair_match(load())
    train = d[d.step <= TRAIN_END]
    valid = d[(d.step > TRAIN_END) & (d.step <= VALID_END)]
    test = d[d.step > VALID_END]

    rows, pr_scores = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)

        def augment(part):
            fraud = part[part.is_fraud == 1]
            return pd.concat([part, take_part(fraud, rng, 0.5, 0.95, keep_pairs=True),
                              take_part(fraud, rng, 0.5, 0.95, keep_pairs=False).assign(pair_match=0)],
                             ignore_index=True)

        alarms = {"rule": rule,
                  "logistic": fit_logistic(train, seed),
                  "old": fit_lgb(train, valid, STRICT, seed),
                  "new": fit_lgb(augment(train), augment(valid), ROBUST, seed)}
        tests = make_tests(test, np.random.default_rng(100 + seed))
        for test_name in TESTS:
            t = tests[test_name]
            for alarm, score_fn in alarms.items():
                s = score_fn(t)
                rows.append({"seed": seed, "alarm": alarm, "test": test_name, **evaluate(s, t)})
                if seed == 0 and test_name == "partial_split_funded_mule":
                    pr_scores.append(pd.DataFrame({"alarm": alarm, "score": s, "is_fraud": t.is_fraud.values}))
        print(f"seed {seed} done", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(REPORTS / "compare_runs.csv", index=False)
    pd.concat(pr_scores).to_parquet(REPORTS / "pr_scores_seed0.parquet", index=False)
    summary = (runs.groupby(["test", "alarm"])
               .agg(recall_mean=("recall", "mean"), recall_min=("recall", "min"), recall_max=("recall", "max"),
                    pr_auc_mean=("pr_auc", "mean"), money_kept_mean=("money_kept", "mean"),
                    stolen_total=("stolen_total", "mean"), review_cost=("review_cost", "mean"),
                    net_saved_mean=("net_saved", "mean"))
               .reset_index())
    summary.to_csv(REPORTS / "compare_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
