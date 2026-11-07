"""Show how a "99% accuracy" result on PaySim comes about, and what it hides.

Recipe used by many public notebooks:
  1. keep TRANSFER and CASH_OUT, 2. undersample honest transactions to 50/50,
  3. random train/test split, 4. use all balance columns, 5. report accuracy.
Then the same model is scored on a realistic test set: the last week of data
at the real fraud rate.

Usage: python src/flawed_replication.py
"""
import pathlib

import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = (ROOT / "data" / "paysim.parquet").as_posix()
FEATURES = ["amount", "orig_bal_before", "orig_bal_after", "dest_bal_before", "dest_bal_after", "is_transfer"]
LAST_WEEK = 504  # train on days 1-21, test on days 22-31


def load(where):
    return duckdb.sql(f"""
        SELECT {', '.join(FEATURES[:-1])}, (type = 'TRANSFER')::INT AS is_transfer, step, is_fraud
        FROM '{DATA}' WHERE type IN ('TRANSFER', 'CASH_OUT') AND {where}
    """).df()


def main():
    rng = np.random.default_rng(0)
    early = load(f"step <= {LAST_WEEK}")

    # The flawed way: balance the classes, then split at random.
    fraud = early[early.is_fraud == 1]
    honest = early[early.is_fraud == 0].sample(len(fraud), random_state=0)
    bal = pd.concat([fraud, honest]).sample(frac=1, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(bal[FEATURES], bal.is_fraud, test_size=0.25, random_state=0)
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
    print(f"Balanced, random split: accuracy {accuracy_score(y_te, model.predict(X_te)):.3f}")

    # The same model on days 22-31, at the real fraud rate.
    late = load(f"step > {LAST_WEEK}")
    pred = model.predict(late[FEATURES])
    score = model.predict_proba(late[FEATURES])[:, 1]
    all_late = duckdb.sql(f"SELECT count(*) n, sum(is_fraud) f FROM '{DATA}' WHERE step > {LAST_WEEK}").fetchall()[0]
    print(f"Days 22-31, real traffic: {all_late[0]:,} transactions, {all_late[1]:,} fraud "
          f"({all_late[1] / all_late[0]:.2%})")
    print(f"  'always honest' accuracy on all days 22-31 traffic: {1 - all_late[1] / all_late[0]:.4f}")
    print(f"  model accuracy (TRANSFER+CASH_OUT): {accuracy_score(late.is_fraud, pred):.4f}")
    print(f"  precision {precision_score(late.is_fraud, pred):.3f}  recall {recall_score(late.is_fraud, pred):.3f}  "
          f"PR-AUC {average_precision_score(late.is_fraud, score):.3f}")
    print(f"  alerts raised: {pred.sum():,} of which real fraud: {(pred & late.is_fraud.values).sum():,}")


if __name__ == "__main__":
    main()
