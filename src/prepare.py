"""Convert the PaySim CSV to Parquet and add the features used by every model.

PaySim (Lopez-Rojas, Elmir & Axelsson, 2016) simulates one month of a mobile
money service in an African country, built from a real operator's logs.
Each `step` is one hour; there are 743 steps (about 31 days).

Usage: python src/prepare.py
Output: data/paysim.parquet
"""
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = next((ROOT / "data" / "raw").glob("PS_*.csv"))
OUT = ROOT / "data" / "paysim.parquet"


def main():
    duckdb.sql(f"""
        COPY (
            SELECT
                row_number() OVER () AS txn_id,
                step,
                (step - 1) % 24 AS hour_of_day,
                (step - 1) // 24 AS day,
                type,
                amount,
                nameOrig AS orig,
                nameDest AS dest,
                oldbalanceOrg AS orig_bal_before,
                newbalanceOrig AS orig_bal_after,
                oldbalanceDest AS dest_bal_before,
                newbalanceDest AS dest_bal_after,
                -- Known when the request arrives: amount compared with the sender's balance.
                amount / NULLIF(oldbalanceOrg, 0) AS amount_to_balance,
                (oldbalanceOrg > 0 AND amount >= oldbalanceOrg) AS empties_account,
                dest LIKE 'M%' AS dest_is_merchant,
                isFraud::INT AS is_fraud,
                isFlaggedFraud::INT AS is_flagged_by_rule
            FROM read_csv_auto('{CSV.as_posix()}')
        ) TO '{OUT.as_posix()}' (FORMAT parquet)
    """)
    print(duckdb.sql(f"SELECT count(*) rows, sum(is_fraud) fraud FROM '{OUT.as_posix()}'"))


if __name__ == "__main__":
    main()
