"""Check that the browser alarm (deploy/web/fraud-alarm.js) gives the same answers as Python.

1. 3,000 real test-day transactions: risk score and every SHAP value.
2. 2,000 random everyday questions: verdict, confidence and the three reasons.

Usage: python tests/parity.py   (needs Node.js)
"""
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baseline import VALID_END, load  # noqa: E402
from robust import add_pair_match  # noqa: E402
from scoring import FEATURES, _booster, check  # noqa: E402

NODE = r"""
const A = require(process.argv[2]);
const data = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const rows = data.rows.map(x => { const p = A._predict(x); return {raw: p.raw, contrib: p.contrib}; });
const qs = data.questions.map(q => A.check(q));
require("fs").writeFileSync(process.argv[4], JSON.stringify({rows, qs}));
"""


def main():
    rng = np.random.default_rng(7)
    d = add_pair_match(load())
    test = d[d.step > VALID_END]
    sample = test.sample(3000, random_state=7)
    X = sample[FEATURES].astype(float)

    questions = [{"kind": str(rng.choice(["send", "withdraw"])),
                  "amount": float(rng.integers(1, 2_000) * 500),
                  "balance": float(rng.choice([0, rng.integers(0, 4_000) * 500])),
                  "hour": int(rng.integers(0, 24)),
                  "receiverIsNew": bool(rng.integers(0, 2)),
                  "sameAmountMoved": bool(rng.integers(0, 2))} for _ in range(2000)]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "in.json").write_text(json.dumps({"rows": X.values.tolist(), "questions": questions}))
        (tmp / "run.js").write_text(NODE)
        subprocess.run(["node", str(tmp / "run.js"), str(ROOT / "deploy" / "web" / "fraud-alarm.js"),
                        str(tmp / "in.json"), str(tmp / "out.json")], check=True)
        out = json.loads((tmp / "out.json").read_text())

    raw_py = _booster.predict(X, raw_score=True)
    contrib_py = _booster.predict(X, pred_contrib=True)[:, : len(FEATURES)]
    raw_js = np.array([r["raw"] for r in out["rows"]])
    contrib_js = np.array([r["contrib"] for r in out["rows"]])
    print(f"Real transactions: max score difference {np.abs(raw_py - raw_js).max():.2e}, "
          f"max SHAP difference {np.abs(contrib_py - contrib_js).max():.2e}")

    same_verdict = same_conf = same_reasons = 0
    for q, js in zip(questions, out["qs"]):
        py = check(q["kind"], q["amount"], q["balance"], q["hour"], q["receiverIsNew"], q["sameAmountMoved"])
        same_verdict += py["verdict"] == js["verdict"]
        same_conf += py["confidence"] == js["confidence"]
        same_reasons += py["reasons"] == js["reasons"]
    n = len(questions)
    print(f"Everyday questions: verdict {same_verdict}/{n}, confidence {same_conf}/{n}, "
          f"reasons word for word {same_reasons}/{n}")
    ok = (np.abs(raw_py - raw_js).max() < 1e-6 and np.abs(contrib_py - contrib_js).max() < 1e-6
          and same_verdict == n and same_conf == n and same_reasons == n)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
