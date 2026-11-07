"""Checks that run in seconds, with or without the PaySim data.

- the three website stories give the expected verdicts
- bad questions are refused
- the browser file (deploy/web/fraud-alarm.js) matches the current model
- the browser alarm gives the same scores, SHAP values and answers as Python
  (real test-day rows if data/paysim.parquet exists, generated rows otherwise)

Run: python -m pytest -q tests      (the parity test needs Node.js)
"""
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from export_js import OUT, build  # noqa: E402
from scoring import FEATURES, _booster, build_features, check  # noqa: E402

NODE = r"""
const A = require(process.argv[2]);
const data = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const rows = data.rows.map(x => { const p = A._predict(x); return {raw: p.raw, contrib: p.contrib}; });
const qs = data.questions.map(q => A.check(q));
require("fs").writeFileSync(process.argv[4], JSON.stringify({rows, qs}));
"""


@pytest.mark.parametrize("story, verdict", [
    (("withdraw", 25_000, 180_000, 14, False, False), "allow"),   # normal withdrawal
    (("send", 450_000, 450_000, 2, True, True), "stop"),          # obvious theft
    (("send", 300_000, 420_000, 23, True, False), "stop"),        # clever theft
])
def test_website_stories(story, verdict):
    result = check(*story)
    assert result["verdict"] == verdict
    assert len(result["reasons"]) == 3


def test_bad_question_refused():
    with pytest.raises(ValueError):
        check("steal", 1_000, 1_000, 3, True, True)


def test_browser_file_matches_model():
    on_disk = OUT.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    assert on_disk == build(), "Run python src/export_js.py and commit deploy/web/fraud-alarm.js"


def sample_rows(rng, n=3000):
    """Real test-day transactions if the data is here, otherwise realistic made-up ones."""
    data = ROOT / "data" / "paysim.parquet"
    if data.exists():
        from baseline import VALID_END, load
        from robust import add_pair_match
        d = add_pair_match(load())
        return d[d.step > VALID_END].sample(n, random_state=7)[FEATURES].astype(float)
    rows = []
    for _ in range(n):
        balance = float(rng.choice([0, rng.lognormal(11, 2)]))
        amount = float(rng.choice([balance, rng.lognormal(11, 1.5)])) or 1.0
        new = bool(rng.integers(0, 2))
        rows.append(build_features(
            str(rng.choice(["TRANSFER", "CASH_OUT"])), amount, balance, int(rng.integers(0, 24)),
            0.0 if new else float(rng.lognormal(13, 1.5)),
            0 if new else int(rng.integers(1, 40)), 0.0 if new else float(rng.lognormal(14, 1.5)),
            int(rng.integers(0, 3)), None if new else int(rng.integers(0, 200)), bool(rng.integers(0, 2))))
    return pd.concat(rows).astype(float)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_browser_alarm_matches_python(tmp_path):
    rng = np.random.default_rng(7)
    X = sample_rows(rng)
    questions = [{"kind": str(rng.choice(["send", "withdraw"])),
                  "amount": float(rng.integers(1, 2_000) * 500),
                  "balance": float(rng.choice([0, rng.integers(0, 4_000) * 500])),
                  "hour": int(rng.integers(0, 24)),
                  "receiverIsNew": bool(rng.integers(0, 2)),
                  "sameAmountMoved": bool(rng.integers(0, 2))} for _ in range(2000)]
    (tmp_path / "in.json").write_text(json.dumps({"rows": X.values.tolist(), "questions": questions}))
    (tmp_path / "run.js").write_text(NODE)
    subprocess.run(["node", str(tmp_path / "run.js"), str(OUT), str(tmp_path / "in.json"),
                    str(tmp_path / "out.json")], check=True)
    out = json.loads((tmp_path / "out.json").read_text())

    raw_js = np.array([r["raw"] for r in out["rows"]])
    contrib_js = np.array([r["contrib"] for r in out["rows"]])
    assert np.abs(_booster.predict(X, raw_score=True) - raw_js).max() < 1e-6
    assert np.abs(_booster.predict(X, pred_contrib=True)[:, : len(FEATURES)] - contrib_js).max() < 1e-6

    for q, js in zip(questions, out["qs"]):
        py = check(q["kind"], q["amount"], q["balance"], q["hour"], q["receiverIsNew"], q["sameAmountMoved"])
        assert (py["verdict"], py["confidence"], py["reasons"]) == (js["verdict"], js["confidence"], js["reasons"]), q
