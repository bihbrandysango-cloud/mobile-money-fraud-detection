"""Score one transaction with the robust model and explain the score in plain words.

Shared by the API (src/api.py) and the app (src/app.py).

The reasons come from SHAP: for this one transaction, how much each clue pushed
the risk up or down. The clues with the biggest push are turned into sentences.
"""
import json
import math
import pathlib
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

ROOT = pathlib.Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", message="LightGBM binary classifier")
REPORTS = ROOT / "reports"

FEATURES = ["is_transfer", "amount", "log_amount", "orig_bal_before", "amount_to_balance",
            "empties_account", "hour_of_day", "dest_bal_before", "dest_prior_in_count",
            "dest_prior_in_amount", "dest_prior_out_count", "dest_hours_since_last", "pair_match"]

_booster = lgb.Booster(model_file=str(REPORTS / "model_robust.txt"))
assert _booster.feature_name() == FEATURES, "model and feature list are out of step"
_explainer = shap.TreeExplainer(_booster)
_alert = json.loads((REPORTS / "alert_threshold.json").read_text())
THRESHOLD = _alert["threshold"]
REVIEWS_PER_DAY = _alert["reviews_per_day"]


def build_features(type, amount, sender_balance, hour, receiver_balance, receiver_payments_in=0,
                   receiver_amount_in=0.0, receiver_payments_out=0, receiver_hours_since_last=None,
                   matching_opposite_move=False):
    """Turn what an operator sees about a request into the model's inputs."""
    amount, sender_balance = float(amount), float(sender_balance)
    return pd.DataFrame([{
        "is_transfer": int(type.upper() == "TRANSFER"),
        "amount": amount,
        "log_amount": math.log1p(amount),
        "orig_bal_before": sender_balance,
        "amount_to_balance": amount / sender_balance if sender_balance > 0 else -1.0,
        "empties_account": int(sender_balance > 0 and amount >= sender_balance),
        "hour_of_day": int(hour),
        "dest_bal_before": float(receiver_balance),
        "dest_prior_in_count": int(receiver_payments_in),
        "dest_prior_in_amount": float(receiver_amount_in),
        "dest_prior_out_count": int(receiver_payments_out),
        "dest_hours_since_last": -1 if receiver_hours_since_last is None else int(receiver_hours_since_last),
        "pair_match": int(bool(matching_opposite_move)),
    }])[FEATURES]


def _sentence(feature, x):
    """Plain-language description of one clue's value."""
    v = x[feature]
    if feature in ("amount", "log_amount"):
        return f"Amount of {x['amount']:,.0f}"
    if feature == "amount_to_balance":
        if v < 0:
            return "The sender's account showed no balance"
        if v > 1.001:
            return f"Sends more than the sender's recorded balance ({v:.1f} times it)"
        return f"Sends {v:.0%} of the sender's balance"
    if feature == "empties_account":
        if v and x["amount_to_balance"] > 1.001:
            return "Amount is larger than the recorded balance"
        return "Takes the whole balance" if v else "Leaves money in the sender's account"
    if feature == "pair_match":
        return ("The same amount just moved the other way (transfer then cash-out), a typical mule chain"
                if v else "No matching transfer or cash-out of the same amount in the past hour")
    if feature == "orig_bal_before":
        return f"Sender's balance before: {v:,.0f}"
    if feature == "is_transfer":
        return "It is a transfer to another account" if v else "It is a cash-out"
    if feature == "dest_bal_before":
        return "The receiving account was empty" if v == 0 else f"The receiving account already held {v:,.0f}"
    if feature == "hour_of_day":
        return f"Made at {int(v):02d}:00" + (", in the middle of the night" if v < 6 else "")
    if feature == "dest_prior_in_count":
        return ("The receiving account had never been paid before" if v == 0
                else f"The receiving account had been paid {int(v)} times before")
    if feature == "dest_prior_in_amount":
        return f"The receiving account had received {v:,.0f} in total before"
    if feature == "dest_prior_out_count":
        return f"The receiving account had sent money {int(v)} times before"
    if feature == "dest_hours_since_last":
        return ("No earlier payment to the receiving account" if v < 0
                else f"The receiving account was last paid {int(v)} hours earlier")
    return f"{feature} = {v}"


def score(features, top=4):
    """Return risk (0-1), whether it would be an alert, and the main reasons."""
    risk = float(_booster.predict(features)[0])
    contrib = pd.Series(np.ravel(_explainer.shap_values(features))[: len(FEATURES)], index=FEATURES)
    # Amount appears twice (raw and log); report it once.
    contrib["amount"] += contrib.pop("log_amount")
    x = features.iloc[0]
    order = contrib.abs().sort_values(ascending=False).index[:top]
    reasons = [{"clue": _sentence(f, x),
                "effect": "raises risk" if contrib[f] > 0 else "lowers risk",
                "strength": round(float(contrib[f]), 2)} for f in order]
    return {"risk": round(risk, 4), "alert": risk >= THRESHOLD, "alert_level": round(THRESHOLD, 4),
            "reasons": reasons}


# ---- Simple version for non-technical users (app page and public API) ----

_typical_receiver = json.loads((REPORTS / "receiver_profiles.json").read_text())["established_receiver"]


def _plain(feature, x, kind):
    """Short, everyday sentence for one clue, written for a customer or agent."""
    share = x["amount_to_balance"]
    hour = int(x["hour_of_day"])
    if feature in ("amount_to_balance", "empties_account"):
        if share < 0:
            return "The account showed no money before this"
        if share > 1.001:
            return "The amount is more than the account showed"
        if share >= 0.999:
            return "It takes all the money in the account"
        return f"It takes {share:.0%} of the money in the account"
    if feature == "pair_match":
        return ("The same amount was just moved out the other way, a common thief pattern"
                if x["pair_match"] else "No matching amount was just moved the other way")
    if feature in ("dest_bal_before", "dest_prior_in_count", "dest_prior_in_amount",
                   "dest_prior_out_count", "dest_hours_since_last"):
        if kind == "withdraw":
            return ("The agent account was empty and has no history" if x["dest_prior_in_count"] == 0
                    else "The agent account has a normal history")
        return ("The money goes to a new, empty account" if x["dest_prior_in_count"] == 0
                else "The money goes to an account with a normal history")
    if feature == "hour_of_day":
        if hour < 6:
            return f"It happens at {hour:02d}:00, in the middle of the night"
        return f"It happens at {hour:02d}:00, during normal hours" if 7 <= hour <= 20 else f"It happens late, at {hour:02d}:00"
    if feature == "is_transfer":
        return "It sends money to another account" if kind == "send" else "It is a cash withdrawal at an agent"
    if feature in ("amount", "log_amount"):
        return f"The amount is {x['amount']:,.0f} FCFA"
    if feature == "orig_bal_before":
        return f"The account had {x['orig_bal_before']:,.0f} FCFA before"
    return feature


def check(kind, amount, balance, hour, receiver_is_new, same_amount_moved):
    """Everyday questions in, plain answer out.

    kind: "send" (to another account) or "withdraw" (cash at an agent)
    amount, balance: FCFA (the simulation's own units, shown as FCFA)
    hour: 0-23
    receiver_is_new: True if the receiving account is new and empty
    same_amount_moved: True if the same amount was just moved the other way
    """
    if kind not in ("send", "withdraw"):
        raise ValueError("kind must be 'send' or 'withdraw'")
    if receiver_is_new:
        r = dict(receiver_balance=0.0, receiver_payments_in=0, receiver_amount_in=0.0,
                 receiver_payments_out=0, receiver_hours_since_last=None)
    else:
        t = _typical_receiver
        r = dict(receiver_balance=t["dest_bal_before"], receiver_payments_in=t["dest_prior_in_count"],
                 receiver_amount_in=t["dest_prior_in_amount"], receiver_payments_out=t["dest_prior_out_count"],
                 receiver_hours_since_last=t["dest_hours_since_last"])
    features = build_features("TRANSFER" if kind == "send" else "CASH_OUT", amount, balance, hour,
                              matching_opposite_move=same_amount_moved, **r)
    risk = float(_booster.predict(features)[0])
    alert = risk >= THRESHOLD
    contrib = pd.Series(np.ravel(_explainer.shap_values(features))[: len(FEATURES)], index=FEATURES)
    x = features.iloc[0]

    # Merge clues that say the same thing, then keep the ones that agree with the verdict.
    groups = {"amount_to_balance": ["amount_to_balance", "empties_account"], "amount": ["amount", "log_amount"],
              "dest_bal_before": ["dest_bal_before", "dest_prior_in_count", "dest_prior_in_amount",
                                  "dest_prior_out_count", "dest_hours_since_last"]}
    merged = {}
    for f in FEATURES:
        key = next((k for k, v in groups.items() if f in v), f)
        merged[key] = merged.get(key, 0.0) + contrib[f]
    merged = pd.Series(merged)
    side = merged[merged > 0] if alert else merged[merged < 0]
    top = side.abs().sort_values(ascending=False).index[:3]
    return {
        "verdict": "stop" if alert else "allow",
        "headline": "Stop and call the customer" if alert else "Looks normal, let it through",
        "confidence": ("very sure" if risk >= 0.5 or risk < THRESHOLD / 10 else "fairly sure"),
        "reasons": [_plain(f, x, kind) for f in top],
    }


if __name__ == "__main__":
    demo = build_features("TRANSFER", 180_000, 200_000, 3, 0, matching_opposite_move=True)
    print(json.dumps(score(demo), indent=2))
    for case in [("withdraw", 25_000, 180_000, 14, False, False),
                 ("send", 450_000, 450_000, 2, True, True),
                 ("send", 300_000, 420_000, 23, True, False)]:
        print(case, json.dumps(check(*case), indent=1))
