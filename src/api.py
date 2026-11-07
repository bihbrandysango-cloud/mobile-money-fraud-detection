"""Scoring API: send one transaction, get a fraud verdict and the reasons.

Run:  .venv/Scripts/python -m uvicorn api:app --app-dir src --port 8000
Docs: http://localhost:8000/docs

/check  - five everyday questions in, plain answer out (used by the web page)
/score  - full set of model inputs, risk score and SHAP reasons (for data people)
"""
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scoring import build_features, check, score

app = FastAPI(title="Mobile money fraud alarm",
              description="Demo trained on PaySim (simulated data), not real MTN or Orange customers.")

# Public demo with no logins or cookies, so any web page may call it.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class Question(BaseModel):
    kind: Literal["send", "withdraw"] = Field(description="send = to another account, withdraw = cash at an agent")
    amount: float = Field(gt=0, le=100_000_000, description="FCFA")
    balance: float = Field(ge=0, le=1_000_000_000, description="Money in the account before, FCFA")
    hour: int = Field(ge=0, le=23)
    receiver_is_new: bool = Field(description="Receiving account is new and empty")
    same_amount_moved: bool = Field(description="Same amount was just moved the other way")


class Transaction(BaseModel):
    type: Literal["TRANSFER", "CASH_OUT"]
    amount: float = Field(gt=0)
    sender_balance: float = Field(ge=0, description="Sender's balance before the transaction")
    hour: int = Field(ge=0, le=23)
    receiver_balance: float = Field(ge=0, description="Receiver's balance before the transaction")
    receiver_payments_in: int = Field(0, ge=0)
    receiver_amount_in: float = Field(0, ge=0)
    receiver_payments_out: int = Field(0, ge=0)
    receiver_hours_since_last: Optional[int] = Field(None, ge=0, description="Leave empty if never paid before")
    matching_opposite_move: bool = Field(False, description="Same amount moved the other way in the past hour")


@app.post("/check")
def check_transaction(q: Question):
    return check(**q.model_dump())


@app.post("/score")
def score_transaction(t: Transaction):
    return score(build_features(**t.model_dump()))


@app.get("/health")
def health():
    return {"status": "ok"}
