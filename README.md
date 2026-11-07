# Mobile Money Fraud Detection: what "99% accuracy" hides

**Live demo:** https://pixel-perfect-page-615.lovable.app (runs in your browser; the alarm file is hosted on the Hugging Face Space [BrandyBih/mobile-money-fraud-alarm](https://huggingface.co/spaces/BrandyBih/mobile-money-fraud-alarm))

Mobile money (MTN MoMo, Orange Money) is how most Cameroonians move money, and wallet takeover fraud follows one pattern: transfer to a mule, then cash out. This project builds a fraud model on PaySim, a public simulation of an African mobile money service, and tests the headline numbers that public notebooks report.

Findings so far: [notes/findings.md](notes/findings.md). Plan: [notes/project_plan.md](notes/project_plan.md).

## Data

PaySim (Lopez-Rojas, Elmir & Axelsson, 2016), Kaggle `ealaxi/paysim1`, CC BY-SA 4.0. Simulated data, not MTN or Orange records. Download the CSV into `data/raw/`.

## Run

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python src/prepare.py
.venv/Scripts/python src/flawed_replication.py
.venv/Scripts/python src/baseline.py
.venv/Scripts/python src/robust.py
.venv/Scripts/python src/explain.py
```

## Try the alarm

The trained model ships in `reports/`, so the app runs without the raw data:

```
.venv/Scripts/python -m streamlit run src/app.py                    # demo page
.venv/Scripts/python -m uvicorn api:app --app-dir src --port 8000   # API, docs at /docs
```

Each score comes with its reasons (SHAP), for example "Sends 90% of the sender's balance" or "The same amount just moved the other way (transfer then cash-out)".

## Credit

The question comes from fcamuz/fraud-detection-for-mobile-transactions, which reported ~99% accuracy. No code was copied; it has no open licence.
