# Mobile money fraud detection: plan

## The problem

Mobile money is how most Cameroonians move money. In 2024 Cameroon handled 65% of all CEMAC mobile money transactions (BEAC). Fraud usually follows one pattern: someone gets into a wallet through a fake MTN/Orange call, a "wrong transfer" SMS or a SIM swap. They **transfer** the balance to a mule account, and the mule **cashes it out** at an agent. An operator needs to catch that while it's happening, without blocking thousands of honest customers.

## The data

PaySim (Kaggle `ealaxi/paysim1`, CC BY-SA 4.0). It's a simulation of one month of a mobile money service in an African country, built from a real operator's logs. **It isn't MTN data.** MTN and Orange MoMo are the setting, not the source. It has 6.36 million transactions, and each `step` is one hour.

## What's wrong with existing work

The reference repo, fcamuz/fraud-detection-for-mobile-transactions (Flatiron educational licence), and many Kaggle notebooks report about 99% accuracy. They do it by:
1. **Balancing the classes before testing.** Real traffic is about 0.13% fraud, so a model that always says "not fraud" already scores 99.87%.
2. **Random train/test split.** The model sees the future, which can't happen in production.
3. **Using balances recorded after the transaction.** PaySim's after-balances contain simulator quirks that give the answer away.

We don't copy their code (no open licence). We credit them and rebuild from scratch.

## Our approach

1. **Replicate the claim:** show how the 99% comes about (`src/flawed_replication.py`).
2. **Honest test:** train on the first ~3 weeks and test on the last week. Report precision-recall, not accuracy.
3. **Two feature sets:**
   - *Realistic:* only what's known when the request arrives (type, amount, sender's balance before, hour, the sender's and receiver's history so far).
   - *Leaky:* adds after-transaction balances, to show how much of the "accuracy" comes from them.
4. **Cost-based threshold:** a missed fraud loses the amount, while a false alarm costs a review call. Choose the cut-off with the lowest total cost, given what a fraud team can check each day.
5. **Explanations:** SHAP gives the reasons for each flagged transaction.
6. **App:** FastAPI scoring endpoint + Streamlit page (later step).

## Deliverables

README with the "99% vs reality" story, `reports/` charts, the model, and a small demo app.
