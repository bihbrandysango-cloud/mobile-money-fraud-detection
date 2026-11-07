# Findings so far (6 Nov 2026)

## 1. The data

6,362,620 transactions over 743 hours (31 days); 8,213 are fraud (0.13%).

- Fraud happens only in TRANSFER (4,097) and CASH_OUT (4,116), the "move it to a mule, then withdraw" pattern.
- The simulator's own rule (`isFlaggedFraud`) flags 16 of 8,213 frauds.
- Fraud runs at about 1,850 cases a week every week, but honest traffic collapses in the last 10 days (1.93 M transactions in week 1, 66 k in the final days). This is a simulator quirk. Any time-based test therefore sees more fraud (0.87% of days 22-31) than real life would.
- Fraud is spread evenly across the 24 hours, while honest use is mostly 08:00-20:00. Between 02:00 and 05:00, 16-22% of transactions are fraud.

## 2. The "99% accuracy" claim (`src/flawed_replication.py`)

| Test | Result |
|---|---|
| Balanced 50/50 sample, random split | **98.6% accuracy** |
| Same model, days 22-31, real mix | 97.9% accuracy, *lower* than calling everything honest (99.1%) |
| Alerts raised on days 22-31 | 5,152, of which 2,566 were innocent customers (precision 0.50) |

Accuracy is the wrong yardstick for rare events.

## 3. Honest test (`src/baseline.py`)

Split by time (train days 1-18, tune on days 19-21, test on days 22-31), scoring LightGBM with precision-recall and cost. Test PR-AUC was 0.998-1.000 for every feature set, **even the strict one that uses no after-transaction balances.**

Why: **97.8% of PaySim frauds transfer the exact full balance (`amount == balance before`). Honest transactions never do (0.000%).** That's how the simulator's fraud agent was programmed. A one-line rule nearly solves the dataset, so a near-perfect score here says little about real fraud.

## 4. Stress test: fraudsters who leave something behind

Same strict model, tested on days 22-31, with each fraud changed to take a random 60-95% of the balance instead of all of it:

| Fraudster behaviour | PR-AUC | Fraud caught when flagging 2,592 transactions |
|---|---|---|
| As simulated (takes everything) | 0.998 | 99.1% |
| Leaves 5-40% behind | **0.249** | **24.5%** |

One small change in behaviour, and three out of four frauds get through. Real fraudsters adapt as soon as a rule is known.

## What this means for the project

The headline isn't "my model scores 99.9%". It's three points:
1. Accuracy on balanced data is misleading. Here's the right way to measure.
2. Even measured properly, PaySim fraud is too easy, because of one simulated habit.
3. A model that relies on that habit fails as soon as fraudsters adapt. So build features that hold up (receiver and mule-account behaviour, timing, how fast money moves on) and test against adapted fraud.

Next decisions are in `project_plan.md`.

## 5. The stronger alarm (`src/robust.py`)

Changes from the strict baseline:
- **Money-chain clue (`pair_match`):** every PaySim fraud is a TRANSFER plus a CASH_OUT of the exact same amount within the hour. 63% of fraud rows have such an earlier match, against 0.1% of honest rows.
- **Receiver clues:** the receiving account's balance and history before the transaction.
- **Practice against smarter thieves:** training includes extra copies of each fraud that take only 50-95% of the balance, some with the chain kept and some with it broken.

The test is days 22-31, with a fraud team that can review **300 alerts a day** (2,987 in total, against 2,592 real frauds).

| How thieves behave in the test | Old alarm catches | New alarm catches |
|---|---|---|
| As simulated (take everything) | 100% | 100% |
| Take 60-95%, chain intact | 29% | **100%** |
| Take 60-95%, chain broken | 29% | **88%** |
| Same, and the mule account already holds money | 29% | **69%** |

Test cases use their own fixed random seed. With other seeds the old alarm lands between about 13% and 30%, so read it as "roughly one in four or fewer".

What the new alarm relies on most: share of balance sent (24%), money-chain match (23%), sender's balance (19%), transfer vs cash-out (16%), receiver's balance (8%), hour (4%).

**Honest limits**
- The "take 60-95%" tests use a trick close to the one the alarm practised on, so 100% and 88% are the best case. The funded-mule test (69%) is the fairer one, because the alarm never practised it.
- The receiver's balance helps mainly because PaySim mules start at zero (99.9% of fraud transfers). When that habit is removed, recall drops from 88% to 69%.
- Everything is still simulated data. These numbers show how to build and test an alarm, not how well it would work at MTN.
- Another simulator quirk: about half of honest senders show a zero balance before sending (46-54%), while fraud senders almost never do (under 1%). The alarm uses this through "share of balance sent", which is coded -1 when the balance is zero. See `reports/shap_summary.png`.
