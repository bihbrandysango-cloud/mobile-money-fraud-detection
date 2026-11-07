"""Mobile money fraud alarm: a simple demo page for non-technical visitors.

Run: .venv/Scripts/python -m streamlit run src/app.py
"""
import pandas as pd
import streamlit as st

from scoring import REPORTS, REVIEWS_PER_DAY, check

st.set_page_config(page_title="Is This Transaction Safe?", page_icon="🛡️", layout="centered")

STORIES = {
    "normal": dict(label="🙂 A normal withdrawal",
                   story="Marie withdraws 25,000 FCFA at her usual agent in the afternoon.",
                   kind="withdraw", amount=25_000, balance=180_000, hour=14, new=False, moved=False),
    "obvious": dict(label="🚨 An obvious theft",
                    story="At 2 a.m., someone empties Marie's account into a brand-new account. "
                          "The same amount is cashed out straight away.",
                    kind="send", amount=450_000, balance=450_000, hour=2, new=True, moved=True),
    "clever": dict(label="🕵️ A clever theft",
                   story="Late at night, a thief takes most (not all) of Marie's money and sends it "
                         "to a new account, hoping to look less suspicious.",
                   kind="send", amount=300_000, balance=420_000, hour=23, new=True, moved=False),
}


def load_story(key):
    s = STORIES[key]
    st.session_state.update(kind="Send money to another account" if s["kind"] == "send" else
                            "Withdraw cash at an agent", amount=s["amount"], balance=s["balance"],
                            hour=s["hour"], new="Yes" if s["new"] else "No",
                            moved="Yes" if s["moved"] else "No", story=s["story"])


if "kind" not in st.session_state:
    load_story("normal")

st.title("🛡️ Is this mobile money transaction safe?")
st.write("Thieves who get into a mobile money account usually **send the money to a new account and "
         "cash it out fast**. This tool checks a transaction and tells you, in plain words, whether to "
         "let it through or stop it.")

tab_check, tab_how, tab_data = st.tabs(["Check a transaction", "How it works", "For data people"])

with tab_check:
    st.subheader("Step 1: Try a story, or type your own")
    cols = st.columns(3)
    for col, key in zip(cols, STORIES):
        col.button(STORIES[key]["label"], on_click=load_story, args=(key,), use_container_width=True)
    st.info(st.session_state.story)

    st.subheader("Step 2: Answer 5 quick questions")
    st.radio("1. What is happening?", ["Withdraw cash at an agent", "Send money to another account"],
             key="kind", horizontal=True)
    a, b = st.columns(2)
    a.number_input("2. How much? (FCFA)", min_value=100, step=5_000, key="amount")
    b.number_input("3. How much was in the account before? (FCFA)", min_value=0, step=5_000, key="balance")
    st.slider("4. What time is it? (hour of the day)", 0, 23, key="hour", format="%d:00")
    c, d = st.columns(2)
    c.radio("5a. Is the receiving account new and empty?", ["No", "Yes"], key="new", horizontal=True)
    d.radio("5b. Was the same amount just moved the other way?", ["No", "Yes"], key="moved", horizontal=True,
            help="For example, the same amount was sent to an account and then withdrawn within the hour.")

    st.subheader("Step 3: The answer")
    kind = "send" if st.session_state.kind.startswith("Send") else "withdraw"
    result = check(kind, st.session_state.amount, st.session_state.balance, st.session_state.hour,
                   st.session_state.new == "Yes", st.session_state.moved == "Yes")
    if result["verdict"] == "stop":
        st.error(f"## 🔴 {result['headline']}\nThe tool is **{result['confidence']}** this could be a theft.")
    else:
        st.success(f"## 🟢 {result['headline']}\nThe tool is **{result['confidence']}** this is safe.")
    st.markdown("**Why:**")
    for reason in result["reasons"]:
        st.markdown(f"- {reason}")
    st.caption("Demo only. The tool learned from simulated transactions, not real MTN or Orange "
               "customers. Amounts are shown in FCFA to make them easy to read.")

with tab_how:
    st.markdown(f"""
**1. It learned from a month of transactions.**
Six million simulated mobile money transactions, a few thousand of them thefts.

**2. It looks for the signs thieves leave.**
Taking most or all of the money, sending it to a new empty account, moving the same amount
straight back out as cash, and doing it at night.

**3. It gives each transaction a risk level.**
The line between "let it through" and "stop it" is set so a fraud team would call about
**{REVIEWS_PER_DAY} customers a day**, a workload a small team can handle.

**4. It explains itself.**
Every answer comes with the main reasons, so the person calling the customer knows what to ask.

**What it can't do.** It isn't perfect. When thieves were made smarter in testing, it still caught
about 7 in 10 thefts, but some got through. A good alarm helps people decide; it doesn't replace them.
""")

with tab_data:
    st.write("A model trained on PaySim (simulated African mobile money data), tested on the last "
             f"10 days with a review budget of {REVIEWS_PER_DAY} alerts a day.")
    res = pd.read_csv(REPORTS / "robust_results.csv")
    names = {"as_simulated": "Thieves take everything", "partial": "Take 60-95%, chain kept",
             "partial_split": "Take 60-95%, chain broken",
             "partial_split_funded_mule": "Same, and mule account holds money"}
    table = res.pivot(index="test", columns="model", values="recall").reindex(list(names)).rename(index=names)
    table.columns = ["New alarm" if "new" in c else "Old alarm" for c in table.columns]
    st.markdown("**Share of thefts caught**")
    st.dataframe(table.map(lambda v: f"{v:.0%}"), use_container_width=True)
    st.image(str(REPORTS / "shap_summary.png"), caption="Which clues push the alarm up or down (SHAP)")
    st.markdown("Method, data quirks and limits: see `notes/findings.md` in the project repository.")
