# Lovable build prompt

Paste everything below the line into Lovable as ONE message, only after the Hugging Face
Static Space is live and its fraud-alarm.js address opens in a browser.
It is written to get the page right in one go, so it doesn't use up free credits on fixes.

---

Build a single-page web app called **"Is this transaction safe?"**. It's a demo of a mobile money fraud alarm for people with no technical background (Cameroon context, MTN MoMo / Orange Money style). Front-end only: **no login, no database, no Supabase, no user accounts, no extra pages.** Mobile-first, works well on a phone.

## Look and feel
- Clean, friendly, trustworthy. White background, dark text, rounded cards, generous spacing, large readable text (base 18px).
- Accent colours: green `#16a34a` for safe, red `#dc2626` for stop.
- A shield icon (lucide `ShieldCheck`) next to the title. No stock photos.

## Page sections, top to bottom (use this exact wording)

**1. Header**
- Title: "Is this mobile money transaction safe?"
- Subtitle: "Thieves who get into a mobile money account usually send the money to a new account and cash it out fast. This tool checks a transaction and tells you, in plain words, whether to let it through or stop it."

**2. "Step 1: Try a story"**: three clickable cards in a row (stacked on phones). Clicking a card fills the form below and immediately runs the check.
- Card "🙂 A normal withdrawal": "Marie withdraws 25,000 FCFA at her usual agent in the afternoon." → kind=withdraw, amount=25000, balance=180000, hour=14, receiverIsNew=false, sameAmountMoved=false
- Card "🚨 An obvious theft": "At 2 a.m., someone empties Marie's account into a brand-new account. The same amount is cashed out straight away." → kind=send, amount=450000, balance=450000, hour=2, receiverIsNew=true, sameAmountMoved=true
- Card "🕵️ A clever theft": "Late at night, a thief takes most (not all) of Marie's money and sends it to a new account, hoping to look less suspicious." → kind=send, amount=300000, balance=420000, hour=23, receiverIsNew=true, sameAmountMoved=false

**3. "Step 2: Answer 5 quick questions"**: a simple form:
1. "What is happening?": two large toggle buttons: "Withdraw cash at an agent" (value `withdraw`) / "Send money to another account" (value `send`).
2. "How much? (FCFA)": number input, min 100, shows thousands separators.
3. "How much was in the account before? (FCFA)": number input, min 0, thousands separators.
4. "What time is it?": a slider from 0 to 23 showing the value as "14:00".
5. "Is the receiving account new and empty?": Yes / No toggle.
6. "Was the same amount just moved the other way?": Yes / No toggle, with a small "?" tooltip: "For example, the same amount was sent to an account and then withdrawn within the hour."
- A big button: "Check this transaction".
- Start with the "normal withdrawal" values filled in.

**4. "Step 3: The answer"**: result card, only shown after a check:
- Get the answer from `window.FraudAlarm.check(...)` (details below). It answers instantly; no loading state is needed.
- If `verdict` is `"stop"`: red card, big text "🔴 " + `headline`, then "The tool is **{confidence}** this could be a theft."
- If `verdict` is `"allow"`: green card, big text "🟢 " + `headline`, then "The tool is **{confidence}** this is safe."
- Below either: "Why:" and the `reasons` array as a bullet list, exactly as returned.
- If the script hasn't loaded or `check` throws an error: grey card, "Sorry, the checker isn't available right now. Please refresh the page."
- Under the card, small grey text: "Demo only. The tool learned from simulated transactions, not real MTN or Orange customers. Amounts are shown in FCFA to make them easy to read."

**5. "How it works"**: four short numbered points:
1. "It learned from a month of transactions: six million simulated mobile money transactions, a few thousand of them thefts."
2. "It looks for the signs thieves leave: taking most or all of the money, sending it to a new empty account, moving the same amount straight back out as cash, and doing it at night."
3. "It gives each transaction a risk level. The line between 'let it through' and 'stop it' is set so a fraud team would call about 300 customers a day."
4. "It explains itself. Every answer comes with the main reasons, so the person calling the customer knows what to ask."
- Then a highlighted note: "**What it can't do.** It isn't perfect. When thieves were made smarter in testing, it still caught about 7 in 10 thefts, but some got through. A good alarm helps people decide; it doesn't replace them."

**6. Footer**: "Built by Bih Brandy Sango · Data: PaySim (simulated) · Code on GitHub" (make "Code on GitHub" a link to `https://github.com/bihbrandysango-cloud`, opening in a new tab).

## The checker (the only source of answers)
The checker is a ready-made JavaScript file. It runs in the visitor's browser; there is no server and no API call.
- Load it once by adding this tag to `index.html`, inside `<head>`:
```html
<script src="https://brandybih-mobile-money-fraud-alarm.static.hf.space/fraud-alarm.js"></script>
```
- It creates `window.FraudAlarm`. Add a TypeScript declaration for it (e.g. in `src/vite-env.d.ts`): `interface Window { FraudAlarm: { check(q: {kind: "send" | "withdraw"; amount: number; balance: number; hour: number; receiverIsNew: boolean; sameAmountMoved: boolean}): {verdict: "stop" | "allow"; headline: string; confidence: string; reasons: string[]; risk: number} } }`
- Call it with the form values, for example:
```js
window.FraudAlarm.check({kind: "send", amount: 300000, balance: 420000, hour: 23, receiverIsNew: true, sameAmountMoved: false})
```
- It returns:
```json
{"verdict": "stop", "headline": "Check with the customer before the money moves", "confidence": "fairly confident",
 "reasons": ["It takes 71% of the money in the account", "The money goes to a new, empty account", "The account had 420,000 FCFA before"],
 "risk": 0.82}
```
- `verdict` is `"stop"` or `"allow"`. Show `headline`, `confidence` and `reasons` exactly as returned; do not invent or reword them. Do not show `risk`.
- Do not write your own fraud rules or risk logic, do not copy or rewrite the script, and do not store anything.

## Don'ts
- Don't add sign-up, login, dashboards, charts, extra pages, dark mode or a chatbot.
- Don't use the words "AI-powered" or "machine learning" anywhere on the page.
- Don't say the data is real MTN or Orange data.
