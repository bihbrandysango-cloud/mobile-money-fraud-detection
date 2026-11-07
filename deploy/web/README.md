---
title: Mobile Money Fraud Alarm
emoji: 🛡️
colorFrom: green
colorTo: red
sdk: static
pinned: false
license: apache-2.0
---

A mobile money fraud alarm that runs entirely in the browser. Trained on PaySim (simulated data, not real MTN or Orange customers).

Use it from any web page:

```html
<script src="https://brandybih-mobile-money-fraud-alarm.static.hf.space/fraud-alarm.js"></script>
<script>
  FraudAlarm.check({kind: "send", amount: 300000, balance: 420000, hour: 23,
                    receiverIsNew: true, sameAmountMoved: false});
  // -> {verdict: "stop", headline, confidence, reasons: [...], risk}
</script>
```
