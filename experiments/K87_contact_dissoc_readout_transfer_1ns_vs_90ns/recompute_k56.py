"""Recompute K87's gain/recovery summary with K56's gain definition,
from the raw per-condition numbers already in results.json. No new
training or evaluation.

K56's definition: the gain of a history condition trained at h1 and
evaluated at h2 is its AUPRC minus the AUPRC of the static-only
condition trained at the EVALUATION horizon h2. The reference gain is
the matched-horizon gain (train@h2 -> eval@h2). Recovery = transfer
gain / matched gain.

The first version of this experiment used the static condition trained
at the TRAIN horizon h1 instead, which is a different measurement when
the static baseline itself transfers poorly across horizons. The old
summary is kept as _summary_old_definition for transparency.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HORIZONS = (1, 5, 10, 30, 50, 90)

results = json.loads((HERE / "results.json").read_text())


def gain(h1, h2, macro=False):
    key = "macro_auprc" if macro else "auprc"
    return float(results[f"history_train[{h1}]"][str(h2)][key]
                 - results[f"static_train[{h2}]"][str(h2)][key])


summary = {}
for h2 in HORIZONS:
    row = {}
    for h1 in HORIZONS:
        g = gain(h1, h2)
        g_ref = gain(h2, h2)
        gm = gain(h1, h2, macro=True)
        gm_ref = gain(h2, h2, macro=True)
        row[str(h1)] = {"transfer_gain": g, "matched_gain": g_ref,
                        "recovery_vs_matched": g / g_ref if g_ref else None,
                        "transfer_gain_macro": gm, "matched_gain_macro": gm_ref,
                        "recovery_vs_matched_macro": gm / gm_ref if gm_ref else None}
    summary[str(h2)] = row

results["_summary_old_definition"] = results.pop("_summary")
results["_summary"] = summary
(HERE / "results.json").write_text(json.dumps(results, indent=2))
print("primary 1->90:", round(gain("1", "90") / gain("90", "90"), 3),
      "| macro:", round(gain("1", "90", True) / gain("90", "90", True), 3))
print("primary 90->1:", round(gain("90", "1") / gain("1", "1"), 3),
      "| macro:", round(gain("90", "1", True) / gain("1", "1", True), 3))
print("saved", HERE / "results.json")
