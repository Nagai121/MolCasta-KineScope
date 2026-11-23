"""G64 premise check: do G19's rebuilt generated series cover the
salt-bridge dissociation candidates?

Counts, per trajectory file, the K53 dissociation candidates whose
(res_i, res_j) residue pair appears in the G19 generated-series files
(experiments/G19_salt_prefix_plus_generated/gen/), which were built
for the K15 salt-bridge FORMATION candidates.

Run: ~/workspace/mt-venv/bin/python count_overlap.py
Expected output:
  train: dissociation candidates 735, matched in G19 gen set 0, positives@1ns among matched 0
  test:  dissociation candidates 360, matched in G19 gen set 0, positives@1ns among matched 0
The sets are disjoint by construction: formation candidates have
origin min distance 4.0 < d <= 10.0 (unbound), dissociation candidates
have origin min distance <= 4.0 (bound).
"""
import json
from pathlib import Path

import numpy as np

K53 = Path("/home/hatch/workspace/kinescope/work/k53_cache")
GEN = Path(__file__).resolve().parent.parent / "G19_salt_prefix_plus_generated" / "gen"
SPLIT = Path(__file__).resolve().parent.parent / "K15_saltbridge_expansion" / "split.json"

split = json.loads(SPLIT.read_text())
train_p = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A") + tuple(split["train_additions"])
test_p = ("1mix_A", "2qt4_A") + tuple(split["test_additions"])

for label, prots in (("train", train_p), ("test", test_p)):
    tot = matched = pos_m = 0
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K53 / f"{p}_R{r}_salt_diss.npz")
            g = np.load(GEN / f"{p}_R{r}.npz")
            gen_pairs = set(zip(g["res_i"].tolist(), g["res_j"].tolist()))
            m = np.array([(int(i), int(j)) in gen_pairs
                          for i, j in zip(z["res_i"], z["res_j"])])
            tot += len(z["res_i"])
            matched += int(m.sum())
            pos_m += int(z["y"][m, 0].sum()) if m.any() else 0
    print(f"{label}: dissociation candidates {tot}, matched in G19 gen set {matched}, "
          f"positives@1ns among matched {pos_m}")
