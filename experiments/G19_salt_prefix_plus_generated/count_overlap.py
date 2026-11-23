"""G19 premise check: how many K15 salt-formation candidates have a matching
residue pair in K17's contact-candidate set?

Counts, per trajectory file, the K15 salt-formation candidates whose
(res_i, res_j) residue pair appears among the K17 contact candidates of
the same trajectory (same protein, same replica). K17's generated series
exist only for the 6 DocD proteins, so the expanded 88-protein set has
no generated series at all; the overlap is counted on the 6 proteins.

Run: ~/workspace/mt-venv/bin/python count_overlap.py
Expected output:
  train6: salt candidates 251, matched 38 (15.1%), positives@1ns among matched 1
  test6:  salt candidates 116, matched 20 (17.2%), positives@1ns among matched 2
  train88: salt candidates 2557, positives@1ns 221
  test88:  salt candidates 1170, positives@1ns 93
"""
import json
from pathlib import Path

import numpy as np

K15 = Path("/home/hatch/workspace/kinescope/work/k15_cache")
K17 = Path("/home/hatch/workspace/k17/out")
SPLIT = Path(__file__).resolve().parent.parent / "K15_saltbridge_expansion" / "split.json"

split = json.loads(SPLIT.read_text())
train6 = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
test6 = ("1mix_A", "2qt4_A")
train88 = train6 + tuple(split["train_additions"])
test88 = test6 + tuple(split["test_additions"])


def count_overlap(prots):
    tot = matched = pos = 0
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K15 / f"{p}_R{r}_salt.npz")
            k = np.load(K17 / f"{p}_R{r}.npz")
            contact = set(zip(k["res_i"].tolist(), k["res_j"].tolist()))
            m = np.array([(int(i), int(j)) in contact
                          for i, j in zip(z["res_i"], z["res_j"])])
            tot += len(z["res_i"])
            matched += int(m.sum())
            pos += int(z["y"][m, 0].sum())
    return tot, matched, pos


def count_all(prots):
    tot = pos = 0
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(K15 / f"{p}_R{r}_salt.npz")
            tot += len(z["res_i"])
            pos += int(z["y"][:, 0].sum())
    return tot, pos


for label, prots in (("train6", train6), ("test6", test6)):
    tot, matched, pos = count_overlap(prots)
    print(f"{label}: salt candidates {tot}, matched {matched} "
          f"({100 * matched / tot:.1f}%), positives@1ns among matched {pos}")

for label, prots in (("train88", train88), ("test88", test88)):
    tot, pos = count_all(prots)
    print(f"{label}: salt candidates {tot}, positives@1ns {pos}")
