"""G79 premise check: do G18's rebuilt method-22 generated series cover
the contact-dissociation candidate set?

G18 rebuilt method 22 (proposal 22) for the HB-dissociation candidates
(K46 cache) with expand_all: every cross-residue donor-acceptor pair
within 6.0 A at the origin, saved as
~/workspace/kinescope/work/g18_gen22/atlas-<prot>-analysis-R<r>.npz
(pair[:, 0] = donor atom index, pair[:, 1] = acceptor atom index,
pair[:, 2] = origin distance, gen_dist = 4-point generated series).

Contact-dissociation candidates (K50 cache,
~/workspace/kinescope/work/k50_cache/<prot>_R<r>_contact_diss.npz)
are residue pairs (res_i, res_j) with the residue-pair minimum distance
<= 4.5 A at the origin. The generated pairs are atom pairs, so the
join is by residue pair: a generated pair's residue pair is
(residue_index[donor atom], residue_index[acceptor atom]) from
atoms.parquet (MiniTopol A format, extracted to /tmp/g79_atoms from
MiniTopol-A-converted-18-trajectories.zip).

The atom->residue map is verified against the K46 cache: donor_res /
acceptor_res match residue_index for all 466 pairs of 1mix_A R1.
The k50 res_i/res_j are 0-based residue_index as well (range 0-204
for 1mix_A with 206 residues).

Counts, per replica: how many dissociation candidates have at least
one generated pair on the same (unordered) residue pair, and how many
of the matched test candidates are positive at 1 ns. Also recorded:
how many generated pairs each matched residue pair has (multiplicity),
because the mixing measurement needs one generated mean per candidate.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
G18 = Path("/home/hatch/workspace/kinescope/work/g18_gen22")
K50 = Path("/home/hatch/workspace/kinescope/work/k50_cache")
ATOMS = Path("/tmp/g79_atoms/trajectories")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")

out = {"per_replica": [], "train": {}, "test": {}}
for group, prots in (("train", TRAIN_P), ("test", TEST_P)):
    cand_total = pos_total = match_total = match_pos_total = 0
    mult = []
    for p in prots:
        a = pd.read_parquet(ATOMS / f"atlas-{p}-analysis-R1" / "atoms.parquet")
        res_idx = a["residue_index"].to_numpy()
        for r in (1, 2, 3):
            g = np.load(G18 / f"atlas-{p}-analysis-R{r}.npz")
            dres = res_idx[g["pair"][:, 0].astype(int)]
            ares = res_idx[g["pair"][:, 1].astype(int)]
            gen_pairs = set()
            gen_count = {}
            for di, ai in zip(dres.tolist(), ares.tolist()):
                key = (min(di, ai), max(di, ai))
                gen_pairs.add(key)
                gen_count[key] = gen_count.get(key, 0) + 1
            k = np.load(K50 / f"{p}_R{r}_contact_diss.npz")
            cand = list(zip(k["res_i"].tolist(), k["res_j"].tolist()))
            y1 = k["y"][:, 0]
            matched = [c for c in cand if (min(c), max(c)) in gen_pairs]
            mpos = int(sum(int(y1[i]) for i, c in enumerate(cand)
                           if (min(c), max(c)) in gen_pairs))
            out["per_replica"].append({
                "protein": p, "replica": r, "group": group,
                "gen_pairs": int(len(gen_pairs)),
                "candidates": len(cand), "matched": len(matched),
                "test_positives_1ns": int(y1.sum()),
                "matched_positives_1ns": mpos,
                "multiplicity_median": float(np.median(
                    [gen_count[(min(c), max(c))] for c in matched])) if matched else None,
                "multiplicity_max": int(max(
                    [gen_count[(min(c), max(c))] for c in matched])) if matched else None,
            })
            cand_total += len(cand)
            pos_total += int(y1.sum())
            match_total += len(matched)
            match_pos_total += mpos
            mult.extend(gen_count[(min(c), max(c))] for c in matched)
    out[group] = {"candidates": cand_total, "matched": match_total,
                  "test_positives_1ns": pos_total,
                  "matched_positives_1ns": match_pos_total,
                  "multiplicity_median": float(np.median(mult)) if mult else None,
                  "multiplicity_max": int(max(mult)) if mult else None}
    print(group, "candidates", cand_total, "matched", match_total,
          "positives", pos_total, "matched positives", match_pos_total, flush=True)

(HERE / "overlap.json").write_text(json.dumps(out, indent=2))
print("saved", HERE / "overlap.json")
