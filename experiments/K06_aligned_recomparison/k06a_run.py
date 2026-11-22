"""K06(a): Doc D-aligned re-comparison for HB formation.

Same candidates, same static features (7), same HistGradientBoosting settings
and same train/test protein split as the Doc D experiment
(fixed_1ns_history_3phen_6h.py). Only the history block changes:
  S        static only
  S+H11    static + real backward history, 6 features from 11 frames (Doc D)
  S+H5     static + real backward history, 6 features from 5 frames
           (frames 0,3,5,7,10; offsets -1.0,-0.7,-0.5,-0.3,0 ns)
  S+G6     static + generated proposal-22 series, same 6 features from
           5 points (origin + 0.1/0.3/0.5/1.0 ns, forward)
  S+G8     static + generated proposal-22 native 8 summary features
  S+R5f    static + REAL forward series, 6 features from 5 points
           (oracle reference only: uses the actual future; at horizon 1 ns
           the +1.0 ns point defines the label, so that cell is a ceiling)
"""
import json
from pathlib import Path
import numpy as np
from minitopol import Workspace
from minitopol.registry import Registry
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

DONOR_N = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"},
           "GLN": {"NE2"}, "HIS": {"ND1", "NE2"}, "TRP": {"NE1"}}
DONOR_O = {"SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}}
ACCEPTOR_O = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}, "ASN": {"OD1"},
              "GLN": {"OE1"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}}
HORIZONS = (1, 5, 10, 30, 50, 90)
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
TRAIN = PROTEINS[:4]

def distance(v, box):
    box = np.asarray(box, dtype=np.float64)
    if not np.isfinite(box).all() or abs(np.linalg.det(box)) < 1e-9:
        return np.linalg.norm(v, axis=-1)
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-5):
        lengths = np.diag(box)
        v = v - lengths * np.rint(v / lengths)
    else:
        inv = np.linalg.inv(box); frac = v @ inv; frac -= np.rint(frac); v = frac @ box
    return np.linalg.norm(v, axis=-1)

def atom_layout(data):
    ridx = np.asarray(data["residue_index"])
    order = list(dict.fromkeys(ridx.tolist()))
    maps, names = [], []
    for residue in order:
        idx = np.flatnonzero(ridx == residue)
        here = {data["name"][i]: int(i) for i in idx}
        if not {"N", "CA", "C", "O"}.issubset(here):
            continue
        maps.append(here); names.append(data["residue_name"][int(idx[0])])
    donors, acceptors = [], []
    for k, (m, r) in enumerate(zip(maps, names)):
        if r != "PRO": donors.append((m["N"], k, 0))
        for name in DONOR_N.get(r, ()):
            if name in m: donors.append((m[name], k, 1))
        for name in DONOR_O.get(r, ()):
            if name in m: donors.append((m[name], k, 2))
        acceptors.append((m["O"], k, 0))
        if "OXT" in m: acceptors.append((m["OXT"], k, 0))
        for name in ACCEPTOR_O.get(r, ()):
            if name in m: acceptors.append((m[name], k, 1))
    return maps, donors, acceptors

def hist6(series, bound=3.5):
    return np.stack([series[0], series.mean(0), series.std(0), series.min(0),
                     series[-1] - series[0], (series <= bound).mean(0)], axis=1).astype(np.float32)

def build(reg, protein, replica):
    tid = f"atlas-{protein}-analysis-R{replica}"
    data = reg.atoms(tid).to_pydict()
    maps, donors, acceptors = atom_layout(data)
    blocks = list(reg.blocks(tid, start=0, stop=911, chunk_frames=256))
    pos = np.concatenate([b.positions for b in blocks], axis=0)
    box = np.concatenate([b.box for b in blocks], axis=0)
    t = 10
    d_at = np.array([d[0] for d in donors]); d_res = np.array([d[1] for d in donors]); d_kind = np.array([d[2] for d in donors])
    a_at = np.array([a[0] for a in acceptors]); a_res = np.array([a[1] for a in acceptors]); a_kind = np.array([a[2] for a in acceptors])
    dd = distance(pos[t, d_at[:, None], :] - pos[t, a_at[None, :], :], box[t])
    ii, jj = np.where((dd > 3.5) & (dd <= 6.0) & (d_res[:, None] != a_res[None, :]))
    a, b = d_at[ii], a_at[jj]
    ca = np.array([m["CA"] for m in maps], dtype=int)
    n = len(maps)
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    ca_d = distance(pos[t, ca[:, None], :] - pos[t, ca[None, :], :], box[t])
    x = np.stack([dd[ii, jj], ca_d[d_res[ii], a_res[jj]], np.log1p(sep[d_res[ii], a_res[jj]]),
                  (d_kind[ii] == 0), (a_kind[jj] == 0), (d_kind[ii] == 2), (a_kind[jj] == 1)],
                 axis=1).astype(np.float32)
    hist = np.stack([distance(pos[f, a] - pos[f, b], box[f]) for f in range(11)])
    y = np.column_stack([distance(pos[t + 10 * h, a] - pos[t + 10 * h, b], box[t + 10 * h]) <= 3.5
                         for h in HORIZONS]).astype(np.uint8)
    with np.load(f"/tmp/gen22_out/{tid}.npz") as z:
        pair, gen, gen8 = z["pair"], z["gen_dist"], z["gen8"]
        real_sub = z["real_sub"]
    key = {(int(p[0]), int(p[1])): k for k, p in enumerate(pair)}
    idx = np.array([key[(int(u), int(v))] for u, v in zip(a, b)])
    assert len(idx) == len(a)
    gen5 = np.column_stack([dd[ii, jj], gen[idx]]).T          # (5, N) generated forward
    real5f = np.column_stack([dd[ii, jj], real_sub[idx]]).T   # (5, N) real forward
    return dict(x=x, h11=hist6(hist), h5=hist6(hist[[0, 3, 5, 7, 10]]),
                g6=hist6(gen5), g8=gen8[idx].astype(np.float32), r5f=hist6(real5f),
                y=y, protein=protein)

def metrics(y, p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return dict(auc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, p, labels=[0, 1])))

def main():
    reg = Registry(Workspace(Path.home() / "workspace/gpt-model-check/data"))
    parts = {}
    for protein in PROTEINS:
        for replica in (1, 2, 3):
            d = build(reg, protein, replica)
            parts.setdefault(protein, []).append(d)
            print(protein, replica, len(d["y"]), flush=True)
    conds = {"S": lambda d: d["x"],
             "S+H11": lambda d: np.column_stack([d["x"], d["h11"]]),
             "S+H5": lambda d: np.column_stack([d["x"], d["h5"]]),
             "S+G6": lambda d: np.column_stack([d["x"], d["g6"]]),
             "S+G8": lambda d: np.column_stack([d["x"], d["g8"]]),
             "S+R5f": lambda d: np.column_stack([d["x"], d["r5f"]])}
    results = {}
    for name, fn in conds.items():
        Xtr = np.concatenate([fn(d) for p in TRAIN for d in parts[p]])
        Xte = np.concatenate([fn(d) for p in PROTEINS[4:] for d in parts[p]])
        ytr = np.concatenate([d["y"] for p in TRAIN for d in parts[p]])
        yte = np.concatenate([d["y"] for p in PROTEINS[4:] for d in parts[p]])
        results[name] = {}
        for j, h in enumerate(HORIZONS):
            m = HistGradientBoostingClassifier(max_iter=120, learning_rate=.075,
                                               max_leaf_nodes=31, min_samples_leaf=15, l2_regularization=.02,
                                               random_state=42)
            m.fit(Xtr, ytr[:, j])
            results[name][f"{h}ns"] = metrics(yte[:, j], m.predict_proba(Xte)[:, 1])
        print(name, "done", flush=True)
    Path("/tmp/k06a_results.json").write_text(json.dumps(results, indent=1))
    hdr = ["cond"] + [f"{h}ns" for h in HORIZONS]
    print("AUPRC table")
    print(" | ".join(hdr))
    for name, r in results.items():
        print(name, "|", " | ".join(f"{r[f'{h}ns']['auprc']:.4f}" for h in HORIZONS))

if __name__ == "__main__":
    main()
