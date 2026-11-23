"""K15 extraction: salt-bridge candidates only, exactly following
fixed_1ns_history_3phen_6h.py (same definition as K13).

Positive residues LYS(NZ)/ARG(NE,NH1,NH2), negative ASP(OD1,OD2)/GLU(OE1,OE2).
Distance = min over charged heavy atom pairs. Candidates at frame 10 (1 ns):
4.0 < d <= 10.0 and sequence separation >= 2.
Static X (5): [origin min dist, CA dist, log1p(sep), res_i is ARG, res_j is GLU].
Label: min dist <= 4.0 at horizon frames (20,60,110,310,510,910).
History: frames 0..10 raw distance series stored; 6 statistics in run script.
Distances use the per-frame box, axis-wise (diagonal) or full MIC.

Trajectories: original 6 proteins from gpt-model-check/data/trajectories,
the 108 K06(b) proteins from the K15 atlas workspace (acquire.py).
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import minitopol  # registers the minitopol_agx3 zarr codec
import numpy as np, pandas as pd, zarr
from pathlib import Path

GPT_BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
ATLAS_WS = Path("/home/hatch/workspace/kinescope/work/k15_atlas")
OUT = Path("/home/hatch/workspace/kinescope/work/k15_cache")
OUT.mkdir(parents=True, exist_ok=True)

ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
import json as _json
_SPLIT = _json.loads(Path(__file__).with_name("split.json").read_text())
TRAIN_P = ORIG_TRAIN + tuple(_SPLIT["train_additions"])
TEST_P = ORIG_TEST + tuple(_SPLIT["test_additions"])
ALL_P = TRAIN_P + TEST_P
HORIZON_FRAMES = (20, 60, 110, 310, 510, 910)
FRAMES = list(range(11)) + list(HORIZON_FRAMES)


def trajdir(prot, rep):
    tid = f"atlas-{prot}-analysis-R{rep}"
    if prot in ORIG_TRAIN + ORIG_TEST:
        return GPT_BASE / tid
    return ATLAS_WS / prot / "trajectories" / tid


def distance(v, box):
    box = np.asarray(box, dtype=np.float64)
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-5):
        lengths = np.diag(box); v = v - lengths * np.rint(v / lengths)
    else:
        inv = np.linalg.inv(box); fr = v @ inv; fr -= np.rint(fr); v = fr @ box
    return np.linalg.norm(v, axis=-1)


def layout(df):
    order = list(dict.fromkeys(df["residue_index"].tolist()))
    maps, names = [], []
    for r in order:
        idx = np.flatnonzero(df["residue_index"].values == r)
        nm = {df["name"].values[i]: int(i) for i in idx}
        if not {"N", "CA", "C", "O"}.issubset(nm):
            continue
        maps.append(nm); names.append(df["residue_name"].values[idx[0]])
    return maps, names


def reduced_series(pos, box, a, b, k, n_pairs):
    out = np.empty((len(FRAMES), n_pairs), dtype=np.float64)
    for fi in range(len(FRAMES)):
        vals = distance(pos[fi, a, :] - pos[fi, b, :], box[fi])
        cur = np.full(n_pairs, np.inf)
        np.minimum.at(cur, k, vals)
        out[fi] = cur
    return out


def build(prot, rep):
    d = trajdir(prot, rep)
    df = pd.read_parquet(d / "atoms.parquet")
    z = zarr.open_group(str(d / "A.zarr"), mode='r')
    try:
        pos = z['positions'][FRAMES].astype(np.float64)
        box = z['box'][FRAMES].astype(np.float64)
    except Exception:
        pos_full = z['positions'][:].astype(np.float64)
        box_full = z['box'][:].astype(np.float64)
        pos, box = pos_full[FRAMES], box_full[FRAMES]
    maps, names = layout(df)
    names_arr = np.array(names, dtype=object)
    n = len(maps)
    ca = np.array([m["CA"] for m in maps])
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    t = 10  # index of frame 10 within FRAMES
    ca_d = distance(pos[t, ca[:, None], :] - pos[t, ca[None, :], :], box[t])

    positive, negative = [], []
    for kk, (m, r) in enumerate(zip(maps, names)):
        if r == "LYS" and "NZ" in m:
            positive.append((kk, np.array([m["NZ"]])))
        if r == "ARG":
            aa = np.array([m[a] for a in ("NE", "NH1", "NH2") if a in m], dtype=int)
            if len(aa): positive.append((kk, aa))
        if r in ("ASP", "GLU"):
            aa = np.array([m[a] for a in (("OD1", "OD2") if r == "ASP" else ("OE1", "OE2")) if a in m], dtype=int)
            if len(aa): negative.append((kk, aa))
    pi, pj = np.indices((len(positive), len(negative)))
    ii, jj = pi.ravel(), pj.ravel()
    pp = [a for _, a in positive]; nn = [a for _, a in negative]
    aL, bL, kL = [], [], []
    for zz, (i, j) in enumerate(zip(ii, jj)):
        aL.append(np.repeat(pp[i], len(nn[j]))); bL.append(np.tile(nn[j], len(pp[i])))
        kL.append(np.full(len(pp[i]) * len(nn[j]), zz, dtype=np.int32))
    a = np.concatenate(aL); b = np.concatenate(bL); k = np.concatenate(kL)
    all_series = reduced_series(pos, box, a, b, k, len(ii))
    cur = all_series[t]
    ri_all = np.array([positive[i][0] for i in ii]); rj_all = np.array([negative[j][0] for j in jj])
    sel = np.flatnonzero((cur > 4.0) & (cur <= 10.0) & (sep[ri_all, rj_all] >= 2))
    ri, rj = ri_all[sel], rj_all[sel]
    hist = all_series[:11][:, sel]
    x = np.stack([hist[-1], ca_d[ri, rj], np.log1p(sep[ri, rj]),
                  (names_arr[ri] == "ARG"), (names_arr[rj] == "GLU")], axis=1).astype(np.float32)
    y = (all_series[11:][:, sel] <= 4.0).T.astype(np.uint8)
    np.savez_compressed(OUT / f"{prot}_R{rep}_salt.npz", X=x, hist=hist.T.copy(), y=y, res_i=ri, res_j=rj)
    print(prot, "R", rep, "salt cand", len(sel), "pos/horizon", y.sum(0), flush=True)


if __name__ == "__main__":
    targets = sys.argv[1:] or list(ALL_P)
    for p in targets:
        for r in (1, 2, 3):
            if (OUT / f"{p}_R{r}_salt.npz").exists():
                continue
            build(p, r)
    print("done")
