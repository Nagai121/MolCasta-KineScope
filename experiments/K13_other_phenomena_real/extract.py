"""K13 extraction: salt-bridge and contact candidates, exactly following
fixed_1ns_history_3phen_6h.py (MolCasta-InternalResearch branch
research/fixed-1ns-3phen-6h-20261121).

Salt: positive residues LYS(NZ)/ARG(NE,NH1,NH2), negative ASP(OD1,OD2)/GLU(OE1,OE2).
Distance = min over charged heavy atom pairs. Candidates at frame 10 (1 ns):
4.0 < d <= 10.0 and sequence separation >= 2.
Static X (5): [origin min dist, CA dist, log1p(sep), res_i is ARG, res_j is GLU].
Label: min dist <= 4.0 at horizon frames.

Contact: residue pairs i<j with CA dist <= 12.0 and sep >= 2.
Distance = min over all heavy atom pairs (name not starting with H and first
char not a digit, same rule as D code's atom_layout-based heavy list used in
the HB base reproduction). Candidates: min dist > 4.5 at frame 10.
Static X (5): [origin min dist, CA dist, log1p(sep), n_heavy_i, n_heavy_j].
Label: min dist <= 4.5 at horizon frames.

History: frames 0..10 raw distance series are stored; the 6 statistics are
computed in run_experiments.py.
Distances use the per-frame box, axis-wise (diagonal) or full MIC.
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import minitopol  # registers the minitopol_agx3 zarr codec
import numpy as np, pandas as pd, zarr
from pathlib import Path

BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
OUT = Path("/home/hatch/workspace/kinescope/work/k13_cache")
OUT.mkdir(parents=True, exist_ok=True)
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
HORIZON_FRAMES = (20, 60, 110, 310, 510, 910)
FRAMES = list(range(11)) + list(HORIZON_FRAMES)  # 0..10 history, then horizons


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


def reduced_series(pos, box, a, b, k, n_pairs, frames):
    """min distance per residue pair for each frame -> (len(frames), n_pairs)"""
    out = np.empty((len(frames), n_pairs), dtype=np.float64)
    for fi, f in enumerate(frames):
        vals = distance(pos[f, a, :] - pos[f, b, :], box[f])
        cur = np.full(n_pairs, np.inf)
        np.minimum.at(cur, k, vals)
        out[fi] = cur
    return out


def build(prot, rep):
    tid = f"atlas-{prot}-analysis-R{rep}"
    d = BASE / tid
    df = pd.read_parquet(d / "atoms.parquet")
    z = zarr.open_group(str(d / "A.zarr"), mode='r')
    pos = z['positions'][:].astype(np.float64)
    box = z['box'][:].astype(np.float64)
    assert pos.shape[0] == 1001, pos.shape
    maps, names = layout(df)
    names_arr = np.array(names, dtype=object)
    n = len(maps)
    ca = np.array([m["CA"] for m in maps])
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    t = 10
    ca_d = distance(pos[t, ca[:, None], :] - pos[t, ca[None, :], :], box[t])
    res = {}

    # ---- salt bridges ----
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
    all_series = reduced_series(pos, box, a, b, k, len(ii), FRAMES)
    cur = all_series[10]  # frame 10 is index 10 in FRAMES
    ri_all = np.array([positive[i][0] for i in ii]); rj_all = np.array([negative[j][0] for j in jj])
    sel = np.flatnonzero((cur > 4.0) & (cur <= 10.0) & (sep[ri_all, rj_all] >= 2))
    ri, rj = ri_all[sel], rj_all[sel]
    hist = all_series[:11][:, sel]  # (11, n_cand)
    x = np.stack([hist[-1], ca_d[ri, rj], np.log1p(sep[ri, rj]),
                  (names_arr[ri] == "ARG"), (names_arr[rj] == "GLU")], axis=1).astype(np.float32)
    y = (all_series[11:][:, sel] <= 4.0).T.astype(np.uint8)  # (n_cand, 6 horizons)
    res["salt"] = dict(X=x, hist=hist.T.copy(), y=y, res_i=ri, res_j=rj)
    print(tid, "salt cand", len(sel), "pos/horizon", y.sum(0), flush=True)

    # ---- contacts ----
    ii, jj = np.where((ca_d <= 12.0) & (sep >= 2) & (np.arange(n)[:, None] < np.arange(n)[None, :]))
    heavy = []
    for m in maps:
        heavy.append(np.asarray([v for key, v in m.items() if not key.startswith("H") and not key[0].isdigit()], dtype=int))
    aL, bL, kL = [], [], []
    for zz, (i, j) in enumerate(zip(ii, jj)):
        ia, ib = heavy[int(i)], heavy[int(j)]
        aL.append(np.repeat(ia, len(ib))); bL.append(np.tile(ib, len(ia)))
        kL.append(np.full(len(ia) * len(ib), zz, dtype=np.int32))
    a = np.concatenate(aL); b = np.concatenate(bL); k = np.concatenate(kL)
    all_series = reduced_series(pos, box, a, b, k, len(ii), FRAMES)
    cur = all_series[10]
    sel = np.flatnonzero(cur > 4.5)
    si, sj = ii[sel], jj[sel]
    hist = all_series[:11][:, sel]
    x = np.stack([hist[-1], ca_d[si, sj], np.log1p(sep[si, sj]),
                  np.array([len(heavy[i]) for i in si]),
                  np.array([len(heavy[j]) for j in sj])], axis=1).astype(np.float32)
    y = (all_series[11:][:, sel] <= 4.5).T.astype(np.uint8)
    res["contact"] = dict(X=x, hist=hist.T.copy(), y=y, res_i=si, res_j=sj)
    print(tid, "contact cand", len(sel), "pos/horizon", y.sum(0), flush=True)

    for ph, dd in res.items():
        np.savez_compressed(OUT / f"{prot}_R{rep}_{ph}.npz", **dd)


for p in PROTEINS:
    for r in (1, 2, 3):
        build(p, r)
print("done")
