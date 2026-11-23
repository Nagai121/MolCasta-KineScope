"""K50 extraction: CONTACT DISSOCIATION candidates (data D conditions).

Mirror image of K13's contact formation, following K46's dissociation
logic (fixed before seeing results):
  formation (K13): residue pairs i<j with CA dist <= 12.0 A and
    sequence separation >= 2; candidates at frame 10 (1 ns) have
    min heavy-atom distance > 4.5 A; label = min dist <= 4.5 A at horizon.
  dissociation (this file): same residue-pair rule; candidates at
    frame 10 have min heavy-atom distance <= 4.5 A (contact formed at
    the origin); label = min dist > 4.5 A at the horizon frame
    (1 = dissociated). Formation and dissociation candidates do not
    overlap; together they are all residue pairs with CA <= 12.0 A
    and sep >= 2.
Static X (5, same as formation): [origin min dist, CA dist, log1p(sep),
n_heavy_i, n_heavy_j]. History: frames 0..10 raw minimum-distance
series are stored; the 6 statistics are computed in run_k50.py exactly
as in K13 (first, mean, std, min, last-first, fraction <= 4.5 A).
Distances use the per-frame box, axis-wise (diagonal) or full MIC.
Heavy atoms are those whose name does not start with H and whose first
character is not a digit (same rule as K13 / the DocD code).
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import minitopol  # registers the minitopol_agx3 zarr codec
import numpy as np, pandas as pd, zarr
from pathlib import Path

BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
OUT = Path("/home/hatch/workspace/kinescope/work/k50_cache")
OUT.mkdir(parents=True, exist_ok=True)
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
HORIZON_FRAMES = (20, 60, 110, 310, 510, 910)
FRAMES = list(range(11)) + list(HORIZON_FRAMES)  # 0..10 history, then horizons
K13 = Path("/home/hatch/workspace/kinescope/work/k13_cache")


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
    n = len(maps)
    ca = np.array([m["CA"] for m in maps])
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    t = 10
    ca_d = distance(pos[t, ca[:, None], :] - pos[t, ca[None, :], :], box[t])
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
    cur = all_series[10]  # frame 10
    sel = np.flatnonzero(cur <= 4.5)
    si, sj = ii[sel], jj[sel]
    hist = all_series[:11][:, sel]  # (11, n_cand)
    x = np.stack([hist[-1], ca_d[si, sj], np.log1p(sep[si, sj]),
                  np.array([len(heavy[i]) for i in si]),
                  np.array([len(heavy[j]) for j in sj])], axis=1).astype(np.float32)
    y = (all_series[11:][:, sel] > 4.5).T.astype(np.uint8)  # (n_cand, 6 horizons)
    out = dict(X=x, hist=hist.T.copy(), y=y, res_i=si, res_j=sj)
    np.savez_compressed(OUT / f"{prot}_R{rep}_contact_diss.npz", **out)
    # check: dissociation + formation candidates = all pairs in the rule
    form = np.load(K13 / f"{prot}_R{rep}_contact.npz")
    assert len(sel) + len(form["hist"]) == len(ii), (tid, len(sel), len(form["hist"]), len(ii))
    assert hist[-1].max() <= 4.5 + 1e-9
    print(tid, "diss cand", len(sel), "formation cand", len(form["hist"]),
          "total pairs", len(ii), "pos/horizon", y.sum(0), flush=True)
    return len(sel)


total = 0
for p in PROTEINS:
    for r in (1, 2, 3):
        total += build(p, r)
print("done total", total)
