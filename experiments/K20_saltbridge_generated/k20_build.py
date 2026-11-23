"""K20 build: K15's expanded salt-bridge candidates (DocD definitions)
+ proposals 11-14 generated minimum-distance tracks (K11 protocol,
same as K17). Real side rebuilt with K15's extract.py definitions and
verified array-exact against the K15 cache. Trajectories are read from
zarr directly (orig 6 from gpt-model-check data, the rest from the
K15 atlas workspace). Boxes handled per axis (cell + _mic)."""
import os, sys, json, time
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
import minitopol  # registers the minitopol_agx3 zarr codec
from pathlib import Path
import numpy as np, pandas as pd, zarr
from tseries.data import cell
from tseries.features import _mic
from tseries.hb_elastic_paths import elastic_modes, mode_paths, local_paths

GPT_BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
ATLAS_WS = Path("/home/hatch/workspace/kinescope/work/k15_atlas")
CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
OUT = Path("/home/hatch/workspace/k20/out")
_SPLIT = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion/split.json").read_text())
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
TRAIN_P = ORIG_TRAIN + tuple(_SPLIT["train_additions"])
TEST_P = ORIG_TEST + tuple(_SPLIT["test_additions"])
ALL_P = TRAIN_P + TEST_P
HORIZON_FRAMES = (20, 60, 110, 310, 510, 910)
READ_FRAMES = list(range(21)) + [60, 110, 310, 510, 910]
FRAMES5 = (10, 11, 13, 15, 20)
SEED = 506166

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

def reduced_series(pos, box, a, b, k, n_pairs, frames):
    out = np.empty((len(frames), n_pairs), dtype=np.float64)
    for fi, f in enumerate(frames):
        vals = distance(pos[f, a, :] - pos[f, b, :], box[f])
        cur = np.full(n_pairs, np.inf)
        np.minimum.at(cur, k, vals)
        out[fi] = cur
    return out

def min_tracks_from_positions(pos10, boxv, disp, a, b, k, n_pairs):
    out = np.empty((4, 8, n_pairs), dtype=np.float64)
    for r in range(4):
        for t in range(8):
            newpos = pos10 + disp[:, r, t]
            vals = np.linalg.norm(_mic(newpos[b] - newpos[a], boxv), axis=1)
            cur = np.full(n_pairs, np.inf)
            np.minimum.at(cur, k, vals)
            out[r, t] = cur
    return out

def focused_tracks(pos10, boxv, modes, values, signal, owner, a, b, k, n_pairs):
    out = np.full((4, 8, n_pairs), np.inf)
    base_all = _mic(pos10[b] - pos10[a], boxv)
    for s in range(0, len(a), 8192):
        aa, bb = a[s:s + 8192], b[s:s + 8192]
        diff = modes[owner[bb]] - modes[owner[aa]]
        base = base_all[s:s + 8192]
        dist = np.linalg.norm(base, axis=1)
        response = np.einsum("ncm,nc->nm", diff, base / np.maximum(dist, 1e-8)[:, None])
        order = np.argsort(np.abs(response) / np.sqrt(values), axis=1)
        selected = np.zeros_like(response, dtype=bool)
        np.put_along_axis(selected, order[:, -min(3, len(values)):], True, axis=1)
        focused = np.einsum("ncm,rtm->nrtc", diff * selected[:, None, :], signal)
        d = np.linalg.norm(base[:, None, None] + focused, axis=-1)
        kk = k[s:s + 8192]
        for r in range(4):
            for t in range(8):
                np.minimum.at(out[r, t], kk, d[:, r, t])
    return out

def run_one(protein, replica):
    d = trajdir(protein, replica)
    df = pd.read_parquet(d / "atoms.parquet")
    z = zarr.open_group(str(d / "A.zarr"), mode='r')
    pos_full = z['positions'][:].astype(np.float64)
    box_full = z['box'][:].astype(np.float64)
    pos = pos_full[READ_FRAMES]; box = box_full[READ_FRAMES]
    idx_of = {f: i for i, f in enumerate(READ_FRAMES)}
    maps, names = layout(df)
    names_arr = np.array(names, dtype=object)
    n = len(maps)
    ca = np.array([m["CA"] for m in maps])
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    t = idx_of[10]
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
    a, b, k = np.concatenate(aL), np.concatenate(bL), np.concatenate(kL)
    fr17 = [idx_of[f] for f in list(range(11)) + list(HORIZON_FRAMES)]
    series17 = reduced_series(pos, box, a, b, k, len(ii), fr17)
    cur = series17[10]
    ri_all = np.array([positive[i][0] for i in ii]); rj_all = np.array([negative[j][0] for j in jj])
    sel = np.flatnonzero((cur > 4.0) & (cur <= 10.0) & (sep[ri_all, rj_all] >= 2))
    ri, rj = ri_all[sel], rj_all[sel]
    hist = series17[:11][:, sel]
    X = np.stack([hist[-1], ca_d[ri, rj], np.log1p(sep[ri, rj]),
                  (names_arr[ri] == "ARG"), (names_arr[rj] == "GLU")], axis=1).astype(np.float32)
    y = (series17[11:][:, sel] <= 4.0).T.astype(np.uint8)
    fr5 = [idx_of[f] for f in FRAMES5]
    fwd5 = reduced_series(pos, box, a, b, k, len(ii), fr5)[:, sel]
    with np.load(CACHE / f"{protein}_R{replica}_salt.npz") as zz:
        ok = (np.array_equal(zz["res_i"], ri) and np.array_equal(zz["res_j"], rj)
              and np.allclose(zz["X"], X, atol=1e-5) and np.allclose(zz["hist"], hist.T, atol=1e-9)
              and np.array_equal(zz["y"], y))
    if not ok:
        raise ValueError(f"K15 cache mismatch for {protein} R{replica}")
    if len(sel) == 0:
        # No candidates (e.g. 4cv4_A R2 has an all-zero box recorded, so
        # K15's distance gives NaN and no candidate survives -- the real
        # side is empty exactly as in K15, and there is nothing to generate).
        out = dict(X=X, hist=hist.T.copy(), y=y, res_i=ri, res_j=rj, fwd5=fwd5.T.copy())
        for num in (11, 12, 13, 14):
            out[f"gen{num}"] = np.empty((4, 9, 0), dtype=np.float32)
        np.savez_compressed(OUT / f"{protein}_R{replica}.npz", **out)
        return f"{protein} R{replica}", 0, 0
    # ---- generation (proposals 11-14), K11/K17 protocol ----
    pos10 = pos[t]; boxv = cell(box[t])
    owner = np.full(len(pos10), -1, dtype=np.intp)
    for i, m in enumerate(maps):
        owner[list(m.values())] = i
    owner = np.maximum(owner, 0)
    ca_pos = pos10[ca]
    values, modes, left, right, blocks_e = elastic_modes(ca_pos, boxv)
    rng = np.random.default_rng([SEED, replica, 10])
    signal = mode_paths(modes, values, rng)
    local0 = local_paths(ca_pos, pos10, owner, boxv, left, right, blocks_e, rng, omega=0.)
    fast = local_paths(ca_pos, pos10, owner, boxv, left, right, blocks_e, rng, omega=.8)
    disp_res = np.einsum("ncm,rtm->nrtc", modes, signal)
    disp11 = disp_res[owner]
    tracks = {}
    tracks[11] = min_tracks_from_positions(pos10, boxv, disp11, a, b, k, len(ii))[:, :, sel]
    tracks[13] = min_tracks_from_positions(pos10, boxv, local0, a, b, k, len(ii))[:, :, sel]
    tracks[14] = min_tracks_from_positions(pos10, boxv, disp11 + .55 * fast, a, b, k, len(ii))[:, :, sel]
    tracks[12] = focused_tracks(pos10, boxv, modes, values, signal, owner, a, b, k, len(ii))[:, :, sel]
    origin = cur[sel]
    out = dict(X=X, hist=hist.T.copy(), y=y, res_i=ri, res_j=rj, fwd5=fwd5.T.copy())
    for num in (11, 12, 13, 14):
        full = np.concatenate([np.broadcast_to(origin, (4, 1, len(sel))), tracks[num]], axis=1)
        out[f"gen{num}"] = full.astype(np.float32)
    np.savez_compressed(OUT / f"{protein}_R{replica}.npz", **out)
    return f"{protein} R{replica}", len(sel), int(y.sum())

def main():
    targets = sys.argv[1:] or list(ALL_P)
    for protein in targets:
        for replica in (1, 2, 3):
            if (OUT / f"{protein}_R{replica}.npz").is_file():
                print("SKIP", protein, replica, flush=True)
                continue
            t0 = time.time()
            tid, nsel, npos = run_one(protein, replica)
            print("DONE", tid, "candidates", nsel, "positives", npos,
                  "sec", round(time.time() - t0, 1), flush=True)

if __name__ == "__main__":
    main()
