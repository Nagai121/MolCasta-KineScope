"""K05 real-side baseline: HB candidate-pair distance series statistics.

Origin = frame 10 (1 ns), candidates 3.5 < d(origin) <= 6.0 A, donor-acceptor
heavy-atom pairs on different residues (same definition as Doc A / Doc D HB).
Backward history = frames 0..10, forward continuation = frames 10..20,
forward subsample = origin +1/+3/+5/+10 frames (0.1/0.3/0.5/1.0 ns, the
teacher time grid of generated proposals 21-25).
Data: MiniTopol A trajectories on this VM (gpt-model-check/data).
"""
import json
from pathlib import Path
import numpy as np
from minitopol import Workspace
from minitopol.registry import Registry

DONOR_N = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"},
           "GLN": {"NE2"}, "HIS": {"ND1", "NE2"}, "TRP": {"NE1"}}
DONOR_O = {"SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}}
ACCEPTOR_O = {"ASN": {"OD1"}, "GLN": {"OE1"}, "ASP": {"OD1", "OD2"},
              "GLU": {"OE1", "OE2"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}}

def distance(v, box):
    box = np.asarray(box, dtype=np.float64)
    if not np.isfinite(box).all() or abs(np.linalg.det(box)) < 1e-9:
        return np.linalg.norm(v, axis=-1)
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-5):
        lengths = np.diag(box)
        v = v - lengths * np.rint(v / lengths)
    else:
        inv = np.linalg.inv(box)
        frac = v @ inv
        frac -= np.rint(frac)
        v = frac @ box
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
        maps.append(here)
        names.append(data["residue_name"][int(idx[0])])
    donors, acceptors = [], []
    for k, (m, r) in enumerate(zip(maps, names)):
        if r != "PRO":
            donors.append((m["N"], k))
        for name in DONOR_N.get(r, ()):
            if name in m: donors.append((m[name], k))
        for name in DONOR_O.get(r, ()):
            if name in m: donors.append((m[name], k))
        acceptors.append((m["O"], k))
        if "OXT" in m: acceptors.append((m["OXT"], k))
        for name in ACCEPTOR_O.get(r, ()):
            if name in m: acceptors.append((m[name], k))
    return donors, acceptors

def series_stats(series, bound=3.5):
    """series: (T, N) distances. Returns dict of pooled/per-pair statistics."""
    T, N = series.shape
    below = series <= bound
    # per-pair lag-1 autocorrelation (pairs with nonzero variance)
    x = series - series.mean(axis=0, keepdims=True)
    var = (x * x).sum(axis=0)
    ok = var > 1e-12
    lag1 = np.full(N, np.nan)
    if ok.any():
        lag1[ok] = (x[:-1, ok] * x[1:, ok]).sum(axis=0) / var[ok]
    crossings = (np.diff(below.astype(np.int8), axis=0) != 0).sum(axis=0)
    # dwell: mean run length of below-bound episodes (in frames), over pairs with any below frame
    dwell = []
    for j in range(N):
        col = below[:, j]
        if not col.any():
            continue
        runs, cur = [], 0
        for v in col:
            if v: cur += 1
            elif cur: runs.append(cur); cur = 0
        if cur: runs.append(cur)
        dwell.append(np.mean(runs))
    incr = np.diff(series, axis=0)
    return {
        "n_pairs": int(N), "n_frames": int(T),
        "dist_mean": float(series.mean()), "dist_std": float(series.std()),
        "dist_q10_q50_q90": [float(v) for v in np.quantile(series, [0.1, 0.5, 0.9])],
        "per_pair_std_mean": float(series.std(axis=0).mean()),
        "per_pair_range_mean": float((series.max(axis=0) - series.min(axis=0)).mean()),
        "lag1_autocorr_mean": float(np.nanmean(lag1)) if ok.any() else None,
        "lag1_autocorr_median": float(np.nanmedian(lag1)) if ok.any() else None,
        "increment_std": float(incr.std()), "increment_abs_mean": float(np.abs(incr).mean()),
        "frac_frames_below": float(below.mean()),
        "frac_pairs_ever_below": float(below.any(axis=0).mean()),
        "crossings_per_pair_mean": float(crossings.mean()),
        "frac_pairs_with_crossing": float((crossings > 0).mean()),
        "dwell_below_frames_mean": float(np.mean(dwell)) if dwell else None,
    }

def main():
    ws = Workspace(Path.home() / "workspace/gpt-model-check/data")
    reg = Registry(ws)
    proteins = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
    out = {"definition": "origin frame10 (1ns); 3.5<d<=6.0; donor-acceptor, diff residue",
           "per_trajectory": {}, "pooled": {}}
    pool_hist, pool_fwd, pool_sub = [], [], []
    labels = []
    for protein in proteins:
        for replica in (1, 2, 3):
            tid = f"atlas-{protein}-analysis-R{replica}"
            data = reg.atoms(tid).to_pydict()
            donors, acceptors = atom_layout(data)
            d_at = np.array([d[0] for d in donors]); d_res = np.array([d[1] for d in donors])
            a_at = np.array([a[0] for a in acceptors]); a_res = np.array([a[1] for a in acceptors])
            blocks = list(reg.blocks(tid, start=0, stop=21, chunk_frames=21))
            pos = np.concatenate([b.positions for b in blocks], axis=0)
            box = np.concatenate([b.box for b in blocks], axis=0)
            t = 10
            dd = distance(pos[t, d_at[:, None], :] - pos[t, a_at[None, :], :], box[t])
            ii, jj = np.where((dd > 3.5) & (dd <= 6.0) & (d_res[:, None] != a_res[None, :]))
            a, b = d_at[ii], a_at[jj]
            series = np.stack([distance(pos[f, a] - pos[f, b], box[f]) for f in range(21)])
            hist, fwd = series[:11], series[10:]
            sub = series[[10, 11, 13, 15, 20]]
            y = (series[20] <= 3.5)
            out["per_trajectory"][tid] = {
                "history_0to1ns": series_stats(hist),
                "forward_0to1ns_full": series_stats(fwd),
                "forward_subsample_0.1_0.3_0.5_1.0": series_stats(sub),
                "formation_rate_1ns": float(y.mean()),
            }
            pool_hist.append(hist); pool_fwd.append(fwd); pool_sub.append(sub); labels.append(y)
            print(tid, "pairs:", len(a), "form@1ns:", round(float(y.mean()), 4), flush=True)
    out["pooled"] = {
        "history_0to1ns": series_stats(np.concatenate(pool_hist, axis=1)),
        "forward_0to1ns_full": series_stats(np.concatenate(pool_fwd, axis=1)),
        "forward_subsample_0.1_0.3_0.5_1.0": series_stats(np.concatenate(pool_sub, axis=1)),
        "formation_rate_1ns": float(np.concatenate(labels).mean()),
        "total_pairs": int(sum(s.shape[1] for s in pool_hist)),
    }
    Path("/tmp/k05_real_baseline.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["pooled"], indent=1))

if __name__ == "__main__":
    main()
