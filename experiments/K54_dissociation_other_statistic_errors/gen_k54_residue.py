"""K54 (redo): residue-level generation (proposals 11-14) for the
hydrogen-bond DISSOCIATION candidates.

Why this exists: proposal 22 cannot be used as-is for dissociation.
Its candidate selection keeps only pairs with origin distance > 3.5
A (tseries/rod.py expand()), and its network was trained on
formation pairs only. The first K54 attempt worked around the
selection filter, but the coordinator's decision is to redo K54 with
the residue-level method instead, which has no origin-distance
restriction.

Method: K20's build (k20_build.py) generation part, unchanged
(K11/K17/K20 protocol). From the origin structure (frame 10):
- elastic modes of the C-alpha positions, mode_paths signal with
  rng seed [506166, replica, 10] (consumption order: signal, then
  local_paths omega=0, then omega=0.8 -- same as K20).
- idea 11: residue-level displacement from the modes.
- idea 12: focused tracks (per atom-pair mode selection, the
  original method).
- idea 13: local_paths with omega=0.
- idea 14: idea 11 + 0.55 * local_paths with omega=0.8.
For each candidate donor-acceptor atom pair the generated distance
series is computed at 8 steps; the origin distance is prepended, so
each idea gives (4 branches, 9 points, N). The residue-pair minimum
machinery from K20 is reused with one atom pair per group
(k = arange(N)), so the minimum over a group is the pair itself.
Distances use the per-frame box, axis-wise (cell + _mic).

Candidates: K46's dissociation candidates (donor_atom,
acceptor_atom per trajectory; origin distance <= 3.5 A). The real
forward 5-point series (frames 10/11/13/15/20) is saved as fwd5 for
the quality check. Origin distances are verified against the K46
cache (max error must be ~0).
"""
import sys, json, time
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
import minitopol  # registers the minitopol_agx3 zarr codec
from pathlib import Path
import numpy as np, pandas as pd, zarr
from tseries.data import cell
from tseries.features import _mic
from tseries.hb_elastic_paths import elastic_modes, mode_paths, local_paths

GPT_BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
CACHE46 = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
OUT = Path("/home/hatch/workspace/kinescope/work/k54_gen_residue")
OUT.mkdir(parents=True, exist_ok=True)
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
FRAMES5 = (10, 11, 13, 15, 20)
SEED = 506166


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
    tid = f"atlas-{protein}-analysis-R{replica}"
    d = GPT_BASE / tid
    df = pd.read_parquet(d / "atoms.parquet")
    z = zarr.open_group(str(d / "A.zarr"), mode='r')
    pos_full = z['positions'][:].astype(np.float64)
    box_full = z['box'][:].astype(np.float64)
    assert pos_full.shape[0] == 1001, pos_full.shape
    maps, names = layout(df)
    n = len(maps)
    ca = np.array([m["CA"] for m in maps])
    with np.load(CACHE46 / f"{protein}_R{replica}.npz") as zz:
        donor, acceptor = zz["donor_atom"], zz["acceptor_atom"]
        origin = zz["hist"][-1].astype(np.float64)
    N = len(donor)
    # real forward 5-point series for the quality check
    fwd5 = np.stack([distance(pos_full[f, acceptor, :] - pos_full[f, donor, :], box_full[f])
                     for f in FRAMES5], axis=1)  # (N, 5)
    assert np.abs(fwd5[:, 0] - origin).max() < 1e-9, (protein, replica, np.abs(fwd5[:, 0] - origin).max())
    pos10 = pos_full[10]; boxv = cell(box_full[10])
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
    a, b, k = donor.astype(np.int64), acceptor.astype(np.int64), np.arange(N, dtype=np.int32)
    tracks = {}
    tracks[11] = min_tracks_from_positions(pos10, boxv, disp11, a, b, k, N)
    tracks[13] = min_tracks_from_positions(pos10, boxv, local0, a, b, k, N)
    tracks[14] = min_tracks_from_positions(pos10, boxv, disp11 + .55 * fast, a, b, k, N)
    tracks[12] = focused_tracks(pos10, boxv, modes, values, signal, owner, a, b, k, N)
    out = dict(donor_atom=donor, acceptor_atom=acceptor, fwd5=fwd5.astype(np.float32))
    for num in (11, 12, 13, 14):
        full = np.concatenate([np.broadcast_to(origin, (4, 1, N)), tracks[num]], axis=1)
        out[f"gen{num}"] = full.astype(np.float32)
    np.savez_compressed(OUT / f"{protein}_R{replica}.npz", **out)
    return dict(tid=tid, n=int(N),
                gen_std={str(num): float(tracks[num].std(axis=1).mean()) for num in (11, 12, 13, 14)},
                fwd5_std=float(fwd5.std(axis=1).mean()))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = []
    for protein in PROTEINS:
        for replica in (1, 2, 3):
            tid = f"atlas-{protein}-analysis-R{replica}"
            if only and only not in tid:
                continue
            if (OUT / f"{protein}_R{replica}.npz").is_file():
                print("SKIP", tid, flush=True)
                continue
            t0 = time.time()
            r = run_one(protein, replica)
            r["seconds"] = round(time.time() - t0, 1)
            results.append(r)
            print(json.dumps(r), flush=True)
    (OUT / "build_summary.json").write_text(json.dumps(results, indent=1))
    print("saved to", OUT)


if __name__ == "__main__":
    main()
