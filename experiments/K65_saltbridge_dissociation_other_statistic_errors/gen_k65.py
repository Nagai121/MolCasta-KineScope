"""K65 build: residue-level generation (proposals 11-14) for the
SALT-BRIDGE DISSOCIATION candidates (K53 conditions, K15 expanded set).

Why this exists: K20's generated series cover only formation
candidates (origin distance > 4.0 A for all 3,727 rows), while
dissociation candidates have origin <= 4.0 A, so they cannot be used
for K65. The coordinator's decision is to redo K65 with the
residue-level method, the same as K54/K58.

Method: K20's build (k20_build.py) generation part, unchanged
(K11/K17/K20 protocol), with the quantity being the residue-pair
minimum charged heavy-atom distance (the salt-bridge definition, as
in K53's extract). From the origin structure (frame 10):
- elastic modes of the C-alpha positions, mode_paths signal with
  rng seed [506166, replica, 10] (consumption order: signal, then
  local_paths omega=0, then omega=0.8 -- same as K20).
- idea 11: residue-level displacement from the modes.
- idea 12: focused tracks (per atom-pair mode selection, the
  original method).
- idea 13: local_paths with omega=0.
- idea 14: idea 11 + 0.55 * local_paths with omega=0.8.
For each candidate residue pair the generated minimum distance is
computed at 8 steps; the origin distance is prepended, so each idea
gives (4 branches, 9 points, N). Positive residues are LYS (NZ) and
ARG (NE, NH1, NH2); negative residues are ASP (OD1, OD2) and GLU
(OE1, OE2) (same as K15/K20/K53). Distances use the per-frame box,
axis-wise (cell + _mic).

Candidates: K53's dissociation candidates (res_i, res_j per
trajectory; origin minimum distance <= 4.0 A) on the K15 expanded
set (88 proteins x 3 replicas = 264 trajectories). The real 11-frame
history is rebuilt and verified against the K53 cache (allclose),
and the real forward 5-point series (frames 10/11/13/15/20) is
saved as fwd5 for the quality check. Trajectories with no
candidates (4cv4_A R2, 2hng_A R3) get empty outputs, as in K20.
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
ATLAS_WS = Path("/home/hatch/workspace/kinescope/work/k15_atlas")
CACHE53 = Path("/home/hatch/workspace/kinescope/work/k53_cache")
OUT = Path("/home/hatch/workspace/kinescope/work/k65_gen_residue")
OUT.mkdir(parents=True, exist_ok=True)
_SPLIT = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion/split.json").read_text())
ORIG_TRAIN = tuple(_SPLIT["orig_train"])
ORIG_TEST = tuple(_SPLIT["orig_test"])
TRAIN_P = ORIG_TRAIN + tuple(_SPLIT["train_additions"])
TEST_P = ORIG_TEST + tuple(_SPLIT["test_additions"])
ALL_P = TRAIN_P + TEST_P
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
    tid = f"atlas-{protein}-analysis-R{replica}"
    d = trajdir(protein, replica)
    df = pd.read_parquet(d / "atoms.parquet")
    z = zarr.open_group(str(d / "A.zarr"), mode='r')
    pos_full = z['positions'][:].astype(np.float64)
    box_full = z['box'][:].astype(np.float64)
    assert pos_full.shape[0] == 1001, pos_full.shape
    maps, names = layout(df)
    n = len(maps)
    ca = np.array([m["CA"] for m in maps])
    with np.load(CACHE53 / f"{protein}_R{replica}_salt_diss.npz") as zz:
        ri, rj = zz["res_i"], zz["res_j"]
        hist_cache = zz["hist"].astype(np.float64)  # (N, 11)
    N = len(ri)
    if N == 0:
        # No candidates (e.g. 4cv4_A R2 has an all-zero box recorded) --
        # the real side is empty exactly as in K53, and there is nothing
        # to generate. Same handling as K20.
        out = dict(res_i=ri, res_j=rj, fwd5=np.empty((0, 5), dtype=np.float32))
        for num in (11, 12, 13, 14):
            out[f"gen{num}"] = np.empty((4, 9, 0), dtype=np.float32)
        np.savez_compressed(OUT / f"{protein}_R{replica}.npz", **out)
        return dict(tid=tid, n=0)
    positive, negative = [], []
    for kk, (m, rname) in enumerate(zip(maps, names)):
        if rname == "LYS" and "NZ" in m:
            positive.append((kk, np.array([m["NZ"]])))
        if rname == "ARG":
            aa = np.array([m[a] for a in ("NE", "NH1", "NH2") if a in m], dtype=int)
            if len(aa): positive.append((kk, aa))
        if rname in ("ASP", "GLU"):
            aa = np.array([m[a] for a in (("OD1", "OD2") if rname == "ASP" else ("OE1", "OE2")) if a in m], dtype=int)
            if len(aa): negative.append((kk, aa))
    pos_idx = {kk: a for kk, a in positive}
    neg_idx = {kk: a for kk, a in negative}
    aL, bL, kL = [], [], []
    for zz, (i, j) in enumerate(zip(ri, rj)):
        ia, ib = pos_idx[int(i)], neg_idx[int(j)]
        aL.append(np.repeat(ia, len(ib))); bL.append(np.tile(ib, len(ia)))
        kL.append(np.full(len(ia) * len(ib), zz, dtype=np.int32))
    a, b, k = np.concatenate(aL), np.concatenate(bL), np.concatenate(kL)
    # real side: rebuild the 11-frame history and verify against K53
    hist11 = reduced_series(pos_full, box_full, a, b, k, N, list(range(11))).T  # (N, 11)
    assert np.allclose(hist_cache, hist11, atol=1e-9), (protein, replica, np.abs(hist_cache - hist11).max())
    origin = hist11[:, -1]
    fwd5 = reduced_series(pos_full, box_full, a, b, k, N, FRAMES5).T  # (N, 5)
    assert np.abs(fwd5[:, 0] - origin).max() < 1e-9
    # ---- generation (proposals 11-14), K11/K17/K20 protocol ----
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
    tracks = {}
    tracks[11] = min_tracks_from_positions(pos10, boxv, disp11, a, b, k, N)
    tracks[13] = min_tracks_from_positions(pos10, boxv, local0, a, b, k, N)
    tracks[14] = min_tracks_from_positions(pos10, boxv, disp11 + .55 * fast, a, b, k, N)
    tracks[12] = focused_tracks(pos10, boxv, modes, values, signal, owner, a, b, k, N)
    out = dict(res_i=ri, res_j=rj, fwd5=fwd5.astype(np.float32))
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
    for protein in ALL_P:
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
