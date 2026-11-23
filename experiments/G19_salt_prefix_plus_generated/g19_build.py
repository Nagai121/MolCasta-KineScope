"""G19 build: proposals 11-14 generated series for the K15 salt-bridge
formation candidates, for all 88 proteins of the K15 expanded set.

Generation follows K17's build (experiments/K17_contact_generated/
k17_build.py) exactly: elastic modes on the frame-10 Calpha positions,
rng seeded [506166, replica, 10], consumed in the order mode_paths ->
local_paths(omega=0) -> local_paths(omega=.8); tracks 11/13/14 from
per-atom displacements, track 12 from per-pair mode selection. The
difference from K17 is only the candidate set and the distance: the
candidates are the K15 salt-bridge formation candidates (residue pairs
and order taken from the K15 cache), and the distance is the minimum
over the charged heavy atom pairs (K15 definitions: LYS NZ; ARG NE,
NH1, NH2; ASP OD1, OD2; GLU OE1, OE2).

Trajectories are read in MiniTopol A format (atoms.parquet + A.zarr),
from the same directories K15's extract.py used. Only frame 10 is
needed. Residue maps use K15's layout() verbatim, so the residue
numbering matches the K15 cache by construction.

Checks per replica:
  - the frame-10 real minimum distance over the charged pairs must
    match the K15 cache hist[:, 10] (max abs diff < 1e-6);
  - with --verify-contact (6 DocD proteins), the contact-candidate
    tracks are also built and must match K17's saved gen11-gen14.

Output: gen/<protein>_R<replica>.npz with gen11..gen14 (4 branches,
9 points, N) float32 and res_i/res_j copied from the K15 cache.
Existing outputs are skipped, so the build is resumable per protein.

Run: ~/workspace/mt-venv/bin/python g19_build.py [protein ...]
     (default: all 88 proteins of the K15 split)
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import minitopol  # noqa: F401  (registers the minitopol_agx3 zarr codec)
sys.path.insert(0, "/home/hatch/workspace/gpt-model-check")
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from tseries.data import cell
from tseries.features import _mic
from tseries.hb_elastic_paths import elastic_modes, mode_paths, local_paths

GPT_BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
ATLAS_WS = Path("/home/hatch/workspace/kinescope/work/k15_atlas")
K15_CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
K17_OUT = Path("/home/hatch/workspace/k17/out")
OUT = Path(__file__).resolve().parent / "gen"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = json.loads((Path(__file__).resolve().parent.parent
                    / "K15_saltbridge_expansion" / "split.json").read_text())
ORIG_TRAIN = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
ORIG_TEST = ("1mix_A", "2qt4_A")
ALL_P = ORIG_TRAIN + tuple(SPLIT["train_additions"]) + ORIG_TEST + tuple(SPLIT["test_additions"])
SEED = 506166


def trajdir(prot, rep):
    tid = f"atlas-{prot}-analysis-R{rep}"
    if prot in ORIG_TRAIN + ORIG_TEST:
        return GPT_BASE / tid
    return ATLAS_WS / prot / "trajectories" / tid


def distance(v, box):
    """K15 extract.py, verbatim."""
    box = np.asarray(box, dtype=np.float64)
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-5):
        lengths = np.diag(box); v = v - lengths * np.rint(v / lengths)
    else:
        inv = np.linalg.inv(box); fr = v @ inv; fr -= np.rint(fr); v = fr @ box
    return np.linalg.norm(v, axis=-1)


def layout(df):
    """K15 extract.py, verbatim."""
    order = list(dict.fromkeys(df["residue_index"].tolist()))
    maps, names = [], []
    for r in order:
        idx = np.flatnonzero(df["residue_index"].values == r)
        nm = {df["name"].values[i]: int(i) for i in idx}
        if not {"N", "CA", "C", "O"}.issubset(nm):
            continue
        maps.append(nm); names.append(df["residue_name"].values[idx[0]])
    return maps, names


def charged_atoms(maps, names):
    """residue index -> charged heavy atom indices (K15 definitions)."""
    charged = {}
    for kk, (m, r) in enumerate(zip(maps, names)):
        if r == "LYS" and "NZ" in m:
            charged[kk] = np.array([m["NZ"]])
        elif r == "ARG":
            aa = np.array([m[a] for a in ("NE", "NH1", "NH2") if a in m], dtype=int)
            if len(aa):
                charged[kk] = aa
        elif r in ("ASP", "GLU"):
            aa = np.array([m[a] for a in (("OD1", "OD2") if r == "ASP" else ("OE1", "OE2")) if a in m], dtype=int)
            if len(aa):
                charged[kk] = aa
    return charged


def pair_arrays(pairs_i, pairs_j, atoms_of):
    aL, bL, kL = [], [], []
    for z, (i, j) in enumerate(zip(pairs_i, pairs_j)):
        ia, ib = atoms_of[int(i)], atoms_of[int(j)]
        aL.append(np.repeat(ia, len(ib))); bL.append(np.tile(ib, len(ia)))
        kL.append(np.full(len(ia) * len(ib), z, dtype=np.int32))
    if not aL:
        return (np.empty(0, dtype=int), np.empty(0, dtype=int),
                np.empty(0, dtype=np.int32))
    return np.concatenate(aL), np.concatenate(bL), np.concatenate(kL)


def min_tracks_from_positions(pos10, boxv, disp, a, b, k, n_pairs):
    """K17 k17_build.py, verbatim."""
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
    """K17 k17_build.py, verbatim."""
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


def generation(pos10, box10, maps):
    """K17 protocol: returns (disp11, local0, fast, modes, values, signal, owner)."""
    boxv = cell(box10)
    ca_idx = np.array([m["CA"] for m in maps])
    owner = np.full(len(pos10), -1, dtype=np.intp)
    for i, m in enumerate(maps):
        owner[list(m.values())] = i
    owner = np.maximum(owner, 0)
    ca_pos = pos10[ca_idx]
    values, modes, left, right, blocks_e = elastic_modes(ca_pos, boxv)
    return boxv, ca_idx, owner, values, modes, left, right, blocks_e


def run_one(protein, replica, verify_contact=False):
    d = trajdir(protein, replica)
    df = pd.read_parquet(d / "atoms.parquet")
    z = zarr.open_group(str(d / "A.zarr"), mode="r")
    try:
        pos10 = z["positions"][10].astype(np.float64)
        box10 = z["box"][10].astype(np.float64)
    except Exception:
        pos10 = z["positions"][:].astype(np.float64)[10]
        box10 = z["box"][:].astype(np.float64)[10]
    maps, names = layout(df)
    charged = charged_atoms(maps, names)

    with np.load(K15_CACHE / f"{protein}_R{replica}_salt.npz") as zc:
        res_i = zc["res_i"].astype(np.int64)
        res_j = zc["res_j"].astype(np.int64)
        hist_last = zc["hist"][:, 10].astype(np.float64)
    n_pairs = len(res_i)

    # origin check: frame-10 real min distance over charged pairs
    a, b, k = pair_arrays(res_i, res_j, charged)
    if n_pairs:
        vals = distance(pos10[a] - pos10[b], box10)
        origin = np.full(n_pairs, np.inf)
        np.minimum.at(origin, k, vals)
        dmax = float(np.abs(origin - hist_last).max())
        assert dmax < 1e-6, f"origin mismatch {protein} R{replica}: {dmax:.3e}"
    else:
        origin = np.empty(0)

    # Box-zero replicas (2hng_A R3, 4cv4_A R2: the whole box array is
    # zero in the ATLAS data, the same finding as K15/K53) have no salt
    # candidates; write the empty series so the join stays uniform.
    if n_pairs == 0:
        out = {"res_i": res_i.astype(np.int32), "res_j": res_j.astype(np.int32)}
        for num in (11, 12, 13, 14):
            out[f"gen{num}"] = np.empty((4, 9, 0), dtype=np.float32)
        np.savez_compressed(OUT / f"{protein}_R{replica}.npz", **out)
        return 0

    # ---- generation (proposals 11-14), K17 protocol ----
    boxv, ca_idx, owner, values, modes, left, right, blocks_e = generation(pos10, box10, maps)
    rng = np.random.default_rng([SEED, replica, 10])
    signal = mode_paths(modes, values, rng)
    ca_pos = pos10[ca_idx]
    local0 = local_paths(ca_pos, pos10, owner, boxv, left, right, blocks_e, rng, omega=0.)
    fast = local_paths(ca_pos, pos10, owner, boxv, left, right, blocks_e, rng, omega=.8)
    disp_res = np.einsum("ncm,rtm->nrtc", modes, signal)
    disp11 = disp_res[owner]

    out = {"res_i": res_i.astype(np.int32), "res_j": res_j.astype(np.int32)}
    if n_pairs:
        tracks = {}
        tracks[11] = min_tracks_from_positions(pos10, boxv, disp11, a, b, k, n_pairs)
        tracks[13] = min_tracks_from_positions(pos10, boxv, local0, a, b, k, n_pairs)
        tracks[14] = min_tracks_from_positions(pos10, boxv, disp11 + .55 * fast, a, b, k, n_pairs)
        tracks[12] = focused_tracks(pos10, boxv, modes, values, signal, owner, a, b, k, n_pairs)
        for num in (11, 12, 13, 14):
            full = np.concatenate([np.broadcast_to(origin, (4, 1, n_pairs)), tracks[num]], axis=1)
            out[f"gen{num}"] = full.astype(np.float32)
    else:
        for num in (11, 12, 13, 14):
            out[f"gen{num}"] = np.empty((4, 9, 0), dtype=np.float32)
    np.savez_compressed(OUT / f"{protein}_R{replica}.npz", **out)

    if verify_contact:
        # rebuild K17's contact candidates and compare gen tracks
        n = len(maps)
        sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        ca_d = distance(pos10[ca_idx[:, None], :] - pos10[ca_idx[None, :], :], box10)
        ii, jj = np.where((ca_d <= 12.0) & (sep >= 2)
                          & (np.arange(n)[:, None] < np.arange(n)[None, :]))
        heavy = [np.asarray([v for key, v in m.items()
                             if not key.startswith("H") and not key[0].isdigit()], dtype=int)
                 for m in maps]
        ah, bh, kh = pair_arrays(ii, jj, heavy)
        vals = np.linalg.norm(_mic(pos10[bh] - pos10[ah], boxv), axis=1)
        cur = np.full(len(ii), np.inf)
        np.minimum.at(cur, kh, vals)
        sel = np.flatnonzero(cur > 4.5)
        t11 = min_tracks_from_positions(pos10, boxv, disp11, ah, bh, kh, len(ii))[:, :, sel]
        t13 = min_tracks_from_positions(pos10, boxv, local0, ah, bh, kh, len(ii))[:, :, sel]
        t14 = min_tracks_from_positions(pos10, boxv, disp11 + .55 * fast, ah, bh, kh, len(ii))[:, :, sel]
        t12 = focused_tracks(pos10, boxv, modes, values, signal, owner, ah, bh, kh, len(ii))[:, :, sel]
        origin_c = cur[sel]
        with np.load(K17_OUT / f"{protein}_R{replica}.npz") as zk:
            assert np.array_equal(zk["res_i"], ii[sel]) and np.array_equal(zk["res_j"], jj[sel]), \
                "contact candidate mismatch vs K17"
            for num, tr in ((11, t11), (12, t12), (13, t13), (14, t14)):
                mine = np.concatenate([np.broadcast_to(origin_c, (4, 1, len(sel))), tr],
                                      axis=1).astype(np.float32)
                ref = zk[f"gen{num}"]
                dmax = float(np.abs(mine - ref).max()) if mine.size else 0.0
                print(f"  verify gen{num}: shape {mine.shape} vs {ref.shape}, "
                      f"max abs diff {dmax:.3e}", flush=True)
                assert mine.shape == ref.shape and dmax < 1e-5, f"gen{num} mismatch"
    return n_pairs


def main():
    args = sys.argv[1:]
    verify = "--verify-contact" in args
    targets = [a for a in args if not a.startswith("--")] or list(ALL_P)
    for protein in targets:
        for replica in (1, 2, 3):
            f = OUT / f"{protein}_R{replica}.npz"
            if f.is_file():
                print("SKIP", protein, replica, flush=True)
                continue
            t0 = time.time()
            try:
                n = run_one(protein, replica, verify_contact=verify)
            except Exception as e:
                print("FAILED", protein, "R", replica, type(e).__name__, e, flush=True)
                continue
            print("DONE", protein, "R", replica, "salt candidates", n,
                  "sec", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
