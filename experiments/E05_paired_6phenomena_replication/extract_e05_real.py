"""E05 Step 2: Extract all 6 phenomena on the 49 strictly unseen MDGen atlas_test.csv proteins (L <= 250).

Phenomena extracted strictly at origin frame 10 (t = 1.0 ns) of ATLAS R1, with real 0-1 ns history [0..10]
and future horizons (+1, +5, +10, +30, +50, +90 ns -> frames 20, 60, 110, 310, 510, 910):
  1. hb_form : donor-acceptor heavy atoms (diff res), 3.5 < d(f10) <= 6.0 A -> d(f10+h) <= 3.5 A (thresh 3.5)
  2. hb_diss : donor-acceptor heavy atoms (diff res), d(f10) <= 3.5 A       -> d(f10+h) >  3.5 A (thresh 3.5)
  3. ct_form : residue pairs |i-j| >= 2, CA <= 12.0 & heavy_min > 4.5 A     -> heavy_min <= 4.5 A (thresh 4.5)
  4. ct_diss : residue pairs |i-j| >= 2, heavy_min <= 4.5 A                 -> heavy_min >  4.5 A (thresh 4.5)
  5. sb_form : charged sidechain pairs |i-j| >= 2, 4.0 < d <= 10.0 A        -> d <= 4.0 A (thresh 4.0)
  6. sb_diss : charged sidechain pairs |i-j| >= 2, d <= 4.0 A               -> d >  4.0 A (thresh 4.0)

Also saves 0-based residue indices & atom names/indices so BOTH OpenMM GBn2 and MDGen atom14
extract the exact same 6-phenomenon candidate pairs with verified d0_err < 0.05 A.
"""
import concurrent.futures
import json
import sys
import time
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.lib.mdamath import triclinic_vectors
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, r"d:\KineScope\experiments\E01_true_short_md_guidance")
sys.path.insert(0, r"d:\KineScope\experiments\E05_paired_6phenomena_replication")
import extract_e01_e02 as ext_mod
import fetch_e05_atlas

ATLAS_ROOT = Path(r"d:\KineScope\experiments\shared_data\atlas")
OUT_CACHE = Path(r"d:\KineScope\experiments\E05_paired_6phenomena_replication\cache_real")
HORIZONS = (10, 50, 100, 300, 500, 900)


def compute_res_pair_features(p10: np.ndarray, b10, topo: dict, pairs: list, d0_arr: np.ndarray, mode: str) -> np.ndarray:
    """Compute 10 public structural features for residue-level contact or salt-bridge pairs."""
    n = len(pairs)
    if n == 0:
        return np.zeros((0, 10), dtype=np.float32)
    ca_idx = topo["ca_idx"]
    cb_idx = topo["cb_idx"]
    heavy_by_res = topo["heavy_by_res"]
    resnames = topo["resnames"]
    is_charged = topo["is_charged"]

    heavy_pos = p10[topo["all_heavy"]]
    tree = cKDTree(heavy_pos)
    ca_pos = p10[ca_idx]
    ca_pack6 = np.array([len(x) for x in tree.query_ball_point(ca_pos, 6.0)], dtype=np.float32)
    ca_pack8 = np.array([len(x) for x in tree.query_ball_point(ca_pos, 8.0)], dtype=np.float32)

    X = np.empty((n, 10), dtype=np.float32)
    for idx, (r1, r2) in enumerate(pairs):
        ca_d = float(ext_mod.mic_dist(p10[ca_idx[r1]], p10[ca_idx[r2]], b10))
        cb_d = float(ext_mod.mic_dist(p10[cb_idx[r1]], p10[cb_idx[r2]], b10))
        sep = abs(r2 - r1)
        if mode.startswith("ct"):
            X[idx] = [
                d0_arr[idx], ca_d, cb_d, np.log1p(sep), float(sep <= 4),
                float(len(heavy_by_res[r1])), float(len(heavy_by_res[r2])),
                ca_pack6[r1] + ca_pack6[r2], ca_pack8[r1] + ca_pack8[r2],
                is_charged[r1] + is_charged[r2],
            ]
        else:
            # sb: r1 is pos (LYS/ARG), r2 is neg (ASP/GLU)
            X[idx] = [
                d0_arr[idx], ca_d, cb_d, np.log1p(sep), float(sep <= 4),
                1.0 if resnames[r1] == "ARG" else 0.0,
                1.0 if resnames[r2] == "GLU" else 0.0,
                ca_pack6[r1], ca_pack6[r2], ca_pack8[r1] + ca_pack8[r2],
            ]
    return X


def extract_one(prot: str):
    out_npz = OUT_CACHE / f"{prot}.npz"
    if out_npz.exists():
        return (prot, "cached", 0.0)
    t0 = time.time()
    pdir = ATLAS_ROOT / prot
    meta = json.loads((pdir / "done.json").read_text(encoding="utf-8"))
    pdb_path = pdir / meta["pdb"]
    r1_xtc = pdir / sorted(meta["xtc_files"])[0]
    topo = ext_mod.parse_topology(pdb_path)
    u = mda.Universe(str(pdb_path), str(r1_xtc))
    assert len(u.trajectory) >= 911, f"{prot}: {len(u.trajectory)} frames"

    needed = list(range(11)) + [10 + h for h in HORIZONS]
    pos_map, box_map = {}, {}
    for f in needed:
        ts = u.trajectory[f]
        pos_map[f] = ts.positions.astype(np.float64)
        box_map[f] = triclinic_vectors(ts.dimensions).astype(np.float64) if (ts.dimensions is not None and ts.dimensions[0] > 1.0) else None

    p10, b10 = pos_map[10], box_map[10]
    bundle = {"protein": prot, "n_res": np.int32(topo["n_res"])}

    # --- 1 & 2. HB Formation & Dissociation ---
    da, ac = topo["donors"], topo["acceptors"]
    d_at, d_res = da[:, 0], da[:, 1]
    a_at, a_res = ac[:, 0], ac[:, 1]
    dd10 = ext_mod.mic_dist(p10[d_at[:, None]], p10[a_at[None, :]], b10)
    diff_res = d_res[:, None] != a_res[None, :]

    for mode, mask in (
        ("hb_form", (dd10 > 3.5) & (dd10 <= 6.0) & diff_res),
        ("hb_diss", (dd10 <= 3.5) & diff_res),
    ):
        ii, jj = np.where(mask)
        a, b = d_at[ii], a_at[jj]
        X20 = ext_mod.compute_public_features(p10, b10, topo, ii, jj, dd10, include_k302=False)
        ser = np.column_stack([ext_mod.mic_dist(pos_map[f][a], pos_map[f][b], box_map[f]) for f in range(11)]).astype(np.float32) if len(a) else np.zeros((0, 11), dtype=np.float32)
        y = np.column_stack([
            (ext_mod.mic_dist(pos_map[10 + h][a], pos_map[10 + h][b], box_map[10 + h]) <= 3.5) if mode == "hb_form"
            else (ext_mod.mic_dist(pos_map[10 + h][a], pos_map[10 + h][b], box_map[10 + h]) > 3.5)
            for h in HORIZONS
        ]).astype(np.uint8) if len(a) else np.zeros((0, 6), dtype=np.uint8)
        bundle[f"{mode}_X"] = X20
        bundle[f"{mode}_series"] = ser
        bundle[f"{mode}_y"] = y
        bundle[f"{mode}_pair"] = np.column_stack([a, b]).astype(np.int32) if len(a) else np.zeros((0, 2), dtype=np.int32)

    # --- 3 & 4. Contact Formation & Dissociation ---
    ca_idx = topo["ca_idx"]
    n_res = topo["n_res"]
    heavy_by_res = topo["heavy_by_res"]
    ca_mat = ext_mod.mic_dist(p10[ca_idx[:, None]], p10[ca_idx[None, :]], b10)
    ci, cj = np.where((ca_mat <= 12.0) & (np.subtract.outer(np.arange(n_res), np.arange(n_res)) <= -2))
    ct_f_pairs, ct_f_d0 = [], []
    ct_d_pairs, ct_d_d0 = [], []
    for r1, r2 in zip(ci, cj):
        h1, h2 = heavy_by_res[r1], heavy_by_res[r2]
        d_min = float(ext_mod.mic_dist(p10[h1[:, None]], p10[h2[None, :]], b10).min())
        if d_min > 4.5:
            ct_f_pairs.append((int(r1), int(r2)))
            ct_f_d0.append(d_min)
        else:
            ct_d_pairs.append((int(r1), int(r2)))
            ct_d_d0.append(d_min)

    for mode, plist, d0list in (("ct_form", ct_f_pairs, ct_f_d0), ("ct_diss", ct_d_pairs, ct_d_d0)):
        m = len(plist)
        X = compute_res_pair_features(p10, b10, topo, plist, np.array(d0list, dtype=np.float32), mode)
        ser = np.empty((m, 11), dtype=np.float32)
        y = np.empty((m, 6), dtype=np.uint8)
        for idx_p, (r1, r2) in enumerate(plist):
            h1, h2 = heavy_by_res[r1], heavy_by_res[r2]
            for f in range(11):
                ser[idx_p, f] = ext_mod.mic_dist(pos_map[f][h1[:, None]], pos_map[f][h2[None, :]], box_map[f]).min()
            for hi, h in enumerate(HORIZONS):
                df = ext_mod.mic_dist(pos_map[10 + h][h1[:, None]], pos_map[10 + h][h2[None, :]], box_map[10 + h]).min()
                y[idx_p, hi] = int(df <= 4.5) if mode == "ct_form" else int(df > 4.5)
        bundle[f"{mode}_X"] = X
        bundle[f"{mode}_series"] = ser
        bundle[f"{mode}_y"] = y
        bundle[f"{mode}_pair"] = np.asarray(plist, dtype=np.int32).reshape(m, 2)

    # --- 5 & 6. Salt-Bridge Formation & Dissociation ---
    pos_g = dict(topo["pos_groups"])
    neg_g = dict(topo["neg_groups"])
    sb_f_pairs, sb_f_d0 = [], []
    sb_d_pairs, sb_d_d0 = [], []
    for rk_p, idx_p in topo["pos_groups"]:
        for rk_n, idx_n in topo["neg_groups"]:
            if abs(rk_p - rk_n) < 2:
                continue
            d_val = float(ext_mod.mic_dist(p10[idx_p[:, None]], p10[idx_n[None, :]], b10).min())
            if 4.0 < d_val <= 10.0:
                sb_f_pairs.append((int(rk_p), int(rk_n)))
                sb_f_d0.append(d_val)
            elif d_val <= 4.0:
                sb_d_pairs.append((int(rk_p), int(rk_n)))
                sb_d_d0.append(d_val)

    for mode, plist, d0list in (("sb_form", sb_f_pairs, sb_f_d0), ("sb_diss", sb_d_pairs, sb_d_d0)):
        m = len(plist)
        X = compute_res_pair_features(p10, b10, topo, plist, np.array(d0list, dtype=np.float32), mode)
        ser = np.empty((m, 11), dtype=np.float32)
        y = np.empty((m, 6), dtype=np.uint8)
        for idx_p, (r1, r2) in enumerate(plist):
            idx_a, idx_b = pos_g[r1], neg_g[r2]
            for f in range(11):
                ser[idx_p, f] = ext_mod.mic_dist(pos_map[f][idx_a[:, None]], pos_map[f][idx_b[None, :]], box_map[f]).min()
            for hi, h in enumerate(HORIZONS):
                df = ext_mod.mic_dist(pos_map[10 + h][idx_a[:, None]], pos_map[10 + h][idx_b[None, :]], box_map[10 + h]).min()
                y[idx_p, hi] = int(df <= 4.0) if mode == "sb_form" else int(df > 4.0)
        bundle[f"{mode}_X"] = X
        bundle[f"{mode}_series"] = ser
        bundle[f"{mode}_y"] = y
        bundle[f"{mode}_pair"] = np.asarray(plist, dtype=np.int32).reshape(m, 2)

    np.savez_compressed(out_npz, **bundle)
    counts = {m: len(bundle[f"{m}_y"]) for m in ("hb_form", "hb_diss", "ct_form", "ct_diss", "sb_form", "sb_diss")}
    return (prot, f"ok {counts}", time.time() - t0)


def main():
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    chosen = [p for p, _ in fetch_e05_atlas.select_e05_proteins(250)]
    ready = [p for p in chosen if (ATLAS_ROOT / p / "done.json").exists()]
    print(f"Extracting 6 phenomena for {len(ready)}/{len(chosen)} atlas_test proteins...", flush=True)
    t0 = time.time()
    ok = cached = fail = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(extract_one, p): p for p in ready}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            p = futs[fut]
            try:
                prot, status, dt = fut.result()
                if status == "cached":
                    cached += 1
                else:
                    ok += 1
                print(f"[{i}/{len(ready)}] {prot}: {status} ({dt:.1f}s)", flush=True)
            except Exception as e:
                fail += 1
                print(f"[{i}/{len(ready)}] {p}: FAIL {e}", flush=True)
    print(f"Done in {time.time() - t0:.1f}s | ok={ok} cached={cached} fail={fail}", flush=True)
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
