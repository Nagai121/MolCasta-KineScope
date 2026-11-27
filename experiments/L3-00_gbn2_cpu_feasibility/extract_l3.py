"""L3 extraction: all 6 phenomena on E03's 25 proteins (ATLAS R1, origin frame 10).

Adapted from E05's extract_e05_real.py (same definitions, same feature code
via E01's extract_e01_e02 module); only the protein set and paths differ.
Per protein, keys {mode}_X / {mode}_series / {mode}_y / {mode}_pair for
mode in hb_form, hb_diss, ct_form, ct_diss, sb_form, sb_diss.
hb X = 20 public features (identical to E03's static20 = X30[:, :20]),
ct/sb X = 10 public features.
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

sys.path.insert(0, "/home/hatch/workspace/kinescope/repo/experiments/E01_true_short_md_guidance")
import extract_e01_e02 as ext_mod

ATLAS_ROOT = Path("/home/hatch/workspace/kinescope/l3work/atlas")
OUT_CACHE = Path("/home/hatch/workspace/kinescope/l3work/extracted")
HORIZONS = (10, 50, 100, 300, 500, 900)
PROTS = ['1a62_A', '1dd3_B', '1ikt_A', '1m8s_A', '1sf9_A', '1uha_A', '1xmk_A',
         '2e5y_B', '2gpi_A', '2i9x_A', '2nwv_A', '2ysk_A', '3c9p_A', '3fpn_A',
         '3kz7_A', '3pes_A', '3t92_A', '3v3l_A', '4cv4_A', '4myy_A', '4zds_B',
         '5ii7_A', '6bk4_A', '6in7_A', '7ead_A']


def compute_res_pair_features(p10, b10, topo, pairs, d0_arr, mode):
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
            X[idx] = [
                d0_arr[idx], ca_d, cb_d, np.log1p(sep), float(sep <= 4),
                1.0 if resnames[r1] == "ARG" else 0.0,
                1.0 if resnames[r2] == "GLU" else 0.0,
                ca_pack6[r1], ca_pack6[r2], ca_pack8[r1] + ca_pack8[r2],
            ]
    return X


def extract_one(prot):
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
    t0 = time.time()
    ok = cached = fail = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(extract_one, p): p for p in PROTS}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            p = futs[fut]
            try:
                prot, status, dt = fut.result()
                if status == "cached":
                    cached += 1
                else:
                    ok += 1
                print(f"[{i}/{len(PROTS)}] {prot}: {status} ({dt:.1f}s)", flush=True)
            except Exception as e:
                fail += 1
                print(f"[{i}/{len(PROTS)}] {p}: FAIL {e}", flush=True)
    print(f"Done in {time.time() - t0:.1f}s | ok={ok} cached={cached} fail={fail}", flush=True)
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
