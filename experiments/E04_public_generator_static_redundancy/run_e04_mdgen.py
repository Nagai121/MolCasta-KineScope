"""E04: Applying KineScope's Finding 4 Diagnostic (Static Redundancy R^2 + Residual Label Correlation + Downstream Guidance)
to a real published SOTA Generative MD model: MDGen (Jing et al., NeurIPS 2024, official `atlas.ckpt`).

Strictly leak-free protocol:
  - Evaluated ONLY on 25 unseen proteins from MDGen's official `splits/atlas_test.csv` (45..140 residues).
  - Input to MDGen: ONLY the single static origin frame (frame 10, t = 1.0 ns) of ATLAS R1.
  - MDGen generates an all-atom (`atom14`) forward trajectory (250 frames @ 400 ps = 100 ns) from that single frame.
  - Compares side-by-side on the exact same HB-formation candidate pairs (3.5 < d(t0) <= 6.0 A, diff residues):
      (1) Static explainability R^2 of pair-mean distance (Finding 4 diagnostic):
          Real explicit ATLAS 0-1 ns vs MDGen short (frames 0..3, ~0-1.2 ns) vs MDGen full (frames 0..249, 0-100 ns).
      (2) Correlation of MDGen pair mean with static-predicted real mean (K43 diagnostic).
      (3) Residual alignment: Pearson r between MDGen residual and real 0-1 ns residual, and correlation with future labels.
      (4) Downstream H-bond formation AUPRC at 1, 5, 10, 30, 50, 90 ns (5-fold protein CV, 100% public 20-feature baseline).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.lib.mdamath import triclinic_vectors
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, r2_score
from sklearn.model_selection import GroupKFold

MDGEN_REPO = Path(r"d:\KineScope\experiments\E04_public_generator_static_redundancy\mdgen_repo")
sys.path.insert(0, str(MDGEN_REPO))
sys.path.insert(0, r"d:\KineScope\experiments\E01_true_short_md_guidance")

import extract_e01_e02 as ext_mod
from mdgen import residue_constants as rc
from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.tensor_utils import tensor_tree_map
from mdgen.wrapper import NewMDGenWrapper

ATLAS_ROOT = Path(r"d:\KineScope\experiments\shared_data\atlas")
OUT = Path(r"d:\KineScope\experiments\E04_public_generator_static_redundancy")
ROLL_CACHE = OUT / "mdgen_rollouts"
CKPT_PATH = OUT / "checkpoints" / "atlas.ckpt"
MDGEN_SPLIT = MDGEN_REPO / "splits" / "atlas_test.csv"

HORIZONS = (10, 50, 100, 300, 500, 900)
HORIZON_NS = [1, 5, 10, 30, 50, 90]

REG_KW = dict(
    max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
    min_samples_leaf=25, l2_regularization=0.2, random_state=20261126,
)
CLF_KW = dict(
    max_iter=160, learning_rate=0.06, max_leaf_nodes=31,
    min_samples_leaf=25, l2_regularization=0.15, random_state=20261126,
)

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "HSD": "H",
    "HSE": "H", "HSP": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T",
    "TRP": "W", "TYR": "Y", "VAL": "V",
}
STD_RES3 = {
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
}


def frame_to_atom14_and_seq(u: mda.Universe):
    """Convert current MDAnalysis frame (in Angstroms) into MDGen (1, L, 14, 3) atom14 + seqres + map."""
    res_list = []
    for res in u.residues:
        nm = {a.name.strip(): a for a in res.atoms}
        if not {"N", "CA", "C", "O"}.issubset(nm):
            continue
        r3 = STD_RES3.get(res.resname.upper(), res.resname.upper())
        if r3 not in rc.restype_name_to_atom14_names:
            continue
        res_list.append((res, r3, nm))

    L = len(res_list)
    arr = np.zeros((1, L, 14, 3), dtype=np.float32)
    seq_chars = []
    orig_atom_to_14 = {}
    for i, (res, r3, nm) in enumerate(res_list):
        seq_chars.append(AA3_TO_1[r3])
        names14 = rc.restype_name_to_atom14_names[r3]
        for aname, atom in nm.items():
            if aname in names14:
                j = names14.index(aname)
                arr[0, i, j] = atom.position.astype(np.float32)
                orig_atom_to_14[int(atom.index)] = (i, j)
    return arr, "".join(seq_chars), orig_atom_to_14


@torch.no_grad()
def generate_mdgen_for_protein(model, prot: str, num_frames: int = 250):
    out_npz = ROLL_CACHE / f"{prot}.npz"
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

    # Extract real frames: 0..10 and future horizons 20, 60, 110, 310, 510, 910
    needed = list(range(11)) + [10 + h for h in HORIZONS]
    pos_map, box_map = {}, {}
    for f in needed:
        ts = u.trajectory[f]
        pos_map[f] = ts.positions.astype(np.float64)
        box_map[f] = triclinic_vectors(ts.dimensions).astype(np.float64) if (ts.dimensions is not None and ts.dimensions[0] > 1.0) else None

    # Origin frame 10 -> candidate pairs & public 20 features
    p10, b10 = pos_map[10], box_map[10]
    da, ac = topo["donors"], topo["acceptors"]
    d_at, d_res = da[:, 0], da[:, 1]
    a_at, a_res = ac[:, 0], ac[:, 1]
    dd10 = ext_mod.mic_dist(p10[d_at[:, None]], p10[a_at[None, :]], b10)
    ii, jj = np.where((dd10 > 3.5) & (dd10 <= 6.0) & (d_res[:, None] != a_res[None, :]))
    a, b = d_at[ii], a_at[jj]
    X20 = ext_mod.compute_public_features(p10, b10, topo, ii, jj, dd10, include_k302=False)
    real_ser = np.column_stack([ext_mod.mic_dist(pos_map[f][a], pos_map[f][b], box_map[f]) for f in range(11)]).astype(np.float32)
    y_fut = np.column_stack([(ext_mod.mic_dist(pos_map[10 + h][a], pos_map[10 + h][b], box_map[10 + h]) <= 3.5) for h in HORIZONS]).astype(np.uint8)

    # Convert ONLY frame 10 to MDGen atom14 input
    u.trajectory[10]
    arr0, seqres, orig_to_14 = frame_to_atom14_and_seq(u)
    L = len(seqres)

    # Map candidate atoms to (res_i, atom14_j)
    valid = np.array([(int(ai) in orig_to_14 and int(bi) in orig_to_14) for ai, bi in zip(a, b)], dtype=bool)
    idx_a_r = np.array([orig_to_14[int(ai)][0] if v else 0 for ai, v in zip(a, valid)], dtype=np.int32)
    idx_a_j = np.array([orig_to_14[int(ai)][1] if v else 0 for ai, v in zip(a, valid)], dtype=np.int32)
    idx_b_r = np.array([orig_to_14[int(bi)][0] if v else 0 for bi, v in zip(b, valid)], dtype=np.int32)
    idx_b_j = np.array([orig_to_14[int(bi)][1] if v else 0 for bi, v in zip(b, valid)], dtype=np.int32)

    # Verify origin distance match in atom14
    d0_14 = np.linalg.norm(arr0[0, idx_a_r[valid], idx_a_j[valid]] - arr0[0, idx_b_r[valid], idx_b_j[valid]], axis=1)
    max_d0_err = float(np.abs(d0_14 - real_ser[valid, 10]).max())
    assert max_d0_err < 0.05, f"{prot}: atom14 origin mismatch {max_d0_err}"

    # Build MDGen batch (exact match to sim_inference.py)
    frames = atom14_to_frames(torch.from_numpy(arr0))
    seq_t = torch.tensor([rc.restype_order[c] for c in seqres], dtype=torch.long)
    atom37 = torch.from_numpy(atom14_to_atom37(arr0, seq_t[None])).float()
    torsions, torsion_mask = atom37_to_torsions(atom37, seq_t[None])
    batch = {
        "torsions": torsions[None].expand(1, num_frames, -1, -1, -1).cuda(),
        "torsion_mask": torsion_mask[0][None].cuda(),
        "trans": frames._trans[None].expand(1, num_frames, -1, -1).cuda(),
        "rots": frames._rots._rot_mats[None].expand(1, num_frames, -1, -1, -1).cuda(),
        "seqres": seq_t[None].cuda(),
        "mask": torch.ones((1, L), device="cuda"),
    }

    torch.manual_seed(20261126)
    torch.cuda.empty_cache()
    try:
        atom14_out, _ = model.inference(batch)  # (1, T, L, 14, 3)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            atom14_out, _ = model.inference(batch)
    gen_xyz = atom14_out[0].detach().cpu().float().numpy()  # (T, L, 14, 3)

    # Extract pair distance trajectories from MDGen generated rollout: (N, T)
    pa = gen_xyz[:, idx_a_r, idx_a_j, :]  # (T, N, 3)
    pb = gen_xyz[:, idx_b_r, idx_b_j, :]  # (T, N, 3)
    gen_dist = np.linalg.norm(pa - pb, axis=-1).T.astype(np.float32)  # (N, T)

    # Short window (frames 0..3 = 0..1.2 ns @ 400 ps/frame) & full rollout (0..T-1)
    # Also frames 1..4 (post-origin ~0.4..1.6 ns)
    def stats6(D):
        return np.column_stack([
            D[:, 0], D.mean(1), D.std(1), D.min(1), D[:, -1] - D[:, 0], (D <= 3.5).mean(1)
        ]).astype(np.float32)

    s_short = stats6(gen_dist[:, :4])       # 0..1.2 ns
    s_10ns = stats6(gen_dist[:, :26])       # 0..10 ns
    s_full = stats6(gen_dist)               # 0..100 ns (250 frames)

    np.savez_compressed(
        out_npz,
        protein=prot,
        n_res=np.int32(L),
        valid=valid,
        max_d0_err=np.float32(max_d0_err),
        X20=X20[valid],
        real_ser=real_ser[valid],
        y_fut=y_fut[valid],
        gen_short_stats=s_short[valid],
        gen_10ns_stats=s_10ns[valid],
        gen_full_stats=s_full[valid],
        gen_d0_frame=gen_dist[valid, 0],
        gen_lag1_ac=np.float32(np.nanmean([
            np.corrcoef(row[:-1], row[1:])[0, 1]
            for row in gen_dist[valid] if row.std() > 1e-5
        ])),
        elapsed_s=np.float32(time.time() - t0),
    )
    return (prot, f"ok (L={L}, N={valid.sum()}, d0_err={max_d0_err:.4f}A)", time.time() - t0)


def oof_reg(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    oof = np.empty(len(y), dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(**REG_KW)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def oof_clf(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    oof = np.empty(len(y), dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingClassifier(**CLF_KW)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def analyze(prots):
    Xs, S_real, Ys, Gs, G10s, Gfs, Ps = [], [], [], [], [], [], []
    times, acs = [], []
    for p in prots:
        f = ROLL_CACHE / f"{p}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        if len(z["X20"]) == 0:
            continue
        Xs.append(z["X20"])
        S_real.append(z["real_ser"])
        Ys.append(z["y_fut"])
        Gs.append(z["gen_short_stats"])
        G10s.append(z["gen_10ns_stats"])
        Gfs.append(z["gen_full_stats"])
        Ps.append(np.full(len(z["X20"]), p, dtype=object))
        times.append(float(z["elapsed_s"]))
        acs.append(float(z["gen_lag1_ac"]))

    X = np.concatenate(Xs).astype(np.float64)
    S = np.concatenate(S_real).astype(np.float64)
    Y = np.concatenate(Ys).astype(np.uint8)
    G_short = np.concatenate(Gs).astype(np.float64)
    G_10ns = np.concatenate(G10s).astype(np.float64)
    G_full = np.concatenate(Gfs).astype(np.float64)
    P = np.concatenate(Ps)

    real_mean = S.mean(axis=1)
    real_h6 = np.column_stack([S[:, 0], S.mean(1), S.std(1), S.min(1), S[:, -1] - S[:, 0], (S <= 3.5).mean(1)])
    m_short = G_short[:, 1]
    m_10ns = G_10ns[:, 1]
    m_full = G_full[:, 1]

    # 1. Finding 4 diagnostic: static explainability R^2 (5-fold protein CV)
    pred_real = oof_reg(X, real_mean, P)
    pred_short = oof_reg(X, m_short, P)
    pred_10ns = oof_reg(X, m_10ns, P)
    pred_full = oof_reg(X, m_full, P)

    res_real = real_mean - pred_real
    res_short = m_short - pred_short
    res_10ns = m_10ns - pred_10ns
    res_full = m_full - pred_full

    # 2. K43 diagnostic: correlation of generated mean with static prediction of real mean, and with real mean
    # 3. Residual alignment & real_mean prediction with MDGen stats
    pred_real_plus_short = oof_reg(np.column_stack([X, G_short]), real_mean, P)
    pred_real_plus_full = oof_reg(np.column_stack([X, G_full]), real_mean, P)

    # Shuffle control within protein
    rng = np.random.default_rng(20261126)
    G_full_sh = G_full.copy()
    for pr in np.unique(P):
        idx = np.where(P == pr)[0]
        G_full_sh[idx] = G_full_sh[idx][rng.permutation(len(idx))]

    print(f"\n=== E04 Analysis: MDGen (atlas.ckpt) on {len(np.unique(P))} strictly unseen atlas_test proteins (N={len(real_mean):,} pairs) ===", flush=True)
    print(f"Static explainability R^2 (Finding 4):", flush=True)
    print(f"  Real ATLAS 0-1 ns explicit MD : R2={r2_score(real_mean, pred_real):.4f}, resid_SD={res_real.std():.4f} A", flush=True)
    print(f"  MDGen short (0-1.2 ns, 4 fr)  : R2={r2_score(m_short, pred_short):.4f}, resid_SD={res_short.std():.4f} A", flush=True)
    print(f"  MDGen 10 ns (26 fr)           : R2={r2_score(m_10ns, pred_10ns):.4f}, resid_SD={res_10ns.std():.4f} A", flush=True)
    print(f"  MDGen full (100 ns, 250 fr)   : R2={r2_score(m_full, pred_full):.4f}, resid_SD={res_full.std():.4f} A", flush=True)
    print(f"K43 / Residual Alignment:", flush=True)
    print(f"  corr(MDGen_short, static_pred_real)={pearsonr(m_short, pred_real)[0]:.4f} | corr(MDGen_short, real_mean)={pearsonr(m_short, real_mean)[0]:.4f}", flush=True)
    print(f"  corr(MDGen_full,  static_pred_real)={pearsonr(m_full, pred_real)[0]:.4f}  | corr(MDGen_full,  real_mean)={pearsonr(m_full, real_mean)[0]:.4f}", flush=True)
    print(f"  corr(res_short, res_real)={pearsonr(res_short, res_real)[0]:.4f} | corr(res_10ns, res_real)={pearsonr(res_10ns, res_real)[0]:.4f} | corr(res_full, res_real)={pearsonr(res_full, res_real)[0]:.4f}", flush=True)
    print(f"  Real-mean SD: static20={res_real.std():.4f} A -> +MDGen_short={(real_mean-pred_real_plus_short).std():.4f} A -> +MDGen_full={(real_mean-pred_real_plus_full).std():.4f} A", flush=True)
    print(f"  Pair within-series SD: real={S.std(1).mean():.3f} A | MDGen_short={G_short[:,2].mean():.3f} A | MDGen_full={G_full[:,2].mean():.3f} A | MDGen lag1 AC={np.mean(acs):.3f}", flush=True)

    # 4. Downstream H-bond formation AUPRC (1..90 ns)
    downstream = {}
    for hi, h_ns in enumerate(HORIZON_NS):
        y = Y[:, hi]
        p_s = oof_clf(X, y, P)
        p_r = oof_clf(np.column_stack([X, real_h6]), y, P)
        p_gs = oof_clf(np.column_stack([X, G_short]), y, P)
        p_g10 = oof_clf(np.column_stack([X, G_10ns]), y, P)
        p_gf = oof_clf(np.column_stack([X, G_full]), y, P)
        p_sh = oof_clf(np.column_stack([X, G_full_sh]), y, P)
        ap_s = float(average_precision_score(y, p_s))
        ap_r = float(average_precision_score(y, p_r))
        ap_gs = float(average_precision_score(y, p_gs))
        ap_g10 = float(average_precision_score(y, p_g10))
        ap_gf = float(average_precision_score(y, p_gf))
        ap_sh = float(average_precision_score(y, p_sh))
        g_r = ap_r - ap_s
        downstream[str(h_ns)] = {
            "ap_static20": ap_s,
            "ap_real_1ns_6stats": ap_r,
            "ap_mdgen_short": ap_gs,
            "ap_mdgen_10ns": ap_g10,
            "ap_mdgen_full": ap_gf,
            "ap_mdgen_shuffle": ap_sh,
            "gain_real": g_r,
            "gain_mdgen_short": ap_gs - ap_s,
            "gain_mdgen_10ns": ap_g10 - ap_s,
            "gain_mdgen_full": ap_gf - ap_s,
            "recovery_mdgen_short": float((ap_gs - ap_s) / g_r) if g_r > 1e-6 else float("nan"),
            "recovery_mdgen_10ns": float((ap_g10 - ap_s) / g_r) if g_r > 1e-6 else float("nan"),
            "recovery_mdgen_full": float((ap_gf - ap_s) / g_r) if g_r > 1e-6 else float("nan"),
            "corr_real_resid_y": float(pearsonr(res_real, y)[0]),
            "corr_mdgen_short_resid_y": float(pearsonr(res_short, y)[0]),
            "corr_mdgen_full_resid_y": float(pearsonr(res_full, y)[0]),
        }
        print(f"  H={h_ns:2d}ns | static={ap_s:.4f} real1ns={ap_r:.4f} (+{g_r:.4f}) | "
              f"MDGen short={ap_gs:.4f} (rec={(ap_gs-ap_s)/g_r:+.3f}) | "
              f"10ns={ap_g10:.4f} (rec={(ap_g10-ap_s)/g_r:+.3f}) | "
              f"full={ap_gf:.4f} (rec={(ap_gf-ap_s)/g_r:+.3f}) | shuf={ap_sh:.4f}", flush=True)

    res = {
        "generator": "MDGen (Jing et al., NeurIPS 2024, official bjing-mit/mdgen atlas.ckpt)",
        "n_proteins": int(len(np.unique(P))),
        "proteins": sorted(np.unique(P).tolist()),
        "n_candidates": int(len(real_mean)),
        "mean_gpu_sec_per_protein": float(np.mean(times)),
        "finding4_static_explainability": {
            "real_1ns_r2": float(r2_score(real_mean, pred_real)),
            "real_1ns_resid_sd": float(res_real.std()),
            "mdgen_short_r2": float(r2_score(m_short, pred_short)),
            "mdgen_short_resid_sd": float(res_short.std()),
            "mdgen_10ns_r2": float(r2_score(m_10ns, pred_10ns)),
            "mdgen_10ns_resid_sd": float(res_10ns.std()),
            "mdgen_full_r2": float(r2_score(m_full, pred_full)),
            "mdgen_full_resid_sd": float(res_full.std()),
        },
        "k43_and_residual_alignment": {
            "corr_mdgen_short_vs_static_pred": float(pearsonr(m_short, pred_real)[0]),
            "corr_mdgen_full_vs_static_pred": float(pearsonr(m_full, pred_real)[0]),
            "corr_mdgen_short_vs_real_mean": float(pearsonr(m_short, real_mean)[0]),
            "corr_mdgen_full_vs_real_mean": float(pearsonr(m_full, real_mean)[0]),
            "corr_res_short_vs_res_real": float(pearsonr(res_short, res_real)[0]),
            "corr_res_10ns_vs_res_real": float(pearsonr(res_10ns, res_real)[0]),
            "corr_res_full_vs_res_real": float(pearsonr(res_full, res_real)[0]),
            "real_mean_sd_static20": float(res_real.std()),
            "real_mean_sd_plus_mdgen_short": float((real_mean - pred_real_plus_short).std()),
            "real_mean_sd_plus_mdgen_full": float((real_mean - pred_real_plus_full).std()),
            "within_pair_sd_real": float(S.std(1).mean()),
            "within_pair_sd_mdgen_short": float(G_short[:, 2].mean()),
            "within_pair_sd_mdgen_full": float(G_full[:, 2].mean()),
            "mdgen_lag1_autocorr": float(np.mean(acs)),
        },
        "downstream_hb_formation": downstream,
    }
    (OUT / "e04_mdgen_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nSaved e04_mdgen_results.json", flush=True)


def main():
    ROLL_CACHE.mkdir(parents=True, exist_ok=True)
    # Import selection list
    sys.path.insert(0, str(OUT))
    import fetch_e04_assets
    chosen = [p for p, _ in fetch_e04_assets.select_e04_proteins(25)]
    ready = [p for p in chosen if (ATLAS_ROOT / p / "done.json").exists()]
    print(f"Loading MDGen checkpoint {CKPT_PATH} on {torch.cuda.get_device_name(0)}...", flush=True)
    torch.serialization.add_safe_globals([argparse.Namespace])
    model = NewMDGenWrapper.load_from_checkpoint(str(CKPT_PATH), map_location="cuda")
    model.eval().to("cuda")
    num_frames = int(getattr(model.args, "num_frames", 250))
    print(f"MDGen loaded! model.args.num_frames={num_frames} | running on {len(ready)} strictly unseen atlas_test proteins...", flush=True)

    done = []
    for i, p in enumerate(ready, 1):
        try:
            prot, status, dt = generate_mdgen_for_protein(model, p, num_frames=num_frames)
            print(f"[{i}/{len(ready)}] {prot}: {status} ({dt:.1f}s)", flush=True)
            done.append(prot)
        except Exception as e:
            print(f"[{i}/{len(ready)}] {p}: FAIL {e}", flush=True)
            torch.cuda.empty_cache()
    assert len(done) >= 10, f"Too few MDGen rollouts completed: {len(done)}"
    analyze(done)


if __name__ == "__main__":
    main()
