"""Shared extractor for E01 (True Short Real-MD Guidance) and E02 (Public K3 + K3-02 + 100-protein Replica Floor).

Reads downloaded ATLAS analysis trajectories (100 unseen proteins x 3 replicas) and produces compact
per-protein .npz bundles with:
  1. HB Formation at origin frame 10 (t=1.0 ns): 11-frame distance series, 6-horizon labels (+1..+90 ns),
     20 public structural features + 10 K3-02 H-bond network features, and exact (donor_atom, acceptor_atom) IDs.
  2. HB Dissociation at origin frame 10 (t=1.0 ns): same structure (d <= 3.5 -> d > 3.5).
  3. Protocol B (Moving-origin true forward prefix from t=0) for HB Formation at origins k in {0, 1, 2, 5}
     (L = 0.0, 0.1, 0.2, 0.5 ns; k=10 is already in #1), with exact horizons k + {10, 50, 100, 300, 500, 900}.
  4. Salt-bridge formation at origin frame 10 (4.0 < d <= 10.0, |i-j| >= 2, charged sidechain atoms) on all 100 proteins.
  5. Contact formation at origin frame 10 (CA <= 12.0, heavy min > 4.5, |i-j| >= 2) on all 100 proteins.
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

ATLAS_ROOT = Path(r"d:\KineScope\experiments\shared_data\atlas")
PROT_LIST_FILE = Path(r"d:\KineScope\MolCasta-KineScope\experiments\K75_readout_transfer_across_horizons_dissociation\proteins108.txt")
OUT_CACHE = Path(r"d:\KineScope\experiments\shared_data\extracted_100")

HORIZONS = (10, 50, 100, 300, 500, 900)  # +1, +5, +10, +30, +50, +90 ns (100 ps/frame)
PROTO_B_ORIGINS = (0, 1, 2, 5)  # L = 0.0, 0.1, 0.2, 0.5 ns (L=1.0 is frame 10)

DONOR_N = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"},
           "GLN": {"NE2"}, "HIS": {"ND1", "NE2"}, "HSD": {"ND1", "NE2"},
           "HSE": {"ND1", "NE2"}, "HSP": {"ND1", "NE2"}, "TRP": {"NE1"}}
DONOR_O = {"SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}}
ACCEPTOR_O = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"},
              "ASN": {"OD1"}, "GLN": {"OE1"}, "SER": {"OG"},
              "THR": {"OG1"}, "TYR": {"OH"}}
ANCHOR = {("LYS", "NZ"): "CE", ("ARG", "NE"): "CZ",
          ("ARG", "NH1"): "CZ", ("ARG", "NH2"): "CZ",
          ("ASN", "ND2"): "CG", ("GLN", "NE2"): "CD",
          ("HIS", "ND1"): "CG", ("HIS", "NE2"): "CE1",
          ("HSD", "ND1"): "CG", ("HSD", "NE2"): "CE1",
          ("HSE", "ND1"): "CG", ("HSE", "NE2"): "CE1",
          ("HSP", "ND1"): "CG", ("HSP", "NE2"): "CE1",
          ("TRP", "NE1"): "CE2", ("SER", "OG"): "CB",
          ("THR", "OG1"): "CB", ("TYR", "OH"): "CZ",
          ("ASP", "OD1"): "CG", ("ASP", "OD2"): "CG",
          ("GLU", "OE1"): "CD", ("GLU", "OE2"): "CD",
          ("ASN", "OD1"): "CG", ("GLN", "OE1"): "CD"}
POS_RES = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}}
NEG_RES = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
CHARGED_RES = {"LYS", "ARG", "ASP", "GLU"}


def mic_vec(v: np.ndarray, box_mat: np.ndarray) -> np.ndarray:
    """Apply minimum-image convention if valid box vectors are present."""
    if box_mat is None or np.linalg.norm(box_mat[0]) < 1.0:
        return v
    if np.allclose(box_mat, np.diag(np.diag(box_mat)), atol=1e-4):
        L = np.diag(box_mat)
        return v - L * np.rint(v / L)
    inv = np.linalg.inv(box_mat)
    fr = v @ inv
    fr -= np.rint(fr)
    return fr @ box_mat


def mic_dist(pos_a: np.ndarray, pos_b: np.ndarray, box_mat: np.ndarray) -> np.ndarray:
    return np.linalg.norm(mic_vec(pos_a - pos_b, box_mat), axis=-1)


def parse_topology(pdb_path: Path):
    u = mda.Universe(str(pdb_path))
    # Map residue -> atom name -> atom index
    maps = []
    resnames = []
    heavy_by_res = []
    all_heavy = []
    for res in u.residues:
        nm = {}
        h_idx = []
        for a in res.atoms:
            nm[a.name] = int(a.index)
            if not a.name.startswith("H") and not a.name[0].isdigit():
                h_idx.append(int(a.index))
                all_heavy.append(int(a.index))
        if not {"N", "CA", "C", "O"}.issubset(nm):
            continue
        maps.append(nm)
        resnames.append(res.resname.upper())
        heavy_by_res.append(np.array(h_idx, dtype=np.int32))
    all_heavy = np.array(all_heavy, dtype=np.int32)

    donors = []     # (atom_idx, res_k, kind: 0=bbN, 1=scN, 2=scO, anchor_idx)
    acceptors = []  # (atom_idx, res_k, kind: 0=bbO, 1=scO, anchor_idx)
    pos_groups = [] # (res_k, atom_indices)
    neg_groups = []
    for k, (m, r) in enumerate(zip(maps, resnames)):
        if r != "PRO":
            donors.append((m["N"], k, 0, m["CA"]))
        for name in sorted(DONOR_N.get(r, ())):
            if name in m:
                donors.append((m[name], k, 1, m.get(ANCHOR.get((r, name)), m["CA"])))
        for name in sorted(DONOR_O.get(r, ())):
            if name in m:
                donors.append((m[name], k, 2, m.get(ANCHOR.get((r, name)), m["CA"])))
        acceptors.append((m["O"], k, 0, m["C"]))
        if "OXT" in m:
            acceptors.append((m["OXT"], k, 0, m["C"]))
        for name in sorted(ACCEPTOR_O.get(r, ())):
            if name in m:
                acceptors.append((m[name], k, 1, m.get(ANCHOR.get((r, name)), m["CA"])))
        if r in POS_RES:
            idxs = [m[nm] for nm in sorted(POS_RES[r]) if nm in m]
            if idxs:
                pos_groups.append((k, np.array(idxs, dtype=np.int32)))
        if r in NEG_RES:
            idxs = [m[nm] for nm in sorted(NEG_RES[r]) if nm in m]
            if idxs:
                neg_groups.append((k, np.array(idxs, dtype=np.int32)))

    ca_idx = np.array([m["CA"] for m in maps], dtype=np.int32)
    cb_idx = np.array([m.get("CB", m["CA"]) for m in maps], dtype=np.int32)
    is_charged = np.array([1.0 if r in CHARGED_RES else 0.0 for r in resnames], dtype=np.float32)

    return {
        "n_res": len(maps),
        "resnames": resnames,
        "ca_idx": ca_idx,
        "cb_idx": cb_idx,
        "is_charged": is_charged,
        "heavy_by_res": heavy_by_res,
        "all_heavy": all_heavy,
        "donors": np.asarray(donors, dtype=np.int32),
        "acceptors": np.asarray(acceptors, dtype=np.int32),
        "pos_groups": pos_groups,
        "neg_groups": neg_groups,
    }


def compute_public_features(pos_f: np.ndarray, box_f: np.ndarray, topo: dict,
                            ii: np.ndarray, jj: np.ndarray, dd_all: np.ndarray,
                            include_k302: bool = True) -> np.ndarray:
    """Compute 20 public structural features (+ optional 10 K3-02 H-bond network features) at frame pos_f."""
    da = topo["donors"]
    ac = topo["acceptors"]
    d_at, d_res, d_kind, d_anc = da[:, 0], da[:, 1], da[:, 2], da[:, 3]
    a_at, a_res, a_kind, a_anc = ac[:, 0], ac[:, 1], ac[:, 2], ac[:, 3]

    a = d_at[ii]
    b = a_at[jj]
    dri = d_res[ii]
    arj = a_res[jj]
    d_anc_i = d_anc[ii]
    a_anc_j = a_anc[jj]

    d0 = dd_all[ii, jj]
    ca_d = mic_dist(pos_f[topo["ca_idx"][dri]], pos_f[topo["ca_idx"][arj]], box_f)
    cb_d = mic_dist(pos_f[topo["cb_idx"][dri]], pos_f[topo["cb_idx"][arj]], box_f)
    sep = np.abs(dri - arj).astype(np.float64)

    # H-bond geometry cosines: cos(anchor_D - D, A - D) and cos(anchor_A - A, D - A)
    v_da = mic_vec(pos_f[b] - pos_f[a], box_f)
    v_danc = mic_vec(pos_f[d_anc_i] - pos_f[a], box_f)
    v_aanc = mic_vec(pos_f[a_anc_j] - pos_f[b], box_f)
    n_da = np.maximum(np.linalg.norm(v_da, axis=-1), 1e-8)
    n_danc = np.maximum(np.linalg.norm(v_danc, axis=-1), 1e-8)
    n_aanc = np.maximum(np.linalg.norm(v_aanc, axis=-1), 1e-8)
    cos_d = np.sum(v_da * v_danc, axis=-1) / (n_da * n_danc)
    cos_a = np.sum((-v_da) * v_aanc, axis=-1) / (n_da * n_aanc)

    # Local heavy-atom packing around donor, acceptor, midpoint (6 A and 8 A)
    heavy_pos = pos_f[topo["all_heavy"]]
    tree = cKDTree(heavy_pos)
    pos_d = pos_f[a]
    pos_a = pos_f[b]
    pos_mid = pos_d + 0.5 * v_da
    # Note: ATLAS analysis protein is whole/centered in box; cKDTree Euclidean within 8 A matches MIC
    pack_d6 = np.array([len(x) for x in tree.query_ball_point(pos_d, 6.0)], dtype=np.float32)
    pack_d8 = np.array([len(x) for x in tree.query_ball_point(pos_d, 8.0)], dtype=np.float32)
    pack_a6 = np.array([len(x) for x in tree.query_ball_point(pos_a, 6.0)], dtype=np.float32)
    pack_a8 = np.array([len(x) for x in tree.query_ball_point(pos_a, 8.0)], dtype=np.float32)
    pack_m6 = np.array([len(x) for x in tree.query_ball_point(pos_mid, 6.0)], dtype=np.float32)
    pack_m8 = np.array([len(x) for x in tree.query_ball_point(pos_mid, 8.0)], dtype=np.float32)

    base20 = np.column_stack([
        d0, ca_d, cb_d, np.log1p(sep), (sep <= 4).astype(np.float32), (sep <= 2).astype(np.float32),
        (d_kind[ii] == 0).astype(np.float32), (a_kind[jj] == 0).astype(np.float32),
        (d_kind[ii] == 2).astype(np.float32), (a_kind[jj] == 1).astype(np.float32),
        topo["is_charged"][dri], topo["is_charged"][arj],
        cos_d, cos_a,
        pack_d6, pack_d8, pack_a6, pack_a8, pack_m6, pack_m8,
    ]).astype(np.float32)

    if not include_k302:
        return base20

    # K3-02 H-bond network features (pre-existing protein H-bonds <= 3.5 A across distinct residues)
    hb_mat = (dd_all <= 3.5) & (d_res[:, None] != a_res[None, :])
    n_res = topo["n_res"]
    # Per-residue outgoing (as donor) and incoming (as acceptor), split by partner backbone vs sidechain
    res_out = np.zeros(n_res, dtype=np.float32)
    res_in = np.zeros(n_res, dtype=np.float32)
    res_bb = np.zeros(n_res, dtype=np.float32)
    res_sc = np.zeros(n_res, dtype=np.float32)
    # Per-atom competing H-bonds
    atom_d_hb = hb_mat.sum(axis=1).astype(np.float32)
    atom_a_hb = hb_mat.sum(axis=0).astype(np.float32)
    # Pair-specific count between residue pair (dri, arj) to subtract self-partner H-bonds
    hi, hj = np.where(hb_mat)
    pair_res_hb = {}
    for di_k, aj_k in zip(hi, hj):
        rd, ra = int(d_res[di_k]), int(a_res[aj_k])
        res_out[rd] += 1.0
        res_in[ra] += 1.0
        if a_kind[aj_k] == 0:
            res_bb[rd] += 1.0
        else:
            res_sc[rd] += 1.0
        if d_kind[di_k] == 0:
            res_bb[ra] += 1.0
        else:
            res_sc[ra] += 1.0
        pair_res_hb[(rd, ra)] = pair_res_hb.get((rd, ra), 0.0) + 1.0

    # Exclude partner residue itself for each candidate (dri, arj)
    self_d_to_a = np.array([pair_res_hb.get((int(r1), int(r2)), 0.0) for r1, r2 in zip(dri, arj)], dtype=np.float32)
    self_a_to_d = np.array([pair_res_hb.get((int(r2), int(r1)), 0.0) for r1, r2 in zip(dri, arj)], dtype=np.float32)
    self_pair_hb = hb_mat[ii, jj].astype(np.float32)

    d_out = np.maximum(res_out[dri] - self_d_to_a, 0.0)
    d_in = np.maximum(res_in[dri] - self_a_to_d, 0.0)
    a_out = np.maximum(res_out[arj] - self_a_to_d, 0.0)
    a_in = np.maximum(res_in[arj] - self_d_to_a, 0.0)
    comp_d = np.maximum(atom_d_hb[ii] - self_pair_hb, 0.0)
    comp_a = np.maximum(atom_a_hb[jj] - self_pair_hb, 0.0)

    k302 = np.column_stack([
        d_out, d_in, d_out + d_in,
        a_out, a_in, a_out + a_in,
        res_bb[dri] + res_bb[arj],
        res_sc[dri] + res_sc[arj],
        comp_d, comp_a,
    ]).astype(np.float32)
    return np.column_stack([base20, k302]).astype(np.float32)


def summarize_hist6(series: np.ndarray, thresh: float) -> np.ndarray:
    """series: (N, T) -> 6 stats: first, mean, std, min, last-first, frac<=thresh."""
    return np.column_stack([
        series[:, 0],
        series.mean(axis=1),
        series.std(axis=1),
        series.min(axis=1),
        series[:, -1] - series[:, 0],
        (series <= thresh).mean(axis=1),
    ]).astype(np.float32)


def process_protein(prot: str):
    out_file = OUT_CACHE / f"{prot}.npz"
    if out_file.exists():
        return (prot, "cached", 0.0)
    t0 = time.time()
    pdir = ATLAS_ROOT / prot
    meta = json.loads((pdir / "done.json").read_text(encoding="utf-8"))
    pdb_path = pdir / meta["pdb"]
    xtc_files = sorted(meta["xtc_files"])
    assert len(xtc_files) == 3, (prot, xtc_files)
    topo = parse_topology(pdb_path)

    # All frames we ever need from a 1001-frame trajectory
    needed_frames = sorted(set(
        list(range(11))
        + [10 + h for h in HORIZONS]
        + [k + h for k in PROTO_B_ORIGINS for h in HORIZONS]
    ))
    frame_to_idx = {f: idx for idx, f in enumerate(needed_frames)}

    da = topo["donors"]
    ac = topo["acceptors"]
    d_at, d_res = da[:, 0], da[:, 1]
    a_at, a_res = ac[:, 0], ac[:, 1]
    ca_idx = topo["ca_idx"]

    bundle = {"protein": prot, "n_res": np.int32(topo["n_res"])}

    for rep_i, xtc_name in enumerate(xtc_files, 1):
        u = mda.Universe(str(pdb_path), str(pdir / xtc_name))
        n_frames = len(u.trajectory)
        assert n_frames >= 911, f"{prot} R{rep_i}: only {n_frames} frames"

        # Read only needed frames once
        pos_buf = np.empty((len(needed_frames), u.atoms.n_atoms, 3), dtype=np.float32)
        box_buf = [None] * len(needed_frames)
        for fi, f in enumerate(needed_frames):
            ts = u.trajectory[f]
            pos_buf[fi] = ts.positions
            if ts.dimensions is not None and ts.dimensions[0] > 1.0:
                box_buf[fi] = triclinic_vectors(ts.dimensions).astype(np.float64)
            else:
                box_buf[fi] = None

        def get_fb(f):
            idx = frame_to_idx[f]
            return pos_buf[idx].astype(np.float64), box_buf[idx]

        # ---- 1 & 2. Fixed origin frame 10: HB Formation & HB Dissociation ----
        p10, b10 = get_fb(10)
        dd10 = mic_dist(p10[d_at[:, None]], p10[a_at[None, :]], b10)
        diff_res = d_res[:, None] != a_res[None, :]

        for mode, mask in (
            ("hb_form", (dd10 > 3.5) & (dd10 <= 6.0) & diff_res),
            ("hb_diss", (dd10 <= 3.5) & diff_res),
        ):
            ii, jj = np.where(mask)
            a, b = d_at[ii], a_at[jj]
            X30 = compute_public_features(p10, b10, topo, ii, jj, dd10, include_k302=True)
            ser = np.empty((len(ii), 11), dtype=np.float32)
            for f in range(11):
                pf, bf = get_fb(f)
                ser[:, f] = mic_dist(pf[a], pf[b], bf)
            y = np.empty((len(ii), len(HORIZONS)), dtype=np.uint8)
            for hi, h in enumerate(HORIZONS):
                pf, bf = get_fb(10 + h)
                df = mic_dist(pf[a], pf[b], bf)
                y[:, hi] = (df <= 3.5) if mode == "hb_form" else (df > 3.5)
            bundle[f"r{rep_i}_{mode}_X"] = X30
            bundle[f"r{rep_i}_{mode}_series"] = ser
            bundle[f"r{rep_i}_{mode}_y"] = y
            bundle[f"r{rep_i}_{mode}_pair"] = np.column_stack([a, b]).astype(np.int32)

        # ---- 3. Protocol B: Moving origins k in {0, 1, 2, 5} for HB Formation ----
        for k in PROTO_B_ORIGINS:
            pk, bk = get_fb(k)
            ddk = mic_dist(pk[d_at[:, None]], pk[a_at[None, :]], bk)
            ii, jj = np.where((ddk > 3.5) & (ddk <= 6.0) & diff_res)
            a, b = d_at[ii], a_at[jj]
            X20 = compute_public_features(pk, bk, topo, ii, jj, ddk, include_k302=False)
            ser_k = np.empty((len(ii), k + 1), dtype=np.float32)
            for f in range(k + 1):
                pf, bf = get_fb(f)
                ser_k[:, f] = mic_dist(pf[a], pf[b], bf)
            h6 = summarize_hist6(ser_k, 3.5)
            yk = np.empty((len(ii), len(HORIZONS)), dtype=np.uint8)
            for hi, h in enumerate(HORIZONS):
                pf, bf = get_fb(k + h)
                yk[:, hi] = (mic_dist(pf[a], pf[b], bf) <= 3.5)
            bundle[f"r{rep_i}_pb_k{k}_X"] = X20
            bundle[f"r{rep_i}_pb_k{k}_h6"] = h6
            bundle[f"r{rep_i}_pb_k{k}_y"] = yk

        # ---- 4. Salt-bridge formation at frame 10 (4.0 < d <= 10.0, |i-j| >= 2) ----
        pos_g = topo["pos_groups"]
        neg_g = topo["neg_groups"]
        sb_pairs = []
        sb_d10 = []
        for rk_p, idx_p in pos_g:
            for rk_n, idx_n in neg_g:
                if abs(rk_p - rk_n) < 2:
                    continue
                d_val = mic_dist(p10[idx_p[:, None]], p10[idx_n[None, :]], b10).min()
                if 4.0 < d_val <= 10.0:
                    sb_pairs.append((rk_p, idx_p, rk_n, idx_n))
                    sb_d10.append(d_val)
        if sb_pairs:
            m_sb = len(sb_pairs)
            sb_ser = np.empty((m_sb, 11), dtype=np.float32)
            sb_y = np.empty((m_sb, len(HORIZONS)), dtype=np.uint8)
            sb_X = np.empty((m_sb, 5), dtype=np.float32)
            for idx_pair, (rk_p, idx_p, rk_n, idx_n) in enumerate(sb_pairs):
                ca_d = float(mic_dist(p10[ca_idx[rk_p]], p10[ca_idx[rk_n]], b10))
                sb_X[idx_pair] = [
                    sb_d10[idx_pair], ca_d, np.log1p(abs(rk_p - rk_n)),
                    1.0 if topo["resnames"][rk_p] == "ARG" else 0.0,
                    1.0 if topo["resnames"][rk_n] == "GLU" else 0.0,
                ]
                for f in range(11):
                    pf, bf = get_fb(f)
                    sb_ser[idx_pair, f] = mic_dist(pf[idx_p[:, None]], pf[idx_n[None, :]], bf).min()
                for hi, h in enumerate(HORIZONS):
                    pf, bf = get_fb(10 + h)
                    sb_y[idx_pair, hi] = int(mic_dist(pf[idx_p[:, None]], pf[idx_n[None, :]], bf).min() <= 4.0)
            bundle[f"r{rep_i}_sb_X"] = sb_X
            bundle[f"r{rep_i}_sb_series"] = sb_ser
            bundle[f"r{rep_i}_sb_y"] = sb_y
            bundle[f"r{rep_i}_sb_pair"] = np.array([[p[0], p[2]] for p in sb_pairs], dtype=np.int32)
        else:
            bundle[f"r{rep_i}_sb_X"] = np.zeros((0, 5), dtype=np.float32)
            bundle[f"r{rep_i}_sb_series"] = np.zeros((0, 11), dtype=np.float32)
            bundle[f"r{rep_i}_sb_y"] = np.zeros((0, len(HORIZONS)), dtype=np.uint8)
            bundle[f"r{rep_i}_sb_pair"] = np.zeros((0, 2), dtype=np.int32)

        # ---- 5. Contact formation at frame 10 (CA <= 12.0, heavy min > 4.5, |i-j| >= 2) ----
        ca_mat = mic_dist(p10[ca_idx[:, None]], p10[ca_idx[None, :]], b10)
        n_res = topo["n_res"]
        heavy_by_res = topo["heavy_by_res"]
        ci, cj = np.where((ca_mat <= 12.0) & (np.subtract.outer(np.arange(n_res), np.arange(n_res)) <= -2))
        ct_pairs = []
        ct_X_list = []
        for r1, r2 in zip(ci, cj):
            h1, h2 = heavy_by_res[r1], heavy_by_res[r2]
            d_min = float(mic_dist(p10[h1[:, None]], p10[h2[None, :]], b10).min())
            if d_min > 4.5:
                ct_pairs.append((int(r1), int(r2)))
                ct_X_list.append([
                    d_min, float(ca_mat[r1, r2]), np.log1p(r2 - r1),
                    float(len(h1)), float(len(h2)),
                ])
        m_ct = len(ct_pairs)
        ct_ser = np.empty((m_ct, 11), dtype=np.float32)
        ct_y = np.empty((m_ct, len(HORIZONS)), dtype=np.uint8)
        for idx_pair, (r1, r2) in enumerate(ct_pairs):
            h1, h2 = heavy_by_res[r1], heavy_by_res[r2]
            for f in range(11):
                pf, bf = get_fb(f)
                ct_ser[idx_pair, f] = mic_dist(pf[h1[:, None]], pf[h2[None, :]], bf).min()
            for hi, h in enumerate(HORIZONS):
                pf, bf = get_fb(10 + h)
                ct_y[idx_pair, hi] = int(mic_dist(pf[h1[:, None]], pf[h2[None, :]], bf).min() <= 4.5)
        bundle[f"r{rep_i}_ct_X"] = np.asarray(ct_X_list, dtype=np.float32).reshape(m_ct, 5)
        bundle[f"r{rep_i}_ct_series"] = ct_ser
        bundle[f"r{rep_i}_ct_y"] = ct_y
        bundle[f"r{rep_i}_ct_pair"] = np.asarray(ct_pairs, dtype=np.int32).reshape(m_ct, 2)

    np.savez_compressed(out_file, **bundle)
    return (prot, f"ok (hb={len(bundle['r1_hb_form_y'])}, ct={len(bundle['r1_ct_y'])}, sb={len(bundle['r1_sb_y'])})", time.time() - t0)


def main():
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    lines = [ln.strip() for ln in PROT_LIST_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    test100 = lines[8:108]
    ready = [p for p in test100 if (ATLAS_ROOT / p / "done.json").exists()]
    print(f"Found {len(ready)}/{len(test100)} downloaded proteins.", flush=True)
    t_start = time.time()
    ok = cached = fail = 0
    workers = min(8, max(1, len(ready)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_protein, p): p for p in ready}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            p = futs[fut]
            try:
                prot, status, dt = fut.result()
                if status == "cached":
                    cached += 1
                else:
                    ok += 1
                if status != "cached" or i % 20 == 0 or i == len(ready):
                    print(f"[{i}/{len(ready)}] {prot}: {status} ({dt:.1f}s)", flush=True)
            except Exception as e:
                fail += 1
                print(f"[{i}/{len(ready)}] {p}: FAIL {e}", flush=True)
    print(f"Extraction finished in {time.time() - t_start:.1f}s | ok={ok} cached={cached} fail={fail}", flush=True)
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
