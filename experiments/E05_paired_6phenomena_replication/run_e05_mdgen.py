"""E05 Step 4: Run MDGen (official atlas.ckpt, 250 frames / 100 ns) on all 49 unseen atlas_test proteins,
extracting all 6 phenomena (hb_form, hb_diss, ct_form, ct_diss, sb_form, sb_diss) from ONE rollout per protein.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import MDAnalysis as mda
import numpy as np
import torch

MDGEN_REPO = Path(r"d:\KineScope\experiments\E04_public_generator_static_redundancy\mdgen_repo")
CKPT_PATH = Path(r"d:\KineScope\experiments\E04_public_generator_static_redundancy\checkpoints\atlas.ckpt")
sys.path.insert(0, str(MDGEN_REPO))
sys.path.insert(0, r"d:\KineScope\experiments\E01_true_short_md_guidance")
sys.path.insert(0, r"d:\KineScope\experiments\E05_paired_6phenomena_replication")

import extract_e01_e02 as ext_mod
import fetch_e05_atlas
from mdgen import residue_constants as rc
from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.wrapper import NewMDGenWrapper

ATLAS_ROOT = Path(r"d:\KineScope\experiments\shared_data\atlas")
REAL_CACHE = Path(r"d:\KineScope\experiments\E05_paired_6phenomena_replication\cache_real")
OUT_CACHE = Path(r"d:\KineScope\experiments\E05_paired_6phenomena_replication\cache_mdgen")

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "HSD": "H",
    "HSE": "H", "HSP": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T",
    "TRP": "W", "TYR": "Y", "VAL": "V",
}
STD_RES3 = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS"}
THRESH = {
    "hb_form": 3.5, "hb_diss": 3.5,
    "ct_form": 4.5, "ct_diss": 4.5,
    "sb_form": 4.0, "sb_diss": 4.0,
}


def frame_to_atom14_and_maps(u: mda.Universe):
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
    heavy14_by_res = []
    pos14_by_res = {}
    neg14_by_res = {}
    for i, (res, r3, nm) in enumerate(res_list):
        seq_chars.append(AA3_TO_1[r3])
        names14 = rc.restype_name_to_atom14_names[r3]
        h_j = []
        p_set = ext_mod.POS_RES.get(r3, set())
        n_set = ext_mod.NEG_RES.get(r3, set())
        p_j, n_j = [], []
        for aname, atom in nm.items():
            aname14 = "CD1" if (r3 == "ILE" and aname == "CD") else aname
            if aname14 in names14:
                j = names14.index(aname14)
                arr[0, i, j] = atom.position.astype(np.float32)
                orig_atom_to_14[int(atom.index)] = (i, j)
                h_j.append(j)
                if aname in p_set:
                    p_j.append(j)
                if aname in n_set:
                    n_j.append(j)
        heavy14_by_res.append(np.array(sorted(h_j), dtype=np.int32))
        if p_j:
            pos14_by_res[i] = np.array(sorted(p_j), dtype=np.int32)
        if n_j:
            neg14_by_res[i] = np.array(sorted(n_j), dtype=np.int32)
    return arr, "".join(seq_chars), orig_atom_to_14, heavy14_by_res, pos14_by_res, neg14_by_res


def stats6(D: np.ndarray, thresh: float) -> np.ndarray:
    if len(D) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return np.column_stack([
        D[:, 0], D.mean(1), D.std(1), D.min(1), D[:, -1] - D[:, 0], (D <= thresh).mean(1)
    ]).astype(np.float32)


def min_dist_traj_from_atom14(gen_xyz: np.ndarray, pairs: np.ndarray, grp_a, grp_b) -> np.ndarray:
    """gen_xyz: (T, L, 14, 3), pairs: (N, 2) res indices -> (N, T) min distance trajectory."""
    T = gen_xyz.shape[0]
    n = len(pairs)
    if n == 0:
        return np.zeros((0, T), dtype=np.float32)
    out = np.empty((n, T), dtype=np.float32)
    for idx in range(n):
        r1, r2 = int(pairs[idx, 0]), int(pairs[idx, 1])
        j1 = grp_a[r1]
        j2 = grp_b[r2]
        p1 = gen_xyz[:, r1, j1, :]  # (T, a, 3)
        p2 = gen_xyz[:, r2, j2, :]  # (T, b, 3)
        diff = p1[:, :, None, :] - p2[:, None, :, :]
        d2 = (diff * diff).sum(axis=-1).reshape(T, -1)
        out[idx] = np.sqrt(d2.min(axis=1))
    return out


@torch.no_grad()
def run_one(model, prot: str, num_frames: int = 250):
    out_npz = OUT_CACHE / f"{prot}.npz"
    if out_npz.exists():
        return (prot, "cached", 0.0)

    t0 = time.time()
    real_f = REAL_CACHE / f"{prot}.npz"
    if not real_f.exists():
        return (prot, "skip_no_real", 0.0)
    ext = np.load(real_f, allow_pickle=True)

    pdir = ATLAS_ROOT / prot
    meta = json.loads((pdir / "done.json").read_text(encoding="utf-8"))
    pdb_path = pdir / meta["pdb"]
    r1_xtc = pdir / sorted(meta["xtc_files"])[0]
    u = mda.Universe(str(pdb_path), str(r1_xtc))
    u.trajectory[10]

    arr0, seqres, orig_to_14, heavy14, pos14, neg14 = frame_to_atom14_and_maps(u)
    L = len(seqres)
    assert L == int(ext["n_res"]), f"{prot}: L mismatch {L} vs {int(ext['n_res'])}"

    # Verify origin distances on all 6 phenomena before running GPU rollout
    max_err_all = 0.0
    hb_maps = {}
    for mode in ("hb_form", "hb_diss"):
        pairs = ext[f"{mode}_pair"]
        vm = np.array([(int(a) in orig_to_14 and int(b) in orig_to_14) for a, b in pairs], dtype=bool)
        ar = np.array([orig_to_14[int(a)][0] if v else 0 for (a, _), v in zip(pairs, vm)], dtype=np.int32)
        aj = np.array([orig_to_14[int(a)][1] if v else 0 for (a, _), v in zip(pairs, vm)], dtype=np.int32)
        br = np.array([orig_to_14[int(b)][0] if v else 0 for (_, b), v in zip(pairs, vm)], dtype=np.int32)
        bj = np.array([orig_to_14[int(b)][1] if v else 0 for (_, b), v in zip(pairs, vm)], dtype=np.int32)
        if vm.any():
            d0 = np.linalg.norm(arr0[0, ar[vm], aj[vm]] - arr0[0, br[vm], bj[vm]], axis=1)
            err = float(np.abs(d0 - ext[f"{mode}_series"][vm, 10]).max())
            max_err_all = max(max_err_all, err)
            assert err < 0.05, f"{prot} {mode}: origin mismatch {err:.4f}A"
        hb_maps[mode] = (ar, aj, br, bj, vm)

    for mode in ("ct_form", "ct_diss"):
        p = ext[f"{mode}_pair"]
        if len(p):
            d0 = min_dist_traj_from_atom14(arr0, p, heavy14, heavy14)[:, 0]
            # Note: C-terminal OXT is absent from atom14; allow small diff only on terminal residue if any
            diff_arr = np.abs(d0 - ext[f"{mode}_series"][:, 10])
            non_term = (p[:, 0] < L - 1) & (p[:, 1] < L - 1)
            if non_term.any():
                err = float(diff_arr[non_term].max())
                max_err_all = max(max_err_all, err)
                assert err < 0.05, f"{prot} {mode}: origin mismatch {err:.4f}A"

    for mode in ("sb_form", "sb_diss"):
        p = ext[f"{mode}_pair"]
        if len(p):
            d0 = min_dist_traj_from_atom14(arr0, p, pos14, neg14)[:, 0]
            err = float(np.abs(d0 - ext[f"{mode}_series"][:, 10]).max())
            max_err_all = max(max_err_all, err)
            assert err < 0.05, f"{prot} {mode}: origin mismatch {err:.4f}A"

    # Build MDGen batch
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
    used_bf16 = False
    try:
        if L > 140:
            used_bf16 = True
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                atom14_out, _ = model.inference(batch)
        else:
            atom14_out, _ = model.inference(batch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        used_bf16 = True
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            atom14_out, _ = model.inference(batch)
    gen_xyz = atom14_out[0].detach().cpu().float().numpy()  # (T, L, 14, 3)

    out = {
        "protein": prot,
        "n_res": np.int32(L),
        "used_bf16": np.bool_(used_bf16),
        "max_d0_err": np.float32(max_err_all),
        "elapsed_s": np.float32(time.time() - t0),
    }

    for mode in ("hb_form", "hb_diss"):
        ar, aj, br, bj, vm = hb_maps[mode]
        if len(ar):
            pa = gen_xyz[:, ar, aj, :]
            pb = gen_xyz[:, br, bj, :]
            D = np.linalg.norm(pa - pb, axis=-1).T.astype(np.float32)
        else:
            D = np.zeros((0, num_frames), dtype=np.float32)
        out[f"{mode}_short"] = stats6(D[:, :4], THRESH[mode])
        out[f"{mode}_10ns"] = stats6(D[:, :26], THRESH[mode])
        out[f"{mode}_full"] = stats6(D, THRESH[mode])
        out[f"{mode}_valid"] = vm

    for mode in ("ct_form", "ct_diss"):
        p = ext[f"{mode}_pair"]
        D = min_dist_traj_from_atom14(gen_xyz, p, heavy14, heavy14)
        out[f"{mode}_short"] = stats6(D[:, :4], THRESH[mode])
        out[f"{mode}_10ns"] = stats6(D[:, :26], THRESH[mode])
        out[f"{mode}_full"] = stats6(D, THRESH[mode])
        out[f"{mode}_valid"] = np.ones(len(p), dtype=bool)

    for mode in ("sb_form", "sb_diss"):
        p = ext[f"{mode}_pair"]
        D = min_dist_traj_from_atom14(gen_xyz, p, pos14, neg14)
        out[f"{mode}_short"] = stats6(D[:, :4], THRESH[mode])
        out[f"{mode}_10ns"] = stats6(D[:, :26], THRESH[mode])
        out[f"{mode}_full"] = stats6(D, THRESH[mode])
        out[f"{mode}_valid"] = np.ones(len(p), dtype=bool)

    np.savez_compressed(out_npz, **out)
    return (prot, f"ok (L={L}, bf16={used_bf16}, d0_err={max_err_all:.4f}A)", time.time() - t0)


def main():
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    chosen = [p for p, _ in fetch_e05_atlas.select_e05_proteins(250)]
    print(f"Loading MDGen {CKPT_PATH} on {torch.cuda.get_device_name(0)}...", flush=True)
    torch.serialization.add_safe_globals([argparse.Namespace])
    model = NewMDGenWrapper.load_from_checkpoint(str(CKPT_PATH), map_location="cuda")
    model.eval().to("cuda")
    num_frames = int(getattr(model.args, "num_frames", 250))
    print(f"E05 MDGen: running 100ns rollouts (all 6 phenomena) on {len(chosen)} unseen atlas_test proteins...", flush=True)

    t0 = time.time()
    ok = cached = fail = 0
    for i, p in enumerate(chosen, 1):
        try:
            prot, status, dt = run_one(model, p, num_frames=num_frames)
            if status == "cached":
                cached += 1
            else:
                ok += 1
            print(f"[{i}/{len(chosen)}] {prot}: {status} ({dt:.1f}s)", flush=True)
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(chosen)}] {p}: FAIL {e}", flush=True)
            torch.cuda.empty_cache()
    print(f"E05 MDGen finished in {time.time() - t0:.1f}s | ok={ok} cached={cached} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
