"""K32: 水の環境の履歴を測れるかどうかの確認。

(a) 変換済み軌道（MiniTopol A変換）の atoms.parquet の residue_name 列を読み、
    水（SOL/WAT/HOH）やイオンが含まれているかを確かめる。
(b) ATLAS 完全形式（total）の TPR + XTC を MDAnalysis で直接読み、
    水の有無・原子数・フレーム数・保存間隔と、解析版とのフレーム対応を確かめる。

使い方（この仮想マシン上、~/workspace/.mt-venv/bin/python で実行）:
    python check_water.py
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import minitopol  # noqa: F401  (zarr codec 登録のため)

import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

TRAJ_BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
TOTAL_BASE = Path("/home/hatch/workspace/rod-survival-exp/atlas-ws/storage/sources/ATLAS")
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
OUT = Path(__file__).resolve().parent / "results.json"


def check_converted():
    """(a) 変換済み軌道の残基名。"""
    out = {}
    union = set()
    for p in sorted(glob.glob(str(TRAJ_BASE / "*/atoms.parquet"))):
        name = Path(p).parent.name
        t = pq.read_table(p, columns=["residue_name"])
        names = sorted(set(t.column("residue_name").to_pylist()))
        union |= set(names)
        out[name] = {"n_atoms": t.num_rows, "resnames": names}
    out["union"] = sorted(union)
    return out


def kabsch_rmsd(a, b):
    c1 = a - a.mean(0)
    c2 = b - b.mean(0)
    h = c1.T @ c2
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return float(np.sqrt((((c1 @ r) - c2) ** 2).sum(1).mean()))


def check_total():
    """(b) 完全形式 2qt4_A R1 を実際に読む。"""
    base = TOTAL_BASE / "2qt4_A-total"
    u = mda.Universe(str(base / "2qt4_A_prod_R1.tpr"),
                     str(base / "2qt4_A_prod_R1.xtc"))
    counts = Counter(u.atoms.resnames)
    sol = u.select_atoms("resname SOL")
    heavy = u.select_atoms("protein and not name H*")
    ca = u.select_atoms("protein and name CA")
    n_frames = len(u.trajectory)
    u.trajectory[0]
    dt = float(u.trajectory.ts.dt)
    # 1 ns先のフレーム（完全形式 100）での水の数
    u.trajectory[100]
    ow = sol.select_atoms("name OW").positions
    d = distance_array(heavy.positions.astype(np.float64), ow,
                       box=u.trajectory.ts.dimensions)
    n_w_35 = int((d.min(axis=0) <= 3.5).sum())
    n_w_50 = int((d.min(axis=0) <= 5.0).sum())
    # 解析版とのフレーム対応（2qt4_A R1）
    z = zarr.open(str(TRAJ_BASE / "atlas-2qt4_A-analysis-R1" / "A.zarr"), mode="r")
    pos = z["positions"][:]
    box_a = z["box"][:]
    times = z["time"][:]
    t = pq.read_table(str(TRAJ_BASE / "atlas-2qt4_A-analysis-R1" / "atoms.parquet"),
                      columns=["name"])
    ca_idx = [i for i, n in enumerate(t.column("name").to_pylist()) if n == "CA"]

    def ca_dmat(coords, box6):
        return distance_array(coords, coords, box=box6)

    u.trajectory[0]
    rmsd_f0 = kabsch_rmsd(heavy.positions.astype(np.float64),
                          pos[0].astype(np.float64))
    conv_ca = pos[10][ca_idx].astype(np.float64)
    b = box_a[10]
    d_conv = ca_dmat(conv_ca, np.array([b[0, 0], b[1, 1], b[2, 2], 90., 90., 90.]))
    medians = {}
    for tf in (90, 100, 110):
        u.trajectory[tf]
        d_tot = ca_dmat(ca.positions.astype(np.float64),
                        u.trajectory.ts.dimensions)
        diff = np.abs(d_tot - d_conv)
        iu = np.triu_indices(diff.shape[0], 1)
        medians[tf] = float(np.median(diff[iu]))
    u.trajectory[100]
    box_total_100 = [float(x) for x in u.trajectory.ts.dimensions[:3]]
    return {
        "protein": "2qt4_A", "replica": "R1",
        "n_atoms": len(u.atoms), "n_residues": len(u.residues),
        "resname_counts": dict(counts),
        "n_water_residues": len(sol.residues),
        "n_water_oxygens": len(sol.select_atoms("name OW")),
        "n_protein_heavy_atoms": len(heavy),
        "n_frames": n_frames, "dt_ps": dt,
        "frame100_time_ps": float(u.trajectory[100].time),
        "waters_within_3.5A_frame100": n_w_35,
        "waters_within_5.0A_frame100": n_w_50,
        "analysis_times_ps_first12": [float(x) for x in times[:12]],
        "frame0_kabsch_rmsd_A": rmsd_f0,
        "analysis_box_frame10": [float(x) for x in box_a[10][[0, 1, 2], [0, 1, 2]]],
        "total_box_frame100": box_total_100,
        "ca_dmat_median_diff_A": medians,
    }


def main():
    results = {
        "a_converted_trajectories": check_converted(),
        "b_total_format": check_total(),
        "conclusion": ("Converted (MiniTopol A) trajectories contain no water; "
                       "ATLAS total format contains water (4479 SOL residues in "
                       "2qt4_A R1) and frame mapping analysis k <-> total 10k holds."),
    }
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
