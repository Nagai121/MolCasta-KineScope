"""K3-05 補足: 起点フレーム（解析版フレーム10、1 ns の時点）での
ペアの間にある水の量の抽出。

静的な予測の入力は起点フレーム（フレーム10）なので、「初期構造」の
指すフレームを起点フレームに統一する。ここではまず起点フレームでの
有無と数を測る。最初の 0.1 ns（フレーム0と1）での滞在時間は、
比較用として extract_water_between.py の結果に残してある。

定義（結果を見る前に固定）:
- 完全形式（total）の TPR + XTC を MDAnalysis で直接読む。
  完全形式のフレーム 100 が、解析版のフレーム 10（起点フレーム）に
  対応する（K36 と同じ対応）。
- 候補の同定は K06(b) の保存済み特徴量（k06b_out の r{r}_pair の
  第0・第1列＝2原子の通し番号）を使う。
- 距離はフレームごとの箱で最小イメージ。
- 量（候補ごと）:
  - n_bridge_f100: 起点フレームで、2原子の両方から 3.5 Å 以内にある
    OW の数
  - has_bridge_f100: 起点フレームで橋渡し水が1個以上あるか（0/1）
- 対応の確認: 起点フレームのペア距離を完全形式から計算し、
  G05 の系列の行10（起点フレームの距離＝K06(b) の pair[:,2] と同じ）と
  突き合わせる。

使い方: python extract_water_f100.py <protein> [<protein> ...]
出力は作業ディレクトリの <protein>__f100.npz。
"""
import argparse
import time
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

K06B_OUT = Path("/home/hatch/workspace/k06b/run/k06b_out")
G05_SERIES = Path("/home/hatch/workspace/kinescope/work/g05_series")
HEAD_ROOT = Path("/home/hatch/workspace/kinescope/work/k3_05_total_head")
OUT = Path("/home/hatch/workspace/kinescope/work/k3_05_water")
FRAME_WANT = 100  # 完全形式のフレーム番号（解析版のフレーム10＝起点フレーム）


def extract(prot, rep):
    base = HEAD_ROOT / prot
    u = mda.Universe(str(base / f"{prot}_prod_R{rep}.tpr"),
                     str(base / f"{prot}_prod_R{rep}.xtc"))
    heavy = u.select_atoms("protein and not name H*")
    ow = u.select_atoms("resname SOL and name OW")
    with np.load(K06B_OUT / f"{prot}.npz") as z:
        pair = z[f"r{rep}_pair"]
    da = pair[:, 0].astype(np.intp)
    aa = pair[:, 1].astype(np.intp)
    with np.load(G05_SERIES / f"{prot}.npz") as z:
        series = z[f"r{rep}_series"]  # (11, N) 解析版フレーム 0〜10
    u.trajectory[FRAME_WANT]
    ts = u.trajectory.ts
    box = ts.dimensions
    dpos = heavy.positions[da].astype(np.float64)
    apos = heavy.positions[aa].astype(np.float64)
    owpos = ow.positions.astype(np.float64)
    dd = distance_array(dpos, owpos, box=box)
    da_ = distance_array(apos, owpos, box=box)
    bridge = (dd <= 3.5) & (da_ <= 3.5)
    n = bridge.sum(axis=1)
    # 対応の確認: ペア距離（最小イメージ）vs G05 系列の行10
    delta = apos - dpos
    delta -= box[:3].astype(np.float64) * np.rint(delta / box[:3].astype(np.float64))
    dpair = np.linalg.norm(delta, axis=1)
    dist_check = float(np.abs(dpair - series[10]).max())
    return {
        "n_bridge_f100": n.astype(np.float32),
        "has_bridge_f100": (n > 0).astype(np.float32),
        "dist_check_f100": dist_check,
        "n_ow": len(ow), "n_heavy": len(heavy), "n_cand": len(da),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proteins", nargs="+")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for prot in args.proteins:
        out_file = OUT / f"{prot}__f100.npz"
        if out_file.exists():
            print(prot, "skip (extracted)", flush=True)
            continue
        t0 = time.time()
        store = {}
        for rep in (1, 2, 3):
            r = extract(prot, rep)
            for k, v in r.items():
                store[f"r{rep}_{k}"] = v
            print(prot, f"R{rep}: cand {r['n_cand']} OW {r['n_ow']} "
                  f"dist_check f100 {r['dist_check_f100']:.4f} "
                  f"mean n_bridge_f100 {r['n_bridge_f100'].mean():.4f}", flush=True)
            assert r["dist_check_f100"] < 0.05, (prot, rep, r["dist_check_f100"])
        np.savez_compressed(out_file, **store)
        print(prot, f"saved ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
