"""K3-05: ペアの間にある水の量の抽出。

強い土台（K06(b)）の水素結合の形成の候補ペアについて、初期構造
（履歴のフレーム0）と最初の 0.1 ns（フレーム0と1）で、ペアの2原子の
両方から 3.5 Å 以内にある水分子（SOL の酸素 OW）の量を取る。

定義（結果を見る前に固定）:
- 完全形式（total）の TPR + XTC を MDAnalysis で直接読む
  （K36 と同じ例外の範囲）。完全形式のフレーム 0 と 10 が、解析版の
  フレーム 0 と 1（履歴の 0 ns と 0.1 ns）に対応する。
- 候補の同定は K06(b) の保存済み特徴量（k06b_out の r{r}_pair の
  第0・第1列＝2原子の通し番号）を使う。通し番号は完全形式の
  タンパク質重原子の並びに対応する（K36 で確認済みの対応）。
  対応の確認として、フレーム0と1のペア距離を完全形式から計算し、
  G05 の系列（g05_series）の行0・行1と突き合わせる。
- 距離はフレームごとの箱で最小イメージ（MDAnalysis の distance_array
  に箱を渡す。K36 と同じ）。
- 量（候補ごと）:
  - n_bridge_f0 / n_bridge_f1: フレーム0 / フレーム1で、2原子の両方から
    3.5 Å 以内にある OW の数
  - has_bridge_f0: フレーム0で橋渡し水が1個以上あるか（0/1）
  - frac_frames_bridge: フレーム0と1のうち、橋渡し水が1個以上ある
    フレームの割合（0, 0.5, 1 のいずれか）
  - persistence: フレーム0の橋渡し水のうち、フレーム1でも橋渡し水で
    あるものの割合（フレーム0に橋渡し水が無い候補では 0 とする）
  - n_bridge_mean: n_bridge_f0 と n_bridge_f1 の平均

使い方: python extract_water_between.py <protein> [<protein> ...]
入力の置き場は --src で切り替える（head = 範囲リクエストで取った先頭、
full = 丸ごと落とした完全形式）。出力は作業ディレクトリの
<protein>.npz（レプリカごとの量と、対応確認の最大差）。
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
FULL_ROOT = Path("/home/hatch/workspace/rod-survival-exp/atlas-ws/storage/sources/ATLAS")
OUT = Path("/home/hatch/workspace/kinescope/work/k3_05_water")
FRAMES_WANT = (0, 10)  # 完全形式のフレーム番号（解析版の 0 と 1）


def extract(prot, rep, src_root, src_suffix=""):
    base = src_root / (f"{prot}{src_suffix}")
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
    n_cand = len(da)
    got = {}
    dist_check = {}
    for ts in u.trajectory:
        if ts.frame > FRAMES_WANT[-1]:
            break
        if ts.frame not in FRAMES_WANT:
            continue
        box = ts.dimensions
        dpos = heavy.positions[da].astype(np.float64)
        apos = heavy.positions[aa].astype(np.float64)
        owpos = ow.positions.astype(np.float64)
        dd = distance_array(dpos, owpos, box=box)
        da_ = distance_array(apos, owpos, box=box)
        bridge = (dd <= 3.5) & (da_ <= 3.5)
        got[ts.frame] = bridge
        # 対応の確認: ペア距離（最小イメージ）vs G05 系列。
        # distance_array は全対全の行列を返すので、ペアごとの距離は
        # 最小イメージの差分から直接計算する。
        delta = apos - dpos
        delta -= box[:3].astype(np.float64) * np.rint(delta / box[:3].astype(np.float64))
        dpair = np.linalg.norm(delta, axis=1)
        row = 0 if ts.frame == 0 else 1
        dist_check[ts.frame] = float(np.abs(dpair - series[row]).max())
    assert set(got) == set(FRAMES_WANT), (prot, rep, sorted(got))
    b0, b1 = got[0], got[10]
    n0 = b0.sum(axis=1)
    n1 = b1.sum(axis=1)
    both = (b0 & b1).sum(axis=1)
    persistence = np.where(n0 > 0, both / np.maximum(n0, 1), 0.0)
    return {
        "n_bridge_f0": n0.astype(np.float32),
        "n_bridge_f1": n1.astype(np.float32),
        "has_bridge_f0": (n0 > 0).astype(np.float32),
        "frac_frames_bridge": (((n0 > 0) + (n1 > 0)) / 2.0).astype(np.float32),
        "persistence": persistence.astype(np.float32),
        "n_bridge_mean": ((n0 + n1) / 2.0).astype(np.float32),
        "dist_check_f0": dist_check[0],
        "dist_check_f1": dist_check[10],
        "n_ow": len(ow), "n_heavy": len(heavy), "n_cand": n_cand,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proteins", nargs="+")
    ap.add_argument("--src", choices=("head", "full"), default="head")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    src_root = HEAD_ROOT if args.src == "head" else FULL_ROOT
    suffix = "" if args.src == "head" else "-total"
    for prot in args.proteins:
        out_file = OUT / f"{prot}__{args.src}.npz"
        if out_file.exists():
            print(prot, "skip (extracted)", flush=True)
            continue
        t0 = time.time()
        store = {}
        for rep in (1, 2, 3):
            r = extract(prot, rep, src_root, suffix)
            for k, v in r.items():
                store[f"r{rep}_{k}"] = v
            print(prot, f"R{rep}: cand {r['n_cand']} OW {r['n_ow']} "
                  f"dist_check f0 {r['dist_check_f0']:.4f} f1 {r['dist_check_f1']:.4f} "
                  f"mean n_bridge_f0 {r['n_bridge_f0'].mean():.4f}", flush=True)
            assert r["dist_check_f0"] < 0.05 and r["dist_check_f1"] < 0.05, \
                (prot, rep, r["dist_check_f0"], r["dist_check_f1"])
        np.savez_compressed(out_file, **store)
        print(prot, f"saved ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
