"""K49: 水の環境の量を変えた履歴特徴量の抽出。

K36 の再確認。定義は結果を見る前に固定した（README 参照）。
- 完全形式（total）の TPR + XTC を MDAnalysis で直接読む（MiniTopol の変換は通さない）。
  完全形式のフレーム 0, 10, ..., 100 が解析版のフレーム 0〜10（履歴 0〜1 ns）に対応する。
  候補の同定（donor_atom / acceptor_atom）は base_reproduction のキャッシュと同一。
- 水は SOL 残基の酸素 OW。距離はフレームごとの箱で最小イメージ（直交箱、軸ごとの箱長）。

条件 (1)「水を数える距離を 5.0 A に変える」:
  K36 の案A・案Bの 3.5 A の量を 5.0 A に置き換えた条件。系列は5つ:
    nd50_d   : ドナー原子から 5.0 A 以内にある OW の数
    nd50_a   : アクセプター原子から 5.0 A 以内にある OW の数
    nbridge50: ドナー原子とアクセプター原子の両方から 5.0 A 以内にある OW の数
    dmin_d   : ドナー原子から最も近い OW までの距離（カットオフによらない）
    dmin_a   : アクセプター原子から最も近い OW までの距離（カットオフによらない）
    dmin_mid : ドナー・アクセプターの中点から最も近い OW までの距離（カットオフによらない）
  要約は系列ごとに4量（平均、標準偏差、最小値、最後−最初）。water5 = 6系列 × 4 = 24特徴量。
  nd50_d / nd50_a / dmin_* は K36 の保存系列（k36_water の series）と同一の計算で、
  この抽出でも同じ値を再計算して一致を確認する。nbridge50 だけが新規の系列。

条件 (2)「滞在」: 水が近くにいたフレームの割合（11フレーム中の割合）。4特徴量:
    frac_d35     : nd35_d >= 1 だったフレームの割合
    frac_a35     : nd35_a >= 1 だったフレームの割合
    frac_bridge35: nbridge（3.5 A の橋渡し水）>= 1 だったフレームの割合
    frac_either35: nd35_d >= 1 または nd35_a >= 1 だったフレームの割合
  これらは K36 の保存系列（nd35_d, nd35_a, nbridge）から計算できるため、
  この抽出では K36 の系列を読み直して作る（再計算はしない）。
"""
import time
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from pathlib import Path

TOTAL = Path("/home/hatch/workspace/rod-survival-exp/atlas-ws/storage/sources/ATLAS")
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
K36 = Path("/home/hatch/workspace/kinescope/work/k36_water")
OUT = Path("/home/hatch/workspace/kinescope/work/k49_water")
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
FRAMES = list(range(0, 101, 10))  # 完全形式の 0, 10, ..., 100（解析版 0〜10 に対応）


def mic_delta(delta, lengths):
    return delta - lengths * np.rint(delta / lengths)


def summ(series):
    return np.stack([series.mean(1), series.std(1), series.min(1),
                     series[:, -1] - series[:, 0]], axis=1)


def extract(prot, rep):
    t0 = time.time()
    base = TOTAL / f"{prot}-total"
    u = mda.Universe(str(base / f"{prot}_prod_R{rep}.tpr"),
                     str(base / f"{prot}_prod_R{rep}.xtc"))
    heavy = u.select_atoms("protein and not name H*")
    ow = u.select_atoms("resname SOL and name OW")
    cz = np.load(CACHE / f"{prot}_R{rep}.npz")
    da, aa = cz["donor_atom"], cz["acceptor_atom"]
    n_cand = len(da)
    n_f = len(FRAMES)
    # 6系列: nd50_d, nd50_a, nbridge50, dmin_d, dmin_a, dmin_mid
    S = np.zeros((n_cand, 6, n_f), dtype=np.float64)
    frame_set = set(FRAMES)
    for ts in u.trajectory:
        if ts.frame > 100:
            break
        if ts.frame not in frame_set:
            continue
        ti = FRAMES.index(ts.frame)
        box = ts.dimensions
        lengths = box[:3].astype(np.float64)
        dpos = heavy.positions[da].astype(np.float64)
        apos = heavy.positions[aa].astype(np.float64)
        owpos = ow.positions.astype(np.float64)
        dd = distance_array(dpos, owpos, box=box)
        da_ = distance_array(apos, owpos, box=box)
        S[:, 0, ti] = (dd <= 5.0).sum(axis=1)
        S[:, 1, ti] = (da_ <= 5.0).sum(axis=1)
        S[:, 2, ti] = ((dd <= 5.0) & (da_ <= 5.0)).sum(axis=1)
        S[:, 3, ti] = dd.min(axis=1)
        S[:, 4, ti] = da_.min(axis=1)
        mid = dpos + mic_delta(apos - dpos, lengths) / 2.0
        dm = distance_array(mid, owpos, box=box)
        S[:, 5, ti] = dm.min(axis=1)
    # K36 の保存系列と再計算値の一致確認（nd50_d=S36[:,1], nd50_a=S36[:,3],
    # dmin_d=S36[:,5], dmin_a=S36[:,6], dmin_mid=S36[:,7]）
    k36 = np.load(K36 / f"{prot}_R{rep}.npz")
    s36 = k36["series"]
    check = {
        "nd50_d": np.abs(S[:, 0, :] - s36[:, 1, :]).max(),
        "nd50_a": np.abs(S[:, 1, :] - s36[:, 3, :]).max(),
        "dmin_d": np.abs(S[:, 3, :] - s36[:, 5, :]).max(),
        "dmin_a": np.abs(S[:, 4, :] - s36[:, 6, :]).max(),
        "dmin_mid": np.abs(S[:, 5, :] - s36[:, 7, :]).max(),
    }
    # 滞在の割合（K36 の保存系列から）
    frac_d35 = (s36[:, 0, :] >= 1).mean(1)
    frac_a35 = (s36[:, 2, :] >= 1).mean(1)
    frac_bridge35 = (s36[:, 4, :] >= 1).mean(1)
    frac_either35 = ((s36[:, 0, :] >= 1) | (s36[:, 2, :] >= 1)).mean(1)
    residence = np.stack([frac_d35, frac_a35, frac_bridge35, frac_either35],
                         axis=1).astype(np.float32)
    water5 = np.concatenate([summ(S[:, i, :]) for i in range(6)], axis=1).astype(np.float32)
    np.savez_compressed(OUT / f"{prot}_R{rep}.npz",
                        water5=water5, residence=residence,
                        series5=S.astype(np.float32), check=check)
    dt = time.time() - t0
    print(f"{prot} R{rep}: cand {n_cand} OW {len(ow)} "
          f"check maxdiff {max(check.values()):.2e} "
          f"mean nd50_d {S[:,0,:].mean():.2f} nbridge50 {S[:,2,:].mean():.3f} "
          f"frac_bridge35 {frac_bridge35.mean():.3f} frac_either35 {frac_either35.mean():.3f} "
          f"elapsed {dt:.1f}s", flush=True)
    return dt


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for p in PROTEINS:
        for r in (1, 2, 3):
            total += extract(p, r)
    print(f"done, total {total:.1f}s")
