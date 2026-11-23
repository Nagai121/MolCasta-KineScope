"""K36: 水の環境の履歴特徴量の抽出。

定義は結果を見る前に固定した（README 参照）。
- 完全形式（total）の TPR + XTC を MDAnalysis で直接読む（MiniTopol の変換は通さない）。
  完全形式のフレーム 0, 10, ..., 100 が解析版のフレーム 0〜10（履歴 0〜1 ns）に対応する。
  候補の同定（donor_atom / acceptor_atom）は base_reproduction のキャッシュと同一。
  変換済み軌道の原子並びと完全形式のタンパク質重原子の並びは一致することを確認済み
  （2qt4_A R1: 名前・残基番号が一致、フレーム0 の RMSD 0.0）。
- 水は SOL 残基の酸素 OW。距離はフレームごとの箱で最小イメージ（直交箱、軸ごとの箱長）。
  完全形式の XTC は原子ごとに箱をまたいで巻き込まれているため、座標をそのまま引いてはいけない。

系列（候補ごと、11フレーム）:
  案A（近くの水の数）: nd35_d, nd50_d, nd35_a, nd50_a
      ドナー原子 / アクセプター原子から 3.5 Å / 5.0 Å 以内にある OW の数。
  案B（橋渡し水と最も近い水）: nbridge, dmin_d, dmin_a, dmin_mid
      nbridge: ドナー原子とアクセプター原子の両方から 3.5 Å 以内にある OW の数。
      dmin_d / dmin_a: ドナー / アクセプター原子から最も近い OW までの距離。
      dmin_mid: ドナー・アクセプターの中点（最小イメージで補正した中点）から
      最も近い OW までの距離。
要約は系列ごとに4量（平均、標準偏差、最小値、最後−最初）。
  WaterA = 案Aの4系列 × 4 = 16特徴量、WaterB = 案Bの4系列 × 4 = 16特徴量、
  WaterAB = 32特徴量。

sanity check: 完全形式フレーム100（起点フレーム）で、タンパク質重原子から
3.5 Å 以内にある水の数を数え、K32 の値（2qt4_A R1 で 242個）と突き合わせる。
"""
import time
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from pathlib import Path

TOTAL = Path("/home/hatch/workspace/rod-survival-exp/atlas-ws/storage/sources/ATLAS")
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path("/home/hatch/workspace/kinescope/work/k36_water")
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
FRAMES = list(range(0, 101, 10))  # 完全形式の 0, 10, ..., 100（解析版 0〜10 に対応）


def mic_delta(delta, lengths):
    return delta - lengths * np.rint(delta / lengths)


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
    # 系列: nd35_d, nd50_d, nd35_a, nd50_a, nbridge, dmin_d, dmin_a, dmin_mid
    S = np.zeros((n_cand, 8, n_f), dtype=np.float64)
    n_w35_protein = None
    frame_set = set(FRAMES)
    for ts in u.trajectory:
        if ts.frame > 100:
            break
        if ts.frame not in frame_set:
            continue
        ti = FRAMES.index(ts.frame)
        box = ts.dimensions  # [Lx, Ly, Lz, 90, 90, 90]
        lengths = box[:3].astype(np.float64)
        dpos = heavy.positions[da].astype(np.float64)
        apos = heavy.positions[aa].astype(np.float64)
        owpos = ow.positions.astype(np.float64)
        dd = distance_array(dpos, owpos, box=box)
        da_ = distance_array(apos, owpos, box=box)
        S[:, 0, ti] = (dd <= 3.5).sum(axis=1)
        S[:, 1, ti] = (dd <= 5.0).sum(axis=1)
        S[:, 2, ti] = (da_ <= 3.5).sum(axis=1)
        S[:, 3, ti] = (da_ <= 5.0).sum(axis=1)
        S[:, 4, ti] = ((dd <= 3.5) & (da_ <= 3.5)).sum(axis=1)
        S[:, 5, ti] = dd.min(axis=1)
        S[:, 6, ti] = da_.min(axis=1)
        mid = dpos + mic_delta(apos - dpos, lengths) / 2.0
        dm = distance_array(mid, owpos, box=box)
        S[:, 7, ti] = dm.min(axis=1)
        if ts.frame == 100:
            dprot = distance_array(heavy.positions.astype(np.float64), owpos, box=box)
            n_w35_protein = int((dprot.min(axis=0) <= 3.5).sum())
    def summ(series):
        return np.stack([series.mean(1), series.std(1), series.min(1),
                         series[:, -1] - series[:, 0]], axis=1)
    waterA = np.concatenate([summ(S[:, i, :]) for i in range(4)], axis=1).astype(np.float32)
    waterB = np.concatenate([summ(S[:, i, :]) for i in range(4, 8)], axis=1).astype(np.float32)
    np.savez_compressed(OUT / f"{prot}_R{rep}.npz",
                        waterA=waterA, waterB=waterB, series=S.astype(np.float32),
                        n_water_ow=len(ow), n_w35_protein=n_w35_protein)
    dt = time.time() - t0
    print(f"{prot} R{rep}: cand {n_cand} OW {len(ow)} "
          f"n_w35_protein@f100 {n_w35_protein} "
          f"mean nd35_d {S[:,0,:].mean():.3f} nd35_a {S[:,2,:].mean():.3f} "
          f"nbridge {S[:,4,:].mean():.3f} dmin_d {S[:,5,:].mean():.2f} "
          f"elapsed {dt:.1f}s", flush=True)
    return dt


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for p in PROTEINS:
        for r in (1, 2, 3):
            total += extract(p, r)
    print(f"done, total {total:.1f}s")
