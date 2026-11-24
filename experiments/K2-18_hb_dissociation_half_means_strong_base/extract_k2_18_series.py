"""K2-18: 強い土台の解離の候補について、11 フレームの距離系列を取り直す.

K75 のキャッシュ（~/workspace/kinescope/work/k75_cache/）には、静的な
特徴 5 個と履歴 10 統計量しか保存されておらず、前半・後半の平均を
作るのに要る 11 フレームの距離系列そのものは残っていない。ここでは
K75 の取り出し（extract_features.py の one_replica）の段取りと同じ
軌道の読み方で、フレーム 0〜10 の距離だけを計算し直し、解離の候補の
系列を保存する。凍結モデルの評価や特徴量の計算はしない（系列だけが
要るため）。

軌道の在りかは、つぎの順番で探す：まず K15 の作業場（k15_atlas）、
つぎに G05 の作業場（g05_atlas_workspace。ATLAS から取り直したものと
同じ軌道が、26 タンパク質のうち 25 タンパク質分すでに手元にある）、
つぎに k2_18_roots（1msc_A だけ。設計の軌道の作業場の trajectories
へのリンクを置いたもの）。どれにもないタンパク質（K15 の変換済み
データが壊れている 4cv4_A と 2hng_A）だけ、ATLAS から取り直す
（tseries の acquire、K75 と同じ）。ATLAS からの取り直しは遅く、
途中で切れることもある（4cv4_A は 1 回、31 MB のうち 5,739 バイト
だけ足りずに失敗した）。取り直した軌道は、取り出しが終わったら
消す（K75 と同じ）。

前提の確認（タンパク質ごと）:
- 系列のフレーム 10（起点）の距離が、K75 のキャッシュの pair[:, 2]
  （起点距離）と 2e-4 Å 以内で一致すること（K75 の取り出しと同じ確認）。
- 系列から計算し直した 10 統計量が、K75 のキャッシュの history と
  1e-4 以内で一致すること（系列とキャッシュの突き合わせ）。
"""
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/hatch/workspace/k06b/pkg_doca")
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import minitopol  # noqa: F401  (zarr を読むのに要る codec)
import numpy as np
from tseries.data import acquire, cell, registry, trajectory
from tseries.features import _mic

K15_ROOT = Path("/home/hatch/workspace/kinescope/work/k15_atlas")
G05_ROOT = Path("/home/hatch/workspace/kinescope/work/g05_atlas_workspace")
LINK_ROOT = Path("/home/hatch/workspace/kinescope/work/k2_18_roots")
OWN_ROOT = Path("/home/hatch/workspace/kinescope/work/k75_atlas")
CACHE = Path("/home/hatch/workspace/kinescope/work/k75_cache")
OUTD = Path("/home/hatch/workspace/kinescope/work/k2_18_series")
OUTD.mkdir(parents=True, exist_ok=True)
PROTEINS = Path("/home/hatch/workspace/kinescope/repo/experiments/"
                "K75_readout_transfer_across_horizons_dissociation/proteins108.txt")


def history10_from_series(series):
    """series は (11, N)。K75 の history と同じ 10 統計量を返す。"""
    return np.column_stack((series[0], series.mean(axis=0), series.std(axis=0),
                            series.min(axis=0), series.max(axis=0),
                            series[-1] - series[0], series[-1] - series[-2],
                            (series <= 3.5).mean(axis=0),
                            (series <= 4.5).mean(axis=0),
                            (np.diff(series, axis=0) > 0).mean(axis=0)))


def one_replica(root, protein, replica, pair):
    reg = registry(root, protein)
    tid = trajectory(protein, replica)
    blocks = list(reg.blocks(tid, start=0, stop=11, chunk_frames=128))
    pos = np.concatenate([b.positions for b in blocks], axis=0)
    boxes = np.concatenate([b.box for b in blocks], axis=0)
    times = np.concatenate([b.time for b in blocks], axis=0)
    assert len(pos) == 11, (tid, len(pos))
    assert np.allclose(times - times[0], np.arange(11) * 100, rtol=0, atol=0.01), \
        (tid, times[:11])
    a, b = pair[:, :2].astype(np.intp).T
    series = np.stack([np.linalg.norm(_mic(pos[k, b] - pos[k, a], cell(boxes[k])),
                                      axis=1) for k in range(11)], axis=0)
    return series


def has_all(root, protein):
    reg = registry(root, protein)
    return all((Path(reg.workspace.trajectory_path(trajectory(protein, r))) / "A.zarr").is_dir()
               for r in (1, 2, 3))


def main():
    proteins = [l.strip() for l in PROTEINS.read_text().splitlines() if l.strip()]
    for p in proteins:
        out = OUTD / f"{p}.npz"
        if out.is_file():
            print("SKIP", p, flush=True)
            continue
        t0 = time.time()
        own = False
        last_err = None
        try:
            roots = []
            for root in (K15_ROOT, G05_ROOT, LINK_ROOT):
                if has_all(root, p):
                    roots.append(root)
            roots.append(OWN_ROOT)
            reps = None
            for root in roots:
                try:
                    if root is OWN_ROOT:
                        acquire(OWN_ROOT, p)
                        own = True
                    reps = []
                    with np.load(CACHE / f"{p}.npz") as z:
                        for r in (1, 2, 3):
                            pair = z[f"r{r}_pair"]
                            hist = z[f"r{r}_history"]
                            series = one_replica(root, p, r, pair)
                            # 前提の確認：起点距離と 10 統計量の突き合わせ
                            assert np.max(np.abs(series[-1] - pair[:, 2])) < 2e-4, \
                                (p, r, np.max(np.abs(series[-1] - pair[:, 2])))
                            h10 = history10_from_series(series)
                            assert np.max(np.abs(h10 - hist.astype(np.float64))) < 1e-4, \
                                (p, r, np.max(np.abs(h10 - hist.astype(np.float64))))
                            reps.append(series.astype(np.float32))
                    break
                except Exception as e:
                    last_err = e
                    reps = None
                    print("SOURCE FAILED", p, root.name, type(e).__name__, str(e)[:120],
                          flush=True)
            if reps is None:
                raise last_err
            np.savez_compressed(out, **{f"r{r}_series": s for r, s in enumerate(reps, 1)})
            print("SCORED", p, "candidates",
                  sum(s.shape[1] for s in reps),
                  "sec", round(time.time() - t0, 1), flush=True)
        except Exception as e:
            print("FAILED", p, type(e).__name__, str(e)[:300], flush=True)
        finally:
            if own:
                shutil.rmtree(OWN_ROOT / p, ignore_errors=True)


if __name__ == "__main__":
    main()
