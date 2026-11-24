"""K2-18: Colab に渡す入力データを 1 つの npz にまとめる（仮想マシン側）.

K75 のキャッシュ（静的な特徴 5 個・履歴 10 統計量・ラベル）と、軌道から
取り直した 11 フレームの系列（k2_18_series）を、較正用 8 タンパク質と
テスト 100 タンパク質に分けて連結し、1 つの npz にする。Colab 側の
スクリプトは、この 1 ファイルだけを GitHub から取ってくる。

前提の確認:
- 較正 6,144件・テスト 76,342件（K75 の記録と一致すること）。
- 系列から計算した平均が、キャッシュの history の 1 列目と一致すること。
- 系列から計算した 10 統計量が、キャッシュの history と一致すること。
"""
import json
from pathlib import Path

import numpy as np

CACHE = Path("/home/hatch/workspace/kinescope/work/k75_cache")
SERIES = Path("/home/hatch/workspace/kinescope/work/k2_18_series")
OUT = Path(__file__).resolve().parent / "input_bundle.npz"
CAL = ["2vh3_A", "3ga3_A", "3wx4_A", "3ol3_A", "2es9_A", "2in8_A", "2iay_A", "1lsl_A"]
TEST = [l.strip() for l in Path(
    "/home/hatch/workspace/kinescope/repo/experiments/"
    "K75_readout_transfer_across_horizons_dissociation/proteins108.txt"
).read_text().splitlines() if l.strip()]
TEST = [p for p in TEST if p not in CAL]
assert len(CAL) == 8 and len(TEST) == 100, (len(CAL), len(TEST))


def history10_from_series(series):
    return np.column_stack((series[0], series.mean(axis=0), series.std(axis=0),
                            series.min(axis=0), series.max(axis=0),
                            series[-1] - series[0], series[-1] - series[-2],
                            (series <= 3.5).mean(axis=0),
                            (series <= 4.5).mean(axis=0),
                            (np.diff(series, axis=0) > 0).mean(axis=0)))


def load(prots):
    rows = []
    for p in prots:
        with np.load(CACHE / f"{p}.npz") as z, np.load(SERIES / f"{p}.npz") as s:
            for r in (1, 2, 3):
                static = z[f"r{r}_static"].astype(np.float64)
                hist = z[f"r{r}_history"].astype(np.float64)
                label = z[f"r{r}_label"]
                series11 = s[f"r{r}_series"].astype(np.float64)  # (11, n)
                assert series11.shape == (11, len(label)), (p, r, series11.shape)
                # 系列とキャッシュの突き合わせ
                assert np.max(np.abs(series11.mean(axis=0) - hist[:, 1])) < 1e-4, (p, r)
                h10 = history10_from_series(series11)
                assert np.max(np.abs(h10 - hist)) < 1e-4, (p, r)
                rows.append({"protein": p, "replica": r, "static": static,
                             "history": hist, "series": series11.T, "label": label})
    return rows


def stack(rows):
    return dict(
        static=np.concatenate([r["static"] for r in rows]).astype(np.float32),
        history=np.concatenate([r["history"] for r in rows]).astype(np.float32),
        series=np.concatenate([r["series"] for r in rows]).astype(np.float32),
        label=np.concatenate([r["label"] for r in rows]),
        protein=np.array([r["protein"] for r in rows for _ in range(len(r["label"]))]),
        replica=np.array([r["replica"] for r in rows for _ in range(len(r["label"]))]),
    )


cal = stack(load(CAL))
test = stack(load(TEST))
print("cal n:", len(cal["label"]), "test n:", len(test["label"]), flush=True)
assert len(cal["label"]) == 6144
assert len(test["label"]) == 76342
assert int(test["label"][:, 0].sum()) == 12696  # 1 ns先のテスト陽性（K75）
np.savez_compressed(
    OUT,
    **{f"cal_{k}": v for k, v in cal.items()},
    **{f"test_{k}": v for k, v in test.items()},
)
print("saved", OUT, OUT.stat().st_size / 1e6, "MB", flush=True)
