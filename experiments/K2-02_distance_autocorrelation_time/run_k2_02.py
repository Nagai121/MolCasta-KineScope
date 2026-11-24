"""K2-02: ペアの距離のゆらぎは、どれくらいの時間で忘れられるか。

資料Dの条件（6 タンパク質 × 3 レプリカ）の、水素結合の形成の候補ペアの
距離の時系列（100 ps 刻み 11 点、K08 の保存値 `hist`、元の軌道は
~/workspace/gpt-model-check/data/trajectories/ の ATLAS データ）について、
遅れ 1〜5（100〜500 ps）での距離の自己相関を測る。

自己相関の見積もり方（結果を見る前に固定）:
- 主の見積もり: ペアごとに、そのペアの 11 点の平均と標準偏差（母標準
  偏差）で標準化し、遅れ k の自己相関を全ペア・全時刻の z[t] * z[t+k]
  の平均とする。標準偏差が 0 のペアは除く。
- 副の見積もり: 変動量（バリオグラム）から出す。rho(k) = 1 - MSD(k) /
  (2 * var)。MSD(k) は (d[t+k] - d[t])^2 の全ペア・全時刻の平均、
  var はペアごとの分散（母分散）に 11/10 を掛けたもの（平均の推定に
  よる縮みへの補正）。差分を使うためペアの平均の推定誤差の影響を
  受けにくい。
- まとめる単位: 全体（全ペア）、タンパク質ごと（3 レプリカをまとめる）、
  レプリカごと（6 タンパク質をまとめる）、タンパク質×レプリカごと。
- 閾値を下回る遅れ: 自己相関が 0.5 と 1/e を初めて下回る遅れ（離散）と、
  前後の遅れの間で直線で補間した交差時刻（ns）の両方を記録する。

粗さと偏りの確認（合成データでの較正）:
- 時系列は 11 点しかないため、どちらの見積もりも、真の自己相関が
  大きいときには下向きに偏る。そこで、真の遅れ 1 の自己相関 rho を
  変えた AR(1) 過程（11 点、ペア数は実データと同じ 14,582）を合成し、
  同じ見積もり方で測った値と実データの値を照合して、実データと
  矛盾しない rho の範囲を読む。
"""
import json
from pathlib import Path

import numpy as np

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path(__file__).resolve().parent
PROTEINS = ["1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A"]
REPLICAS = [1, 2, 3]
LAGS = [1, 2, 3, 4, 5]
THRESHOLDS = {"0.5": 0.5, "1/e": float(np.exp(-1.0))}


def ac_naive(hist):
    """hist: (11, N)。ペアごとに標準化した自己相関（主の見積もり）。"""
    d = hist.astype(np.float64)
    mu = d.mean(axis=0)
    sd = d.std(axis=0)
    valid = sd > 0
    z = np.zeros_like(d)
    z[:, valid] = (d[:, valid] - mu[valid]) / sd[valid]
    out = {k: float((z[:-k] * z[k:])[:, valid].mean()) for k in LAGS}
    return out, int(valid.sum()), int((~valid).sum())


def ac_variogram(hist):
    """hist: (11, N)。変動量から出す自己相関（副の見積もり）。"""
    d = hist.astype(np.float64)
    T = d.shape[0]
    var = d.var(axis=0) * T / (T - 1)
    valid = var > 0
    out = {}
    for k in LAGS:
        msd = ((d[k:] - d[:-k]) ** 2).mean(axis=0)
        out[k] = float(1.0 - (msd[valid] / (2.0 * var[valid])).mean())
    return out


def crossings(ac):
    res = {}
    pts = {0: 1.0}
    pts.update(ac)
    for name, th in THRESHOLDS.items():
        first_below = None
        interp = None
        for k in LAGS:
            if ac[k] < th:
                first_below = k
                prev_k = k - 1
                prev_v = pts[prev_k]
                frac = (prev_v - th) / (prev_v - ac[k])
                interp = (prev_k + frac) * 0.1
                break
        res[name] = dict(first_lag_below=first_below,
                         interp_crossing_ns=None if interp is None else float(interp))
    return res


def summarize(series_list):
    H = np.concatenate(series_list, axis=1)
    acn, nv, nx = ac_naive(H)
    acv = ac_variogram(H)
    return dict(n_pairs=int(H.shape[1]), n_valid=nv, n_excluded=nx,
                naive={str(k): acn[k] for k in LAGS},
                variogram={str(k): acv[k] for k in LAGS},
                naive_crossings=crossings(acn),
                variogram_crossings=crossings(acv))


def calibration(n_pairs):
    """AR(1) の合成データで、真の rho(遅れ1) と測定値の対応を作る。"""
    rng = np.random.default_rng(42)
    T = 11
    rows = {}
    for rho in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
        if rho == 0.0:
            x = rng.normal(size=(T, n_pairs))
        else:
            e = rng.normal(size=(T, n_pairs))
            x = np.zeros((T, n_pairs))
            x[0] = e[0] / np.sqrt(1 - rho ** 2)
            for t in range(1, T):
                x[t] = rho * x[t - 1] + e[t]
        acn, _, _ = ac_naive(x)
        acv = ac_variogram(x)
        rows[str(rho)] = dict(naive_lag1=acn[1], variogram_lag1=acv[1],
                              naive={str(k): acn[k] for k in LAGS},
                              variogram={str(k): acv[k] for k in LAGS})
        print(f"calibration AR(1) rho={rho}: naive lag1 {acn[1]:.3f} "
              f"variogram lag1 {acv[1]:.3f}", flush=True)
    return rows


def main():
    by_protein_series = {p: [] for p in PROTEINS}
    by_replica_series = {r: [] for r in REPLICAS}
    per_pr = {}
    all_series = []
    for p in PROTEINS:
        for r in REPLICAS:
            z = np.load(CACHE / f"{p}_R{r}.npz")
            hist = z["hist"]
            assert hist.shape[0] == 11, (p, r, hist.shape)
            per_pr[f"{p}_R{r}"] = summarize([hist])
            by_protein_series[p].append(hist)
            by_replica_series[r].append(hist)
            all_series.append(hist)
            s = per_pr[f"{p}_R{r}"]
            print(f"{p} R{r}: n={s['n_pairs']} naive lag1 {s['naive']['1']:.3f} "
                  f"variogram lag1 {s['variogram']['1']:.3f}", flush=True)

    pooled = summarize(all_series)
    print("pooled naive:", {k: round(pooled['naive'][k], 4) for k in pooled['naive']})
    print("pooled variogram:", {k: round(pooled['variogram'][k], 4) for k in pooled['variogram']})

    result = {
        "definition": ("naive: per-pair standardization (pair mean, population std), "
                       "rho(k) = mean over pairs and t of z[t]*z[t+k]. "
                       "variogram: rho(k) = 1 - MSD(k)/(2*var), var corrected by 11/10. "
                       "pairs with std=0 excluded"),
        "lags_ps": [k * 100 for k in LAGS],
        "pooled": pooled,
        "by_protein": {p: summarize(by_protein_series[p]) for p in PROTEINS},
        "by_replica": {f"R{r}": summarize(by_replica_series[r]) for r in REPLICAS},
        "by_protein_replica": per_pr,
        "calibration_ar1": calibration(pooled["n_pairs"]),
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("saved", OUT / "results.json")


if __name__ == "__main__":
    main()
