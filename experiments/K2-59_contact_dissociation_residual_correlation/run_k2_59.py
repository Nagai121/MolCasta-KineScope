"""K2-27: K2-08 の問い（平均の残りが前半と後半で続くか）を水素結合の解離で測る。

K2-08（水素結合の形成）では残りの平均の前半・後半の相関が合算
ピアソン +0.703（GBM の残り）・+0.729（線形の残り）、K2-17
（接触の形成）では +0.719・+0.726、K2-22（塩橋の形成）では
+0.641・+0.659、K2-27（水素結合の解離）では +0.693・+0.733
だった。形成の 3 現象と水素結合の解離ではどれも先に固定した
0.5 の線を超えた。ここでは同じ問いを接触の解離で測る。接触の
解離でも同じ形かどうかを見る。

残りの平均の作り方（K2-08 と同じ形）:
- 資料Dの条件。接触の解離の候補ペア。学習は 4 タンパク質
  （1bgf_A・1ifg_A・1j8r_A・2b0a_A、6,383件）、テストは 1mix_A と
  2qt4_A（2,915件）。起点はフレーム 10（1 ns）、履歴は 0〜1 ns の
  11 フレーム。距離は残基ペアの最小の重原子間距離で、境界は 4.5 Å。
  候補は起点で接触している（最小距離 4.5 Å以下）残基ペア。
- 前半の平均 = 履歴フレーム [0..4] の最小の重原子間距離の
  平均、後半の平均 = フレーム [6..10] の平均（フレーム 5 は除く。
  K2-01・K2-08 と同じ分け方）。
- 静的な予測は、資料Dの静的な特徴量（X の 5 列）から各半分の平均
  を予測したもの。線形回帰と HistGradientBoosting 回帰（K39 と
  同じ設定: 160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、
  seed 20261121）の 2 通りで、どちらも学習側の候補だけで当てはめ、
  テスト側の統計は使わない。残りの平均 = 半分の平均 − 静的な予測。
- 予測は半分ごとに当てはめ直す。解離の候補は原点に向けて距離が
  離れる向きに動くため、前半と後半では平均の水準が系統的に違う。
  全部の平均に対する予測をそのまま当てはめると、その水準の違いが
  残りに混ざるためである。
- K2-08 のスクリプトと同じ形で、X は保存されたままの型で
  sklearn に渡す。K50 の保存値は hist が (N, 11) の向きなので、
  転置しない（K12 と同じ向き）。sklearn の学習はスレッド数
  を 1 に絞る。

測るもの（先に固定）:
- 前半の残りと後半の残りの相関（ピアソンとスピアマン、合算と
  テスト 2 タンパク質ごと）。比較のため、生の平均の前半・後半の
  相関も同じ形で記録する。

読み方（先に固定、K2-08 と同じ）:
- 合算ピアソンが 0.5 を超えれば「残りは前半から後半へと続いて
  いる」、未満なら「続いていない」と記録する。GBM の残りと線形の
  残りの両方に同じ線を当てる。
"""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

CACHE = Path("/home/hatch/workspace/kinescope/work/k50_cache")
OUT = Path(__file__).resolve().parent
TRAIN_PROTEINS = ["1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A"]
TEST_PROTEINS = ["1mix_A", "2qt4_A"]
LINE = 0.5  # 先に固定した読み方の線（K2-08 と同じ）


def load_cache(proteins):
    Xs, hs, prots = [], [], []
    for p in proteins:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_contact_diss.npz")
            Xs.append(z["X"])
            hs.append(z["hist"])
            prots.append(np.full(len(z["X"]), p))
    return (np.concatenate(Xs), np.concatenate(hs, axis=0),
            np.concatenate(prots))


def corr_pair(a, b):
    return dict(pearson=float(pearsonr(a, b)[0]),
                spearman=float(spearmanr(a, b)[0]))


def main():
    Xtr, htr, _ = load_cache(TRAIN_PROTEINS)
    Xte, hte, prot_of = load_cache(TEST_PROTEINS)
    half1_tr = htr[:, 0:5].mean(axis=1)
    half2_tr = htr[:, 6:11].mean(axis=1)
    half1_te = hte[:, 0:5].mean(axis=1)
    half2_te = hte[:, 6:11].mean(axis=1)
    full_te = hte.mean(axis=1)
    assert len(half1_tr) == 6383, len(half1_tr)
    assert len(half1_te) == 2915, len(half1_te)
    print(f"train {len(half1_tr)} test {len(half1_te)}", flush=True)
    print(f"half means (test): half1 {half1_te.mean():.3f} half2 {half2_te.mean():.3f} "
          f"full {full_te.mean():.3f}", flush=True)

    models = {
        "linear": LinearRegression(),
        "gbm": HistGradientBoostingRegressor(
            max_iter=160, learning_rate=.065, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=.15, random_state=20261121),
    }
    resid = {}
    fit_info = {}
    for half, (ytr, yte) in {"half1": (half1_tr, half1_te),
                             "half2": (half2_tr, half2_te)}.items():
        for name, model in models.items():
            model.fit(Xtr, ytr)
            pred_tr = model.predict(Xtr)
            pred_te = model.predict(Xte)
            resid[(half, name)] = yte - pred_te
            fit_info[f"{half}_{name}"] = dict(
                r2_train=float(1 - ((ytr - pred_tr) ** 2).sum() / ((ytr - ytr.mean()) ** 2).sum()),
                r2_test=float(1 - ((yte - pred_te) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()),
                resid_std_test=float(resid[(half, name)].std()),
                resid_std_train=float((ytr - pred_tr).std()))
            print(f"{half} {name}: R2 train {fit_info[f'{half}_{name}']['r2_train']:.3f} "
                  f"test {fit_info[f'{half}_{name}']['r2_test']:.3f} "
                  f"resid std test {fit_info[f'{half}_{name}']['resid_std_test']:.3f} A", flush=True)

    scopes = (("pooled", np.ones(len(half1_te), bool)),
              ("1mix_A", prot_of == "1mix_A"),
              ("2qt4_A", prot_of == "2qt4_A"))
    rows = []
    corr = {}
    for name in ("gbm", "linear"):
        corr[name] = {}
        for label, a, b in (("residual", resid[("half1", name)], resid[("half2", name)]),
                            ("raw_mean", half1_te, half2_te)):
            entry = {}
            for scope, mask in scopes:
                c = corr_pair(a[mask], b[mask])
                entry[scope] = dict(c, n=int(mask.sum()))
                rows.append([name, label, scope, int(mask.sum()),
                             round(c["pearson"], 3), round(c["spearman"], 3)])
            corr[name][label] = entry
            e = entry["pooled"]
            print(f"{name} {label}: pooled pearson {e['pearson']:+.3f} "
                  f"spearman {e['spearman']:+.3f} | "
                  f"1mix {entry['1mix_A']['pearson']:+.3f}/{entry['1mix_A']['spearman']:+.3f} | "
                  f"2qt4 {entry['2qt4_A']['pearson']:+.3f}/{entry['2qt4_A']['spearman']:+.3f}",
                  flush=True)

    verdict = {}
    for name in ("gbm", "linear"):
        r = corr[name]["residual"]["pooled"]["pearson"]
        verdict[name] = ("残りは前半から後半へと続いている" if r > 0.5
                         else "続いていない")
        print(f"verdict ({name}): pooled pearson {r:+.3f} -> {verdict[name]}",
              flush=True)

    np.savez(OUT / "halves.npz",
             resid_half1_gbm=resid[("half1", "gbm")],
             resid_half2_gbm=resid[("half2", "gbm")],
             resid_half1_linear=resid[("half1", "linear")],
             resid_half2_linear=resid[("half2", "linear")],
             half1_mean=half1_te, half2_mean=half2_te, full_mean=full_te,
             protein=prot_of)

    out = {
        "experiment": "K2-59",
        "horizon_ns": 1,
        "reading": verdict,
        "reading_rule_fixed_in_advance": "pooled Pearson of the residual "
            "(front-half vs back-half) > 0.5 -> 'residual persists from "
            "front to back'; < 0.5 -> 'does not persist' (K2-08 rule)",
        "residual_fit": fit_info,
        "n_train": int(len(half1_tr)), "n_test": int(len(half1_te)),
        "half_mean_test": dict(half1=float(half1_te.mean()),
                               half2=float(half2_te.mean()),
                               full=float(full_te.mean())),
        "line": LINE,
        "correlations": corr,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["residual", "quantity", "scope", "n", "pearson", "spearman"])
        w.writerows(rows)
    print("saved", OUT / "results.json", OUT / "results.csv", OUT / "halves.npz")


if __name__ == "__main__":
    main()
