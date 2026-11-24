"""K2-22: K2-08 の問い（平均の残りが前半と後半で続くか）を塩橋の形成で測る。

K2-08（水素結合の形成）では残りの平均の前半・後半の相関が合算
ピアソン +0.703（GBM）、K2-17（接触の形成）では +0.719 だった。
ここでは同じ問いを塩橋の形成で測る。塩橋の形成では担い手が分散
している（K31）ため、残りの続き方も違うかもしれない。

残りの平均の作り方（K2-08 と同じ形）:
- K15 の広げた条件。塩橋の形成の候補ペア。学習は 58 タンパク質
  （元の 4 ＋追加 54、2,557件）、テストは 30 タンパク質（元の 2 ＋
  追加 28、1,170件）。タンパク質の分け方は K15 の split.json の
  凍結した集合。起点はフレーム 10（1 ns）、履歴は 0〜1 ns の
  11 フレーム。距離は荷電重原子ペアの最小値で、境界は 4.0 Å。
  候補は起点で最小距離が 4.0 Å 超〜10.0 Å 以下で、配列距離が 2
  以上の残基ペア。
- 前半の平均 = 履歴フレーム [0..4] の最小距離の平均、後半の平均 =
  フレーム [6..10] の平均（フレーム 5 は除く。K2-01・K2-08 と同じ
  分け方）。
- 静的な予測は、塩橋の静的な特徴量（X の 5 列: 起点の最小距離、
  Cα間距離、配列距離の log1p、正側が ARG か、負側が GLU か）から
  各半分の平均を予測したもの。線形回帰と HistGradientBoosting
  回帰（K39 と同じ設定: 160回、学習率 0.065、葉 15、葉あたり最少
  20、L2 0.15、seed 20261121）の 2 通りで、どちらも学習側の候補
  だけで当てはめ、テスト側の統計は使わない。残りの平均 = 半分の
  平均 − 静的な予測。
- 予測は半分ごとに当てはめ直す。形成の候補は原点に向けて距離が
  縮む向きに動くため、前半と後半では平均の水準が系統的に違う。
  全部の平均に対する予測をそのまま当てはめると、その水準の違いが
  残りに混ざるためである。
- K2-08 のスクリプトと同じ形で、X は保存されたままの型で
  sklearn に渡す（K15 の保存値は (N, 11) の向きで、転置は要らない）。
  sklearn の学習はスレッド数を 1 に絞る。
- 注意: 2hng_A の R3 と 4cv4_A の R2 はキャッシュで候補が 0 件
  （K15 の記録どおり）。0 件のファイルは読み飛ばす。

測るもの（先に固定）:
- 前半の残りと後半の残りの相関（ピアソンとスピアマン、合算と
  テスト 30 タンパク質ごと）。比較のため、生の平均の前半・後半の
  相関も同じ形で記録する。タンパク質ごとの候補数は平均で約 39 件
  と少ないため、タンパク質ごとの相関はぶれやすい。合算を主とし、
  タンパク質ごとの相関は中央値と範囲でまとめる。

読み方（先に固定）:
- 相関の絶対値が 0.5 以上なら「指導力を運ぶずれは前半から後半へ
  続いている」、0.3 未満なら「続いていない」と記録する。0.3 以上
  0.5 未満の場合は、そのどちらでもない（中間）として数値をその
  まま記録する。読み方の判定は合算の相関で行う。
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
SPLIT = json.loads(Path(
    "/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion/split.json"
).read_text())
OUT = Path(__file__).resolve().parent
TRAIN_PROTEINS = list(SPLIT["orig_train"]) + list(SPLIT["train_additions"])
TEST_PROTEINS = list(SPLIT["orig_test"]) + list(SPLIT["test_additions"])
LINE_HIGH = 0.5
LINE_LOW = 0.3


def load_cache(proteins):
    Xs, hs, prots = [], [], []
    for p in proteins:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt.npz")
            if len(z["X"]) == 0:
                continue
            Xs.append(z["X"])
            hs.append(z["hist"])
            prots.append(np.full(len(z["X"]), p))
    return (np.concatenate(Xs), np.concatenate(hs, axis=0),
            np.concatenate(prots))


def corr_pair(a, b):
    return dict(pearson=float(pearsonr(a, b)[0]),
                spearman=float(spearmanr(a, b)[0]))


def main():
    assert len(TRAIN_PROTEINS) == 58 and len(TEST_PROTEINS) == 30, (
        len(TRAIN_PROTEINS), len(TEST_PROTEINS))
    Xtr, htr, _ = load_cache(TRAIN_PROTEINS)
    Xte, hte, prot_of = load_cache(TEST_PROTEINS)
    half1_tr = htr[:, 0:5].mean(axis=1)
    half2_tr = htr[:, 6:11].mean(axis=1)
    half1_te = hte[:, 0:5].mean(axis=1)
    half2_te = hte[:, 6:11].mean(axis=1)
    full_te = hte.mean(axis=1)
    assert len(half1_tr) == 2557, len(half1_tr)
    assert len(half1_te) == 1170, len(half1_te)
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

    scopes = [("pooled", np.ones(len(half1_te), bool))]
    for p in TEST_PROTEINS:
        scopes.append((p, prot_of == p))
    rows = []
    corr = {}
    for name in ("gbm", "linear"):
        corr[name] = {}
        for label, a, b in (("residual", resid[("half1", name)], resid[("half2", name)]),
                            ("raw_mean", half1_te, half2_te)):
            entry = {}
            for scope, mask in scopes:
                if mask.sum() < 3:
                    continue
                c = corr_pair(a[mask], b[mask])
                entry[scope] = dict(c, n=int(mask.sum()))
                rows.append([name, label, scope, int(mask.sum()),
                             round(c["pearson"], 3), round(c["spearman"], 3)])
            corr[name][label] = entry
            e = entry["pooled"]
            per = np.array([entry[p]["pearson"] for p in TEST_PROTEINS if p in entry])
            print(f"{name} {label}: pooled pearson {e['pearson']:+.3f} "
                  f"spearman {e['spearman']:+.3f} | per-protein pearson "
                  f"median {np.median(per):+.3f} range [{per.min():+.3f}, {per.max():+.3f}]",
                  flush=True)

    np.savez(OUT / "halves.npz",
             resid_half1_gbm=resid[("half1", "gbm")],
             resid_half2_gbm=resid[("half2", "gbm")],
             resid_half1_linear=resid[("half1", "linear")],
             resid_half2_linear=resid[("half2", "linear")],
             half1_mean=half1_te, half2_mean=half2_te, full_mean=full_te,
             protein=prot_of)

    out = {
        "residual_fit": fit_info,
        "n_train": int(len(half1_tr)), "n_test": int(len(half1_te)),
        "n_test_proteins": len(TEST_PROTEINS),
        "half_mean_test": dict(half1=float(half1_te.mean()),
                               half2=float(half2_te.mean()),
                               full=float(full_te.mean())),
        "lines": {"high": LINE_HIGH, "low": LINE_LOW},
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
