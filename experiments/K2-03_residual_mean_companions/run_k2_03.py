"""K2-03: ペアの平均距離のうち静的な特徴量で予測できない部分
（残りの平均）と一緒に動く量は、ほかに何かあるか。

K39・K42 で、実の平均のうち静的な特徴量で予測できない部分（残りの
平均、残差）が指導力を担うことが分かりました。ここでは資料Dの条件で
残りの平均を作り直し、次の候補の量と一緒に動くかを見ます。

候補の量（履歴の 11 フレーム、フレーム 0〜10 で測る）:
- (a) ペアの中点から 8 Å 以内にある重原子の数の平均（周りの詰まり具合）
- (b) その数のばらつき（11 フレームでの標準偏差）
- (c) ペアの 2 残基の Cα の履歴内でのゆらぎの大きさ
  （各残基の Cα の 11 フレームでの RMSF の、2 残基での平均）
- (d) ペアのどちらかの残基と接触する残基の数の平均
  （接触 = 重原子同士の最小距離が 4.5 Å 以下。ペアの 2 残基自身は
  数に入れない）

水の出入り（K36・K49 で閉じた）と残基の環境そのもの（K25 で閉じた）
は候補に入れません。今回は数とゆらぎの量です。

残りの平均の作り方（K39・K42 の形）:
- 実の平均 = 履歴 11 フレームのドナー・アクセプター間距離の平均。
- 静的な予測 = 資料Dの静的な特徴量（X の 7 列）から実の平均を予測。
  予測は線形回帰と HistGradientBoosting 回帰（K39 と同じ設定:
  160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、
  seed 20261121）の 2 通りで、どちらも学習側の候補（4 タンパク質、
  9,610件）だけで当てはめ、テスト側の統計は使いません。
- 残りの平均 = 実の平均 − 静的な予測（テスト 4,972件）。

読み方（先に固定）:
- 残りの平均との相関（ピアソンとスピアマン、合算とテスト 2 タンパ
  ク質ごと）で、絶対値 0.3 以上なら「候補として残す」、未満なら
  「この量ではない」と記録します。

座標の扱い:
- 軌道は /home/hatch/workspace/gpt-model-check/data/trajectories の
  atlas-<タンパク質>-analysis-R<レプリカ>（atoms.parquet と
  A.zarr）から読みます。資料Dのキャッシュを作った extract.py と
  同じ並び（layout）で残基と原子を対応付けます。
- 距離はすべて周期境界の最小イメージで測ります（箱は対角で、軸ごと
  に箱の長さで丸める。extract.py の distance と同じ）。
- 実行前に、読み直した座標から計算したドナー・アクセプター間距離が
  資料Dのキャッシュの hist と一致すること（最大の差が 1e-6 Å 未満）
  を確かめます。
- (c) の Cα のゆらぎは、座標が箱をまたいで飛んでいないことを確かめ
  たうえで（Cα の隣のフレームとの移動が 20 Å 未満）、素の座標の
  RMSF で測ります。参考として、フレームごとの Cα の重心を引いた
  座標での RMSF（全体の平行移動を除いた形）も記録します。
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import minitopol  # noqa: F401  (zarr のコーデック登録のため)

import numpy as np
import pandas as pd
import zarr
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
TRAJ = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
OUT = Path(__file__).resolve().parent
TRAIN_PROTEINS = ["1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A"]
TEST_PROTEINS = ["1mix_A", "2qt4_A"]
FRAMES = list(range(11))
CROWD_RADIUS = 8.0
CONTACT_CUTOFF = 4.5
LINE = 0.3  # 先に固定した線


def layout(df):
    """extract.py と同じ残基・原子の対応付け。"""
    order = list(dict.fromkeys(df["residue_index"].tolist()))
    maps = []
    for r in order:
        idx = np.flatnonzero(df["residue_index"].values == r)
        nm = {df["name"].values[i]: int(i) for i in idx}
        if not {"N", "CA", "C", "O"}.issubset(nm):
            continue
        maps.append(nm)
    return maps


def mic_dist(a, b, lengths):
    """a: (..., 3), b: (..., 3) の最小イメージ距離。箱は対角前提。"""
    v = a - b
    v = v - lengths * np.rint(v / lengths)
    return np.linalg.norm(v, axis=-1)


def mic_vec(v, lengths):
    return v - lengths * np.rint(v / lengths)


def load_cache(proteins):
    Xs, hs, das, aas, drs, ars = [], [], [], [], [], []
    for p in proteins:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            Xs.append(z["X"]); hs.append(z["hist"])
            das.append(z["donor_atom"]); aas.append(z["acceptor_atom"])
            drs.append(z["donor_res"]); ars.append(z["acceptor_res"])
    return (np.concatenate(Xs), np.concatenate(hs, axis=1),
            np.concatenate(das), np.concatenate(aas),
            np.concatenate(drs), np.concatenate(ars))


def trajectory_candidates(prot, rep, da, ac, dres, ares, hist_cached):
    """1 軌道分の候補の量を返す: crowd_mean, crowd_std, ca_rmsf,
    ca_rmsf_centroid, contact_mean（すべて候補ごとの配列）。"""
    d = TRAJ / f"atlas-{prot}-analysis-R{rep}"
    df = pd.read_parquet(d / "atoms.parquet")
    z = zarr.open_group(str(d / "A.zarr"), mode="r")
    pos = z["positions"][:11].astype(np.float64)
    box = z["box"][:11].astype(np.float64)
    maps = layout(df)
    n_res = len(maps)
    ca_idx = np.array([m["CA"] for m in maps])
    heavy = [np.array([v for k, v in m.items()
                       if not k.startswith("H") and not k[0].isdigit()], dtype=int)
             for m in maps]
    heavy_all = np.concatenate(heavy)
    heavy_res = np.concatenate([np.full(len(h), i) for i, h in enumerate(heavy)])
    n = len(da)

    crowd = np.empty((len(FRAMES), n))
    contact = np.empty((len(FRAMES), n))
    ca_pos = pos[:, ca_idx, :]  # (11, n_res, 3)
    for fi, f in enumerate(FRAMES):
        b = box[f]
        assert np.allclose(b, np.diag(np.diag(b)), atol=1e-5), "box not diagonal"
        L = np.diag(b)
        p = pos[f]
        # 前提の確認: キャッシュの hist と一致すること
        d_da = mic_dist(p[da], p[ac], L)
        assert np.abs(d_da - hist_cached[fi]).max() < 1e-6, (prot, rep, f)
        # (a)(b) 中点から 8 Å 以内の重原子の数
        mid = p[da] + 0.5 * mic_vec(p[ac] - p[da], L)
        dm = mic_dist(mid[:, None, :], p[heavy_all][None, :, :], L)
        crowd[fi] = (dm <= CROWD_RADIUS).sum(axis=1)
        # (d) 接触する残基の数
        dh = mic_dist(p[heavy_all][:, None, :], p[heavy_all][None, :, :], L)
        rmin = np.full((n_res, n_res), np.inf)
        np.minimum.at(rmin, (heavy_res[:, None], heavy_res[None, :]), dh)
        touching = rmin <= CONTACT_CUTOFF
        np.fill_diagonal(touching, False)
        ci = touching[dres] | touching[ares]
        ci[np.arange(n), dres] = False
        ci[np.arange(n), ares] = False
        contact[fi] = ci.sum(axis=1)
    # (c) Cα のゆらぎ（RMSF）。箱またぎが無いことの確認つき
    step = np.linalg.norm(np.diff(ca_pos, axis=0), axis=-1)
    assert step.max() < 20.0, (prot, rep, step.max())
    rmsf = np.sqrt(((ca_pos - ca_pos.mean(axis=0)) ** 2).sum(axis=-1).mean(axis=0))
    ca_c = ca_pos - ca_pos.mean(axis=1, keepdims=True)
    rmsf_c = np.sqrt(((ca_c - ca_c.mean(axis=0)) ** 2).sum(axis=-1).mean(axis=0))
    return (crowd.mean(axis=0), crowd.std(axis=0),
            0.5 * (rmsf[dres] + rmsf[ares]),
            0.5 * (rmsf_c[dres] + rmsf_c[ares]),
            contact.mean(axis=0))


def main():
    Xtr, htr, *_ = load_cache(TRAIN_PROTEINS)
    Xte, hte, da, ac, dres, ares = load_cache(TEST_PROTEINS)
    mean_tr = htr.mean(axis=0)
    mean_te = hte.mean(axis=0)
    print(f"train {len(mean_tr)} test {len(mean_te)}", flush=True)

    resid = {}
    fit_info = {}
    models = {
        "linear": LinearRegression(),
        "gbm": HistGradientBoostingRegressor(
            max_iter=160, learning_rate=.065, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=.15, random_state=20261121),
    }
    for name, model in models.items():
        model.fit(Xtr, mean_tr)
        pred_tr = model.predict(Xtr)
        pred_te = model.predict(Xte)
        resid[name] = mean_te - pred_te
        fit_info[name] = dict(
            r2_train=float(1 - ((mean_tr - pred_tr) ** 2).sum() / ((mean_tr - mean_tr.mean()) ** 2).sum()),
            r2_test=float(1 - ((mean_te - pred_te) ** 2).sum() / ((mean_te - mean_te.mean()) ** 2).sum()),
            resid_std_test=float(resid[name].std()),
            resid_std_train=float((mean_tr - pred_tr).std()))
        print(f"{name}: R2 train {fit_info[name]['r2_train']:.3f} "
              f"test {fit_info[name]['r2_test']:.3f} "
              f"resid std test {fit_info[name]['resid_std_test']:.3f} A", flush=True)

    # テスト側の候補の量（軌道ごとに計算してキャッシュの並びに合わせる）
    cand = {k: [] for k in ("crowd_mean", "crowd_std", "ca_rmsf",
                            "ca_rmsf_centroid", "contact_mean")}
    prot_of = []
    off = 0
    spans = {}
    for p in TEST_PROTEINS:
        start = off
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}.npz")
            n_r = len(z["donor_atom"])
            sl = slice(off, off + n_r)
            vals = trajectory_candidates(p, r, da[sl], ac[sl], dres[sl],
                                         ares[sl], hte[:, sl])
            for k, v in zip(cand.keys(), vals):
                cand[k].append(v)
            off += n_r
        spans[p] = (start, off)
        prot_of += [p] * (off - start)
    cand = {k: np.concatenate(v) for k, v in cand.items()}
    assert off == len(mean_te) == 4972, (off, len(mean_te))
    print("protein spans:", spans, flush=True)
    for k, v in cand.items():
        print(f"  {k}: mean {v.mean():.3f} std {v.std():.3f} "
              f"min {v.min():.3f} max {v.max():.3f}", flush=True)

    np.savez(OUT / "candidates.npz",
             resid_gbm=resid["gbm"], resid_linear=resid["linear"],
             real_mean=mean_te, protein=np.array(prot_of),
             **cand)

    CAND_LABELS = {
        "crowd_mean": "(a) 中点から8A以内の重原子の数の平均",
        "crowd_std": "(b) その数のばらつき",
        "ca_rmsf": "(c) 2残基のCaのゆらぎ (RMSF)",
        "ca_rmsf_centroid": "(c参考) Caのゆらぎ (重心を引いた座標)",
        "contact_mean": "(d) 接触する残基の数の平均",
    }
    rows = []
    corr = {}
    for rname in ("gbm", "linear"):
        corr[rname] = {}
        for ck, label in CAND_LABELS.items():
            entry = {}
            for scope, mask in (("pooled", np.ones(len(mean_te), bool)),
                                ("1mix_A", np.array([p == "1mix_A" for p in prot_of])),
                                ("2qt4_A", np.array([p == "2qt4_A" for p in prot_of]))):
                pe = float(pearsonr(resid[rname][mask], cand[ck][mask])[0])
                sp = float(spearmanr(resid[rname][mask], cand[ck][mask])[0])
                entry[scope] = dict(pearson=pe, spearman=sp)
                rows.append([rname, ck, label, scope, int(mask.sum()),
                             round(pe, 3), round(sp, 3),
                             bool(abs(pe) >= LINE or abs(sp) >= LINE)])
            corr[rname][ck] = entry
            e = entry["pooled"]
            print(f"{rname} {ck}: pooled pearson {e['pearson']:+.3f} "
                  f"spearman {e['spearman']:+.3f} | "
                  f"1mix {entry['1mix_A']['pearson']:+.3f}/{entry['1mix_A']['spearman']:+.3f} | "
                  f"2qt4 {entry['2qt4_A']['pearson']:+.3f}/{entry['2qt4_A']['spearman']:+.3f}",
                  flush=True)

    out = {
        "residual_fit": fit_info,
        "n_train": int(len(mean_tr)), "n_test": int(len(mean_te)),
        "protein_spans": {k: list(v) for k, v in spans.items()},
        "candidate_summary": {k: dict(mean=float(v.mean()), std=float(v.std()),
                                      min=float(v.min()), max=float(v.max()))
                              for k, v in cand.items()},
        "line": LINE,
        "correlations": corr,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["residual", "candidate", "candidate_label", "scope", "n",
                    "pearson", "spearman", "abs_ge_0.3"])
        w.writerows(rows)
    print("saved", OUT / "results.json", OUT / "results.csv", OUT / "candidates.npz")


if __name__ == "__main__":
    main()
