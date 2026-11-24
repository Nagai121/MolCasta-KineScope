"""K2-11: K2-07 のタンパク質ごとの形の違いは、何で決まっているか。

K2-07（強い土台、水素結合の形成、1 ns先）では、タンパク質ごとの
前半・後半の割合で形が分かれた。「今の状態」11、「続いている状態」
18、どちらでもない 63、利得が 0.01 以下で判定から外したもの 8。
何がこの分かれ方を決めているのかを、タンパク質ごとの 4 つの量で
比べる。

グループの作り方（K2-07 の読み方と同じ。分母はそのタンパク質の
全部入りの利得）:
- 今の状態: 後半の割合 0.8 以上かつ前半の割合 0.5 以下
- 続いている状態: 前半も後半も 0.8 以上
- どちらでもない: それ以外
- 判定外: 全部入りの利得が 0.01 以下（K2-07 の per_protein_fractions
  に入っていない 8 タンパク質）。比較には使わず、表には載せる。
グループは K2-07 の per_protein_fractions から作り直し、数が
11 / 18 / 63 になることを確かめる。

比べる量（タンパク質ごと。新しい学習は要らない）:
- (a) 履歴内での距離のばらつき: G05 の系列（11 フレーム）の、ペア
  ごとの標準偏差（母標準偏差）をタンパク質の候補で平均したもの。
- (b) 平均の残りの大きさ: K39 の形で作った残りの平均の、タンパク
  質の候補での標準偏差（母標準偏差）。残り = 実の平均 − 静的な
  予測。静的な予測は K39 と同じ GBM（160回、学習率 0.065、葉 15、
  葉あたり最少 20、L2 0.15、seed 20261121）を、K39 と同じ較正
  タンパク質（8 つ、13,808件）の静的特徴 5 個で当てはめたもの。
  新しいモデルではなく、K39 の当てはめの再現である。合算での
  R² と残りの標準偏差が K39 の記録（R² 0.6352、残り 0.6002 Å）と
  合うことを確かめてから使う。
- (c) 全部入りの利得: K2-07 のタンパク質ごとの AUPRC で、
  全部入り − 静的のみ。
- (d) テストの候補数: G05 の系列の候補数を 3 レプリカで合計したもの。

読み方（先に固定）:
- グループごとの中央値を比べ、どれかの量でグループの中央値が
  プールした標準偏差（比べる 92 タンパク質での標準偏差）の 1 倍
  以上離れていれば「その量で分かれる」と記録し、なければ「この
  4 つの量では分かれない」と記録する。プールした標準偏差は
  ddof=1 で測る。
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

K2_07 = Path("/home/hatch/workspace/kinescope/repo/experiments/K2-07_hb_formation_half_means_strong_base")
G05 = Path("/home/hatch/workspace/kinescope/work/g05_series")
SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
OUT = Path(__file__).resolve().parent
GROUPS = ("now", "continuing", "other")
GROUP_LABELS = {"now": "今の状態", "continuing": "続いている状態", "other": "どちらでもない"}


def classify(frac):
    if frac["second"] >= 0.8 and frac["first"] <= 0.5:
        return "now"
    if frac["first"] >= 0.8 and frac["second"] >= 0.8:
        return "continuing"
    return "other"


def main():
    k2_07 = json.loads((K2_07 / "results.json").read_text())
    fracs = k2_07["per_protein_fractions"]
    auprc = k2_07["per_protein_auprc"]
    assert k2_07["per_protein_shapes"] == {
        "now_state": 11, "continuing": 18, "other": 63,
        "excluded_gain_le_0.01": 8}, k2_07["per_protein_shapes"]

    proteins = sorted(auprc["static_only"].keys())
    assert len(proteins) == 100, len(proteins)
    group_of = {}
    for p in proteins:
        if p in fracs:
            group_of[p] = classify(fracs[p])
        else:
            group_of[p] = "excluded"
    counts = {g: sum(1 for p in proteins if group_of[p] == g) for g in GROUPS + ("excluded",)}
    print("groups:", counts, flush=True)
    assert counts == {"now": 11, "continuing": 18, "other": 63, "excluded": 8}, counts

    # (a) 履歴内での距離のばらつきと (d) 候補数（G05 の系列）
    spread, ncand = {}, {}
    for p in proteins:
        z = np.load(G05 / f"{p}.npz")
        stds, ns = [], 0
        for r in (1, 2, 3):
            s = z[f"r{r}_series"].astype(np.float64)
            assert s.shape[0] == 11, (p, r, s.shape)
            stds.append(s.std(axis=0))
            ns += s.shape[1]
        spread[p] = float(np.concatenate(stds).mean())
        ncand[p] = int(ns)
    print("(a) spread: pooled mean %.3f" % np.mean(list(spread.values())), flush=True)

    # (b) 平均の残りの大きさ（K39 の当てはめの再現）
    cal, test = ORIG["calibration_proteins"], ORIG["test_proteins"]
    assert not (set(cal) & set(test)) and len(test) == 100
    Str, Ste, mr_cal, mr_te = [], [], [], []
    test_rows = {}
    for p in cal:
        z = np.load(SAVED / f"{p}.npz")
        for r in (1, 2, 3):
            Str.append(z[f"r{r}_static"].astype(np.float64))
            mr_cal.append(z[f"r{r}_history"].astype(np.float64)[:, 1])
    for p in test:
        z = np.load(SAVED / f"{p}.npz")
        rows_s, rows_m = [], []
        for r in (1, 2, 3):
            rows_s.append(z[f"r{r}_static"].astype(np.float64))
            rows_m.append(z[f"r{r}_history"].astype(np.float64)[:, 1])
        test_rows[p] = (np.concatenate(rows_s), np.concatenate(rows_m))
        Ste.append(test_rows[p][0])
        mr_te.append(test_rows[p][1])
    Str = np.concatenate(Str)
    Ste = np.concatenate(Ste)
    mr_cal = np.concatenate(mr_cal)
    mr_te = np.concatenate(mr_te)
    print(f"(b) K39 reproduction: n_cal {len(mr_cal)} n_test {len(mr_te)}", flush=True)
    model = HistGradientBoostingRegressor(
        max_iter=160, learning_rate=.065, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=.15, random_state=20261121)
    model.fit(Str, mr_cal)
    pred = model.predict(Ste)
    r2 = float(r2_score(mr_te, pred))
    resid_sd = float((mr_te - pred).std())
    print(f"(b) pooled R2 {r2:.4f} (K39: 0.6352) resid sd {resid_sd:.4f} (K39: 0.6002)", flush=True)
    assert abs(r2 - 0.6352) < 0.002 and abs(resid_sd - 0.6002) < 0.002, (r2, resid_sd)
    resid = {}
    off = 0
    for p in test:
        n = len(test_rows[p][1])
        resid[p] = float((test_rows[p][1] - pred[off:off + n]).std())
        off += n
    assert off == len(mr_te)

    # (c) 全部入りの利得（K2-07 のタンパク質ごとの AUPRC）
    gain = {p: float(auprc["full_history"][p] - auprc["static_only"][p]) for p in proteins}

    quantities = {
        "spread": ("(a) 履歴内での距離のばらつき（Å）", spread),
        "resid_size": ("(b) 平均の残りの大きさ（Å）", resid),
        "gain": ("(c) 全部入りの利得（AUPRC）", gain),
        "ncand": ("(d) テストの候補数", {p: float(ncand[p]) for p in proteins}),
    }
    compared = [p for p in proteins if group_of[p] in GROUPS]
    per_protein = {}
    for p in proteins:
        per_protein[p] = dict(group=group_of[p],
                              **{q: quantities[q][1][p] for q in quantities})
    summary = {}
    for q, (label, vals) in quantities.items():
        arr_all = np.array([vals[p] for p in compared])
        pooled_sd = float(arr_all.std(ddof=1))
        by_group = {}
        for g in GROUPS:
            arr = np.array([vals[p] for p in compared if group_of[p] == g])
            by_group[g] = dict(n=int(len(arr)), median=float(np.median(arr)),
                               mean=float(arr.mean()), sd=float(arr.std(ddof=1)),
                               q25=float(np.percentile(arr, 25)),
                               q75=float(np.percentile(arr, 75)))
        pairs = {}
        for i, g1 in enumerate(GROUPS):
            for g2 in GROUPS[i + 1:]:
                diff = abs(by_group[g1]["median"] - by_group[g2]["median"])
                pairs[f"{g1}_vs_{g2}"] = dict(median_diff=float(diff),
                                             ratio=float(diff / pooled_sd))
        summary[q] = dict(label=label, pooled_sd=pooled_sd,
                          by_group=by_group, pairs=pairs)
        print(f"{label}: pooled sd {pooled_sd:.4f}", flush=True)
        for g in GROUPS:
            bg = by_group[g]
            print(f"  {GROUP_LABELS[g]}: n={bg['n']} median {bg['median']:.4f} "
                  f"mean {bg['mean']:.4f} sd {bg['sd']:.4f}", flush=True)
        for pk, pv in pairs.items():
            print(f"  {pk}: median diff {pv['median_diff']:.4f} "
                  f"= {pv['ratio']:.2f} pooled sd", flush=True)

    max_ratio = max(pv["ratio"] for s in summary.values() for pv in s["pairs"].values())
    verdict = ("separates" if max_ratio >= 1.0 else "does_not_separate")
    print("max median-diff / pooled sd:", round(max_ratio, 3), "->", verdict, flush=True)

    out = {
        "groups": counts,
        "group_rule": ("now: second>=0.8 and first<=0.5; continuing: first>=0.8 and "
                       "second>=0.8; other: rest; excluded: full gain<=0.01"),
        "k39_reproduction": dict(r2_test=r2, resid_sd_pooled=resid_sd,
                                 r2_k39=0.6352, resid_sd_k39=0.6002),
        "summary": summary,
        "max_ratio": float(max_ratio),
        "verdict": verdict,
        "per_protein": per_protein,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    with open(OUT / "results.csv", "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["protein", "group", "spread_A", "resid_size_A", "gain_auprc", "n_candidates"])
        for p in proteins:
            row = per_protein[p]
            w.writerow([p, row["group"], round(row["spread"], 4), round(row["resid_size"], 4),
                        round(row["gain"], 4), int(row["ncand"])])
    print("saved", OUT / "results.json", OUT / "results.csv")


if __name__ == "__main__":
    main()
