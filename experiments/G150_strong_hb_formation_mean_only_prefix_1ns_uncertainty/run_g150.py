"""G150: 強い土台で、水素結合の形成の切り取り 1.0 ns の平均だけの
回復率に、区間を付ける (K06(b) の条件)。

G145 では、切り取り 1.0 ns の平均だけの回復率が、1 ns先で 1.028 /
0.994、5 ns先で 0.888 / 0.877、10 ns先で 1.097 / 1.060、30 ns先で
1.115 / 1.132、50 ns先で 0.904 / 0.908、90 ns先で 1.149 / 1.138
（合算 / タンパク質ごとの平均）でした。ここでは 6 つの予測幅すべてに
ついて、その回復率のブートストラップ 95% 区間を出します（G59・G62・
G67 と同じやり方）。テストは 100 タンパク質なので、2 タンパク質の
土台より区間は細かくなります。2 タンパク質のときは取りうる組み合わせが
3通りしかありませんでしたが、100 タンパク質では組み合わせは実質的に
無数で、ブートストラップ分布も 3つの値しか取らないという粗さは
ありません。

ブートストラップの段取り（結果を見る前に固定）:
- 回数: 10,000回。乱数 seed: 42。区間はパーセンタイル法（2.5%点と
  97.5%点）。G23・G49・G59・G62・G67・G116・G132 と同じです。
- 再標本化の単位: タンパク質。テストの 100 タンパク質から復元抽出で
  100 タンパク質を選びます。
- 各ブートストラップ標本の中でも、静的のみ・全部入り・切り取り 1.0 ns
  の平均だけの AUPRC を測り直し、実履歴の利得と回復率をその標本の
  中で計算し直します。モデルは学習し直さず、学習済みのモデルの候補ごと
  の予測確率（`test_predictions.npz` に保存）を再標本化しています。
  測っているのはテスト集合の取り方による揺れで、学習の揺れでは
  ありません。
- 合算の AUPRC の再標本化は、候補ごとの予測確率を並べ替えて
  sklearn の average_precision_score を 1万回計算し直すのでは
  163,074行 × 36条件×予測幅で時間がかかりすぎるため、並べ替えを
  1回だけ行い、重み付きの累積和で同じ値を出すやり方にします。
  予測確率は抽出ごとに変わらないので、スコアの降順の並びは 1回
  決めれば十分で、抽出ごとの違いは各タンパク質の重み（何回選ばれたか）
  だけです。重み付きの累積和で、タイスコアの組の境界での精度と
  再現率の増分から AP を出すと、sklearn の average_precision_score
  と同じ値になります（タイスコアの組をまとめて扱う点も同じ）。
  この速い計算が sklearn と一致することは、最初の 3 抽出について
  sklearn で直接計算し直して確かめます（1e-9 以内で一致すること）。
- タンパク質ごとの平均の AUPRC の再標本化は、タンパク質ごとの AUPRC
  （学習し直した予測確率から出したもの）を重み付き平均することで
  出します。タンパク質ごとの AUPRC が定義できないタンパク質
  （陽性が 0件か全件）は、その抽出の分母から除きます（G141 と同じ
  扱い）。

段取り:
- G145 と同じ段取りでモデルを学習し直す（同じ種・同じ条件で決定的な
  ため、合算とタンパク質ごとの平均の AUPRC は記録と一致するはずで、
  一致を確認する）。候補ごとの予測確率を保存し、そこからブートス
  トラップする。学習し直した予測確率は、ブートストラップの前に
  `test_predictions.npz` に先に保存しておく（実行が途中で止まっても
  学習し直しからやり直さずに済むようにするため）。

読み方（先に固定）:
- 切り取り 1.0 ns の平均だけの回復率の区間が 0.5 を含むかどうかを、
  予測幅ごとに見ます（合算とタンパク質ごとの平均の両方）。

切り取りのフレームの範囲についての注記: フレームは 0.1 ns 刻みで、
フレーム 0 が起点（0 ns）、フレーム 9 が 0.9 ns です。切り取り 1.0 ns
と呼ぶフレーム [0,1,2,3,4,5,6,7,8,9] は、実際には 0〜0.9 ns を覆い
ます（0.25 ns の切り取りのフレーム [0,1,2] が実際には 0.2 ns を覆う
のと同じ形。G145 の README にも同じ
注記があります）。
"""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

K06B = Path("/home/hatch/workspace/k06b/run/k06b_out")
SERIES = Path("/home/hatch/workspace/kinescope/work/g05_series")
OUT = Path(__file__).resolve().parent
SEED = 20261121
HORIZONS = [1, 5, 10, 30, 50, 90]
N_BOOT = 10000  # 結果を見る前に固定（G23・G49・G59・G62・G67 と同じ）
BOOT_SEED = 42  # 結果を見る前に固定（G23・G49・G59・G62・G67 と同じ）
BATCH = 100  # ブートストラップの抽出をまとめて処理する数

orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

# k06b_results.json の記録値 (再現の確認用)
REF = {
    1: dict(static_pooled=0.4591, static_macro=0.4539, full_pooled=0.5293, full_macro=0.5251),
    5: dict(static_pooled=0.4575, static_macro=0.4505, full_pooled=0.5089, full_macro=0.5023),
    10: dict(static_pooled=0.4505, static_macro=0.4438, full_pooled=0.4881, full_macro=0.4826),
    30: dict(static_pooled=0.4492, static_macro=0.4418, full_pooled=0.4855, full_macro=0.4788),
    50: dict(static_pooled=0.4470, static_macro=0.4407, full_pooled=0.4825, full_macro=0.4752),
    90: dict(static_pooled=0.4418, static_macro=0.4357, full_pooled=0.4683, full_macro=0.4630),
}

# G145 の記録値（全条件の AUPRC、合算 / タンパク質ごとの平均）。照合用。
G145_REF = {
    1: {"static_only": (0.4591, 0.4539), "full_history": (0.5293, 0.5251),
        "mean_only_full": (0.5275, 0.5221), "prefix_mean_0.1ns": (0.4974, 0.4933),
        "prefix_mean_0.25ns": (0.4993, 0.4958), "prefix_mean_0.5ns": (0.5079, 0.5027),
        "prefix_mean_1.0ns": (0.5313, 0.5247)},
    5: {"static_only": (0.4575, 0.4505), "full_history": (0.5089, 0.5023),
        "mean_only_full": (0.5028, 0.4957), "prefix_mean_0.1ns": (0.4857, 0.4808),
        "prefix_mean_0.25ns": (0.4883, 0.4826), "prefix_mean_0.5ns": (0.4994, 0.4935),
        "prefix_mean_1.0ns": (0.5031, 0.4959)},
    10: {"static_only": (0.4505, 0.4438), "full_history": (0.4881, 0.4826),
         "mean_only_full": (0.4902, 0.4839), "prefix_mean_0.1ns": (0.4763, 0.4703),
         "prefix_mean_0.25ns": (0.4757, 0.4700), "prefix_mean_0.5ns": (0.4804, 0.4759),
         "prefix_mean_1.0ns": (0.4918, 0.4850)},
    30: {"static_only": (0.4492, 0.4418), "full_history": (0.4855, 0.4788),
         "mean_only_full": (0.4880, 0.4819), "prefix_mean_0.1ns": (0.4719, 0.4658),
         "prefix_mean_0.25ns": (0.4724, 0.4686), "prefix_mean_0.5ns": (0.4829, 0.4772),
         "prefix_mean_1.0ns": (0.4897, 0.4837)},
    50: {"static_only": (0.4470, 0.4407), "full_history": (0.4825, 0.4752),
         "mean_only_full": (0.4791, 0.4723), "prefix_mean_0.1ns": (0.4656, 0.4593),
         "prefix_mean_0.25ns": (0.4700, 0.4644), "prefix_mean_0.5ns": (0.4747, 0.4680),
         "prefix_mean_1.0ns": (0.4790, 0.4720)},
    90: {"static_only": (0.4418, 0.4357), "full_history": (0.4683, 0.4630),
         "mean_only_full": (0.4705, 0.4646), "prefix_mean_0.1ns": (0.4616, 0.4547),
         "prefix_mean_0.25ns": (0.4651, 0.4577), "prefix_mean_0.5ns": (0.4689, 0.4632),
         "prefix_mean_1.0ns": (0.4722, 0.4667)},
}

CUTS = {
    "0.1ns": ("0.1 ns", [0, 1], "0.1 ns"),
    "0.25ns": ("0.25 ns", [0, 1, 2], "0.2 ns"),
    "0.5ns": ("0.5 ns", [0, 1, 2, 3, 4, 5], "0.5 ns"),
    "1.0ns": ("1.0 ns", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "0.9 ns"),
}
ALL_FRAMES = list(range(11))
CONDS = ["static_only", "full_history", "mean_only_full",
         "prefix_mean_0.1ns", "prefix_mean_0.25ns",
         "prefix_mean_0.5ns", "prefix_mean_1.0ns"]


def prefix_mean(series, idx):
    """series: (n_frames, n_candidates). 切り取りの平均だけを返す (n, 1)。"""
    return series[idx].mean(axis=0)[:, None].astype(np.float32)


def load(proteins):
    """Returns dicts of concatenated arrays per replica-split."""
    static, hist, frozen, label, series = [], [], [], [], []
    for p in proteins:
        z = np.load(K06B / f"{p}.npz")
        s = np.load(SERIES / f"{p}.npz")
        for r in (1, 2, 3):
            static.append(z[f"r{r}_static"])
            hist.append(z[f"r{r}_history"])
            frozen.append(z[f"r{r}_frozen"])
            label.append(z[f"r{r}_label"])
            series.append(s[f"r{r}_series"])
    return dict(static=np.concatenate(static), hist=np.concatenate(hist),
                frozen=np.concatenate(frozen), label=np.concatenate(label),
                series=np.concatenate(series, axis=1))


def learner(X, y):
    return HistGradientBoostingClassifier(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=.15,
                                          random_state=SEED).fit(X, y)


def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), positives=int(y.sum()),
                auprc=float(average_precision_score(y, p)),
                auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def fast_ap_sorted(y_sorted, prot_sorted, ends, W):
    """重み付き合算 AUPRC。W: (n_draws, n_proteins) の重み。

    y_sorted / prot_sorted はスコア降順に並べたラベルとタンパク質番号、
    ends はタイスコアの組の末尾の位置（cumsum の切り出し位置）。
    sklearn の average_precision_score と同じ値になる（タイスコアの
    組をまとめて扱う点も同じ）。戻り値は (n_draws,)。
    """
    Ws = W[:, prot_sorted]  # (n_draws, n_rows)
    yw = Ws * y_sorted[None, :]
    cum_pos = np.cumsum(yw, axis=1)[:, ends]
    cum_n = np.cumsum(Ws, axis=1)[:, ends]
    # 重み 0 の組（抽出されなかったタンパク質だけの組）では cum_n が 0 に
    # なる。そこでは陽性の重みも 0 なので精度は積に寄与しない。0 で割って
    # NaN にならないよう、cum_n が 0 の組の精度は 0 とする。
    prec = np.divide(cum_pos, cum_n, out=np.zeros_like(cum_pos), where=cum_n > 0)
    gpos = np.diff(cum_pos, prepend=0.0, axis=1)
    total_pos = cum_pos[:, -1]
    return (gpos * prec).sum(axis=1) / total_pos


def main():
    cal, test = load(CAL), load(TEST)
    print("cal", len(cal["label"]), "test", len(test["label"]), flush=True)

    # 全部入りの平均（11フレームの平均）
    full_mean_cal = cal["series"].mean(axis=0)
    full_mean_test = test["series"].mean(axis=0)

    # per-protein spans for macro AUPRC (eval_b.py と同じ)
    spans, cur = [], 0
    for p in TEST:
        z = np.load(K06B / f"{p}.npz")
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n)); cur += n
    n_prot = len(TEST)
    prot_idx = np.zeros(len(test["label"]), dtype=np.int64)
    for i, (_, s, e) in enumerate(spans):
        prot_idx[s:e] = i

    partial_path = OUT / "results_g150_partial.json"
    done = {}
    if partial_path.exists():
        done = json.loads(partial_path.read_text())
        print("resuming from partial:", sorted(done.keys()), flush=True)

    results, recovery, base_gain = {}, {}, {}
    per_protein = {}
    preds = {}
    for hi, h in enumerate(HORIZONS):
        hkey = str(h)
        y_cal, y_test = cal["label"][:, hi], test["label"][:, hi]
        print(f"--- horizon {h} ns: cal pos {int(y_cal.sum())} test pos {int(y_test.sum())} ---", flush=True)
        if hkey in done:
            results[hkey] = done[hkey]["metrics"]
            recovery[hkey] = done[hkey]["recovery"]
            base_gain[hkey] = done[hkey]["base_gain"]
            per_protein[hkey] = done[hkey]["per_protein_auprc"]
            preds[hkey] = {c: np.array(done[hkey]["predictions"][c], dtype=np.float32)
                           for c in CONDS}
            print("  (loaded from partial)", flush=True)
            continue

        conditions = {}
        conditions["static_only"] = (cal["static"], test["static"])
        conditions["full_history"] = (
            np.column_stack([cal["static"], cal["hist"]]),
            np.column_stack([test["static"], test["hist"]]))
        conditions["mean_only_full"] = (
            np.column_stack([cal["static"], full_mean_cal[:, None].astype(np.float32)]),
            np.column_stack([test["static"], full_mean_test[:, None].astype(np.float32)]))
        for name, (label, idx, covered) in CUTS.items():
            conditions[f"prefix_mean_{name}"] = (
                np.column_stack([cal["static"], prefix_mean(cal["series"], idx)]),
                np.column_stack([test["static"], prefix_mean(test["series"], idx)]))

        m_h, pp_h, pr_h = {}, {}, {}
        for cname, (xtr, xte) in conditions.items():
            pred = learner(xtr, y_cal).predict_proba(xte)[:, 1]
            m = metrics(y_test, pred)
            per = []
            for pname, s, e in spans:
                ys, ps = y_test[s:e], pred[s:e]
                if 0 < ys.sum() < e - s:
                    per.append(float(average_precision_score(ys, ps)))
                else:
                    per.append(None)
            valid = [v for v in per if v is not None]
            m["macro_auprc"] = float(np.mean(valid))
            m["n_proteins_valid"] = len(valid)
            m_h[cname] = m
            pp_h[cname] = {pname: v for (pname, _, _), v in zip(spans, per)}
            pr_h[cname] = pred.astype(np.float32)
            print(f"  {cname}: pooled auprc {m['auprc']:.4f} macro auprc {m['macro_auprc']:.4f}", flush=True)

        # 土台の再現の確認 (k06b_results.json の記録値)
        ref = REF[h]
        assert abs(m_h["static_only"]["auprc"] - ref["static_pooled"]) < 5e-4, (h, m_h["static_only"])
        assert abs(m_h["static_only"]["macro_auprc"] - ref["static_macro"]) < 5e-4, (h, m_h["static_only"])
        assert abs(m_h["full_history"]["auprc"] - ref["full_pooled"]) < 5e-4, (h, m_h["full_history"])
        assert abs(m_h["full_history"]["macro_auprc"] - ref["full_macro"]) < 5e-4, (h, m_h["full_history"])
        # G145 の記録との照合 (全条件、合算とタンパク質ごとの平均)
        for cname, (ref_p, ref_m) in G145_REF[h].items():
            assert abs(m_h[cname]["auprc"] - ref_p) < 5e-4, (h, cname, m_h[cname])
            assert abs(m_h[cname]["macro_auprc"] - ref_m) < 5e-4, (h, cname, m_h[cname])

        gp = m_h["full_history"]["auprc"] - m_h["static_only"]["auprc"]
        gm = m_h["full_history"]["macro_auprc"] - m_h["static_only"]["macro_auprc"]
        base_gain[hkey] = dict(pooled=float(gp), macro=float(gm))
        print(f"  base gain pooled {gp:.4f} macro {gm:.4f}", flush=True)

        rec_h = {}
        for cname, m in m_h.items():
            if cname == "static_only":
                continue
            rec_h[cname] = dict(
                pooled=float((m["auprc"] - m_h["static_only"]["auprc"]) / gp),
                macro=float((m["macro_auprc"] - m_h["static_only"]["macro_auprc"]) / gm))
            vals = []
            for pname, _, _ in spans:
                a_s = pp_h["static_only"][pname]
                a_f = pp_h["full_history"][pname]
                a_c = pp_h[cname][pname]
                if a_s is None or a_f is None or a_c is None:
                    continue
                denom = a_f - a_s
                if abs(denom) < 1e-6:
                    continue
                vals.append((a_c - a_s) / denom)
            rec_h[cname]["per_protein_mean"] = float(np.mean(vals)) if vals else None
            rec_h[cname]["per_protein_median"] = float(np.median(vals)) if vals else None
            rec_h[cname]["per_protein_n"] = len(vals)
            print(f"  {cname}: recovery pooled {rec_h[cname]['pooled']:.3f} "
                  f"macro {rec_h[cname]['macro']:.3f} "
                  f"per-protein mean {rec_h[cname]['per_protein_mean']:.3f} "
                  f"median {rec_h[cname]['per_protein_median']:.3f}", flush=True)

        results[hkey] = m_h
        recovery[hkey] = rec_h
        per_protein[hkey] = pp_h
        preds[hkey] = pr_h
        done[hkey] = dict(metrics=m_h, recovery=rec_h, base_gain=base_gain[hkey],
                          per_protein_auprc=pp_h,
                          predictions={c: pr_h[c].tolist() for c in CONDS})
        partial_path.write_text(json.dumps(done, indent=2, ensure_ascii=False))

    # 学習し直した予測確率を先に保存しておく（ブートストラップの前に）
    np.savez_compressed(
        OUT / "test_predictions.npz",
        test_proteins=np.array(TEST),
        prot_idx=prot_idx,
        y_test=test["label"].astype(np.uint8),
        **{f"pred_{h}_{c}": preds[str(h)][c]
           for h in HORIZONS for c in CONDS})
    print("checkpoint: saved test_predictions.npz", flush=True)

    # ---- ブートストラップ（タンパク質単位の再標本化、10,000回、seed 42）----
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, n_prot, size=(N_BOOT, n_prot))
    W = np.zeros((N_BOOT, n_prot), dtype=np.float64)
    for i in range(n_prot):
        W[:, i] = (draws == i).sum(axis=1)
    del draws

    # 速い合算 AUPRC の検証: 最初の 3 抽出について sklearn で直接計算し直す
    for di in (0, 1, 2):
        w = W[di]
        rows = []
        for pi in range(n_prot):
            if w[pi] > 0:
                s, e = spans[pi][1], spans[pi][2]
                rows.append(np.repeat(np.arange(s, e), int(w[pi])))
        idx = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
        for hkey, cname in (("1", "prefix_mean_1.0ns"), ("90", "static_only")):
            hi = HORIZONS.index(int(hkey))
            y_h = test["label"][:, hi][idx]
            p_h = preds[hkey][cname][idx]
            ref_ap = float(average_precision_score(y_h, p_h))
            order = np.argsort(-preds[hkey][cname], kind="stable")
            sc = preds[hkey][cname][order]
            ends = np.flatnonzero(np.r_[sc[1:] != sc[:-1], True])
            fast = float(fast_ap_sorted(test["label"][:, hi][order].astype(np.float64),
                                        prot_idx[order], ends,
                                        w[None, :])[0])
            assert abs(ref_ap - fast) < 1e-9, (di, hkey, cname, ref_ap, fast)
    print("fast AP check: matches sklearn to 1e-9 on 3 draws x 2 conditions", flush=True)

    # 抽出ごとの回復率をためる（合算とタンパク質ごとの平均）
    boot = {str(h): {c: {"pooled": [], "macro": []} for c in CONDS if c != "static_only"}
            for h in HORIZONS}
    for b0 in range(0, N_BOOT, BATCH):
        Wb = W[b0:b0 + BATCH]
        for hi, h in enumerate(HORIZONS):
            hkey = str(h)
            y_h = test["label"][:, hi].astype(np.float64)
            # タンパク質ごとの AUPRC（NaN は定義できないタンパク質）
            pp = {}
            valid = {}
            for cname in CONDS:
                arr = np.array([per_protein[hkey][cname][p] for p in TEST], dtype=np.float64)
                pp[cname] = np.where(np.isnan(arr), 0.0, arr)
                valid[cname] = (~np.isnan(arr)).astype(np.float64)
            # 合算 AUPRC（速い計算）
            ap_pooled = {}
            for cname in CONDS:
                order = np.argsort(-preds[hkey][cname], kind="stable")
                sc = preds[hkey][cname][order]
                ends = np.flatnonzero(np.r_[sc[1:] != sc[:-1], True])
                ap_pooled[cname] = fast_ap_sorted(
                    y_h[order], prot_idx[order], ends, Wb)
            # タンパク質ごとの平均 AUPRC（重み付き平均）
            ap_macro = {}
            for cname in CONDS:
                num = Wb @ pp[cname]
                den = Wb @ valid[cname]
                ap_macro[cname] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
            gp = ap_pooled["full_history"] - ap_pooled["static_only"]
            gm = ap_macro["full_history"] - ap_macro["static_only"]
            with np.errstate(invalid="ignore", divide="ignore"):
                for cname in boot[hkey]:
                    boot[hkey][cname]["pooled"].append(
                        (ap_pooled[cname] - ap_pooled["static_only"]) / gp)
                    boot[hkey][cname]["macro"].append(
                        (ap_macro[cname] - ap_macro["static_only"]) / gm)
        if (b0 // BATCH + 1) % 20 == 0:
            print(f"  bootstrap {b0 + BATCH}/{N_BOOT} draws", flush=True)

    boot_ci = {}
    for h in HORIZONS:
        hkey = str(h)
        boot_ci[hkey] = {}
        for cname, kinds in boot[hkey].items():
            boot_ci[hkey][cname] = {}
            for kind, vals in kinds.items():
                arr = np.concatenate(vals)
                lo, hi_ci = np.nanpercentile(arr, [2.5, 97.5])
                boot_ci[hkey][cname][kind] = dict(
                    point=float(recovery[hkey][cname][kind]),
                    ci_lower=float(lo), ci_upper=float(hi_ci),
                    contains_0_5=bool(lo <= 0.5 <= hi_ci),
                    contains_0=bool(lo <= 0 <= hi_ci),
                    n_nan=int(np.isnan(arr).sum()))
        print(f"{h}ns bootstrap:", flush=True)
        for cname in boot[hkey]:
            for kind in ("pooled", "macro"):
                b = boot_ci[hkey][cname][kind]
                print(f"  {cname} {kind}: {b['point']:.3f} "
                      f"[{b['ci_lower']:.3f}, {b['ci_upper']:.3f}] "
                      f"contains_0.5={b['contains_0_5']} n_nan={b['n_nan']}", flush=True)

    out = {
        "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        "cuts": {k: {"label": v[0], "frames": v[1], "covered_length": v[2]} for k, v in CUTS.items()},
        "horizons": HORIZONS,
        "references": {
            "base_gain_ref": {str(h): dict(
                pooled=REF[h]["full_pooled"] - REF[h]["static_pooled"],
                macro=REF[h]["full_macro"] - REF[h]["static_macro"]) for h in HORIZONS},
            "base_gain_recomputed": base_gain,
        },
        "metrics": results,
        "recovery_vs_real_history_gain": recovery,
        "per_protein_auprc": per_protein,
        "bootstrap_ci": boot_ci,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    csv_rows = []
    for h in HORIZONS:
        hkey = str(h)
        for cname, m in results[hkey].items():
            rec = recovery[hkey].get(cname)
            b = boot_ci[hkey].get(cname)
            csv_rows.append([
                h, cname, m["n"], m["positives"], round(m["auc"], 4),
                round(m["auprc"], 4), round(m["macro_auprc"], 4),
                round(m["brier"], 4), round(m["logloss"], 4),
                "" if rec is None else round(rec["pooled"], 3),
                "" if rec is None else round(rec["macro"], 3),
                "" if b is None else round(b["pooled"]["ci_lower"], 3),
                "" if b is None else round(b["pooled"]["ci_upper"], 3),
                "" if b is None else round(b["macro"]["ci_lower"], 3),
                "" if b is None else round(b["macro"]["ci_upper"], 3),
            ])
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_ns", "condition", "n", "positives", "auc", "auprc_pooled",
                    "auprc_macro", "brier", "logloss", "recovery_pooled", "recovery_macro",
                    "ci_lower_pooled", "ci_upper_pooled", "ci_lower_macro", "ci_upper_macro"])
        w.writerows(csv_rows)
    print("saved", OUT / "results.json", OUT / "results.csv", OUT / "test_predictions.npz")


if __name__ == "__main__":
    main()
