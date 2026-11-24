"""K2-111: K2-14 の問い（割合の前半と後半で指導力がどう違うか）を
強い土台で測る（水素結合の形成、1 ns先、割合だけ）。

K2-14 は資料Dの条件（テスト 2 タンパク質）で、90 ns先の割合の前半
と後半を測ったところ、前半 0.816、後半 0.676 と、読み方のどちらにも
当てはまらなかった。強い土台では平均の前半と後半を測り終えたが
（K2-07：合算で前半 0.680・後半 0.805、マクロで前半 0.694・後半
0.789、読み方はどちらにも当てはまらず、タンパク質ごとの形は「今の
状態」11・「続いている状態」18・どちらでもない 63・利得が小さく
判定から外したもの 8）、割合の前半と後半はまだ測っていない。ここ
では割合の前半と後半を強い土台で測る。

条件は強い土台（G81・G86・G141 の土台、K06(b) の条件）で、水素結合
の形成、1 ns先、割合だけ。割合は、ペアが境界（3.5 Å、K71 と同じ）
より近い時間の割合で、与えられたフレームの範囲について、境界以下
だったフレームの数 ÷ その範囲のフレームの数で計算する。静的な
特徴量に、つぎのどれか 1 つを足した形で比べる:
- 前半の割合だけ: フレーム [0..4] の 3.5 Å以下割合
- 後半の割合だけ: フレーム [6..10] の 3.5 Å以下割合
- 前半と後半の両方: 前半の割合と後半の割合の 2 つ
- 全部の割合だけ: フレーム [0..10] 全部の 3.5 Å以下割合
フレーム 5（0.5 ns）は K2-01 と同じく前半にも後半にも入らない
（前半と後半をきれいに分けるためで、「両方」の条件でも使わない）。

読み方（K2-01 と同じく先に固定）:
- 後半の利得が全部入りの 8 割以上で、前半が 5 割以下なら
  「今の状態が効いている」。
- 前半も後半も 8 割以上なら「続いている状態が効いている」。
- どちらでもなければ、数値をそのまま記録する（無理に結論を付けない）。
- 割合の分母は全部入りの利得（静的＋実履歴 10 特徴の利得）を主とし、
  あわせて全部の割合の利得に対する割合も記録する（分母の確認のため、
  K2-07 と同じ）。
- 合算とタンパク質ごとの平均の両方を記録する。タンパク質ごとの形も
  見る（参考として、タンパク質ごとに「今の状態」「続いている状態」
  に当てはまる数を数える。ただしタンパク質ごとの利得が小さいと割合
  は不安定になるので、全部入りの利得が 0.01 を超えるタンパク質だけを
  分類し、除いた数を記録する）。

土台（K06(b) と同じ）:
- 凍結した本体モデル T1I2006 + HistGradientBoosting の補正モデル
  （160回、学習率 0.065、葉 15、葉あたり最少 20、L2 0.15、
  seed 20261121）。較正タンパク質 8つで学習し、未見の100タンパク質
  ×3レプリカ（163,074行）で評価する。
- 系列は G05 で作り直したもの（~/workspace/kinescope/work/g05_series/）
  をそのまま使う。作り直した系列から導いた要約値は保存値と 1e-05 Å
  未満で一致することが G05 で 108タンパク質すべてで確かめてある。
- 土台の再現の確認: 1 ns先で静的のみ・実履歴の AUPRC（合算 / タンパ
  ク質ごとの平均）が k06b_results.json の記録（静的 0.4591 / 0.4539、
  全部 0.5293 / 0.5251、利得 +0.0702 / +0.0712）と 5e-4 以内で一致
  すること。
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

K06B = Path("/home/hatch/workspace/k06b/run/k06b_out")
SERIES = Path("/home/hatch/workspace/kinescope/work/g05_series")
OUT = Path(__file__).resolve().parent
SEED = 20261121
H = 1  # 1 ns先

orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

# k06b_results.json の記録値（再現の確認用、1 ns先）
REF = dict(static_pooled=0.4591, static_macro=0.4539,
           full_pooled=0.5293, full_macro=0.5251)
BOUND = 3.5  # 割合の境界（K71 と同じ）

FIRST = [0, 1, 2, 3, 4]
SECOND = [6, 7, 8, 9, 10]
ALL = list(range(11))


def half_frac(series, idx, bound=BOUND):
    """series: (n_frames, n_candidates)。指定フレームの境界以下割合を返す (n,)。"""
    return (series[idx] <= bound).mean(axis=0).astype(np.float32)


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


def main():
    cal, test = load(CAL), load(TEST)
    print("cal", len(cal["label"]), "test", len(test["label"]), flush=True)

    # 実行前の確認: 11フレーム全部の平均が保存値と一致すること
    # （保存された実履歴10特徴の列 [1] が平均。平均の確認は割合の
    # 条件とは別で、系列の読み込みが正しいことを確かめるため）
    d_cal = float(np.abs(cal["series"][ALL].mean(axis=0) - cal["hist"][:, 1]).max())
    d_test = float(np.abs(test["series"][ALL].mean(axis=0) - test["hist"][:, 1]).max())
    print(f"full mean vs stored history col 1: cal max diff {d_cal:.2e}, test {d_test:.2e}", flush=True)
    assert d_cal < 1e-4 and d_test < 1e-4

    # per-protein spans for macro AUPRC (G141 と同じ)
    spans, cur = [], 0
    for p in TEST:
        z = np.load(K06B / f"{p}.npz")
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n)); cur += n

    y_cal, y_test = cal["label"][:, 0], test["label"][:, 0]
    print(f"--- horizon {H} ns: cal pos {int(y_cal.sum())} test pos {int(y_test.sum())} ---", flush=True)

    f_cal = half_frac(cal["series"], FIRST)
    s_cal = half_frac(cal["series"], SECOND)
    a_cal = half_frac(cal["series"], ALL)
    f_test = half_frac(test["series"], FIRST)
    s_test = half_frac(test["series"], SECOND)
    a_test = half_frac(test["series"], ALL)

    conditions = {
        "static_only": (cal["static"], test["static"]),
        "full_history": (
            np.column_stack([cal["static"], cal["hist"]]),
            np.column_stack([test["static"], test["hist"]])),
        "first_half_frac": (
            np.column_stack([cal["static"], f_cal[:, None]]),
            np.column_stack([test["static"], f_test[:, None]])),
        "second_half_frac": (
            np.column_stack([cal["static"], s_cal[:, None]]),
            np.column_stack([test["static"], s_test[:, None]])),
        "both_halves_frac": (
            np.column_stack([cal["static"], f_cal[:, None], s_cal[:, None]]),
            np.column_stack([test["static"], f_test[:, None], s_test[:, None]])),
        "full_frac": (
            np.column_stack([cal["static"], a_cal[:, None]]),
            np.column_stack([test["static"], a_test[:, None]])),
    }

    m_h, pp_h, preds = {}, {}, {}
    for cname, (xtr, xte) in conditions.items():
        pred = learner(xtr, y_cal).predict_proba(xte)[:, 1]
        preds[cname] = pred
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
        print(f"  {cname}: pooled auprc {m['auprc']:.4f} macro auprc {m['macro_auprc']:.4f}", flush=True)

    # 土台の再現の確認（k06b_results.json の記録値）
    assert abs(m_h["static_only"]["auprc"] - REF["static_pooled"]) < 5e-4, m_h["static_only"]
    assert abs(m_h["static_only"]["macro_auprc"] - REF["static_macro"]) < 5e-4, m_h["static_only"]
    assert abs(m_h["full_history"]["auprc"] - REF["full_pooled"]) < 5e-4, m_h["full_history"]
    assert abs(m_h["full_history"]["macro_auprc"] - REF["full_macro"]) < 5e-4, m_h["full_history"]
    print("  base reproduced (k06b_results.json, 1 ns)", flush=True)

    gp = m_h["full_history"]["auprc"] - m_h["static_only"]["auprc"]
    gm = m_h["full_history"]["macro_auprc"] - m_h["static_only"]["macro_auprc"]
    gp_frac = m_h["full_frac"]["auprc"] - m_h["static_only"]["auprc"]
    gm_frac = m_h["full_frac"]["macro_auprc"] - m_h["static_only"]["macro_auprc"]
    print(f"  base gain pooled {gp:.4f} macro {gm:.4f} "
          f"(full-frac gain pooled {gp_frac:.4f} macro {gm_frac:.4f})", flush=True)

    frac = {}
    for cname, m in m_h.items():
        if cname == "static_only":
            continue
        f = dict(
            vs_full_history=dict(
                pooled=float((m["auprc"] - m_h["static_only"]["auprc"]) / gp),
                macro=float((m["macro_auprc"] - m_h["static_only"]["macro_auprc"]) / gm)),
            vs_full_frac=dict(
                pooled=float((m["auprc"] - m_h["static_only"]["auprc"]) / gp_frac),
                macro=float((m["macro_auprc"] - m_h["static_only"]["macro_auprc"]) / gm_frac)))
        # タンパク質ごとの割合（そのタンパク質の利得を分母にする）
        for denom_name, denom_key in (("vs_full_history", "full_history"),
                                      ("vs_full_frac", "full_frac")):
            vals = []
            for pname, _, _ in spans:
                a_s = pp_h["static_only"][pname]
                a_d = pp_h[denom_key][pname]
                a_c = pp_h[cname][pname]
                if a_s is None or a_d is None or a_c is None:
                    continue
                denom = a_d - a_s
                if abs(denom) < 1e-6:
                    continue
                vals.append((a_c - a_s) / denom)
            f[denom_name]["per_protein_mean"] = float(np.mean(vals)) if vals else None
            f[denom_name]["per_protein_median"] = float(np.median(vals)) if vals else None
            f[denom_name]["per_protein_n"] = len(vals)
        frac[cname] = f
        print(f"  {cname}: frac vs full-history pooled {f['vs_full_history']['pooled']:.3f} "
              f"macro {f['vs_full_history']['macro']:.3f} | "
              f"vs full-frac pooled {f['vs_full_frac']['pooled']:.3f} "
              f"macro {f['vs_full_frac']['macro']:.3f} | "
              f"per-protein median vs full-history "
              f"{f['vs_full_history']['per_protein_median']:.3f}", flush=True)

    # 先に固定した読み方（主の分母 = 全部入りの利得）
    def reading(frac_first, frac_second):
        if frac_second >= 0.8 and frac_first <= 0.5:
            return "今の状態が効いている"
        if frac_first >= 0.8 and frac_second >= 0.8:
            return "続いている状態が効いている"
        return "どちらでもない（数値をそのまま記録）"

    judgment = {
        "pooled": reading(frac["first_half_frac"]["vs_full_history"]["pooled"],
                          frac["second_half_frac"]["vs_full_history"]["pooled"]),
        "macro": reading(frac["first_half_frac"]["vs_full_history"]["macro"],
                         frac["second_half_frac"]["vs_full_history"]["macro"]),
        "pooled_vs_full_frac": reading(
            frac["first_half_frac"]["vs_full_frac"]["pooled"],
            frac["second_half_frac"]["vs_full_frac"]["pooled"]),
        "macro_vs_full_frac": reading(
            frac["first_half_frac"]["vs_full_frac"]["macro"],
            frac["second_half_frac"]["vs_full_frac"]["macro"]),
    }
    print("  judgment:", judgment, flush=True)

    # 参考：タンパク質ごとの形の数え上げ（全部入りの利得が 0.01 を超えるものだけ）
    shapes = {"now_state": 0, "continuing": 0, "other": 0, "excluded_gain_le_0.01": 0}
    per_protein_frac = {}
    for pname, _, _ in spans:
        a_s = pp_h["static_only"][pname]
        a_f = pp_h["full_history"][pname]
        a_1 = pp_h["first_half_frac"][pname]
        a_2 = pp_h["second_half_frac"][pname]
        if None in (a_s, a_f, a_1, a_2) or (a_f - a_s) <= 0.01:
            shapes["excluded_gain_le_0.01"] += 1
            continue
        d = a_f - a_s
        f1, f2 = (a_1 - a_s) / d, (a_2 - a_s) / d
        per_protein_frac[pname] = dict(first=float(f1), second=float(f2))
        if f2 >= 0.8 and f1 <= 0.5:
            shapes["now_state"] += 1
        elif f1 >= 0.8 and f2 >= 0.8:
            shapes["continuing"] += 1
        else:
            shapes["other"] += 1
    print("  per-protein shapes (vs full-history gain, gain > 0.01):", shapes, flush=True)

    np.savez(OUT / "test_predictions.npz",
             y_test=y_test,
             **{f"pred_{k}": v for k, v in preds.items()})

    out = {
        "label": "HB formation, first-half vs second-half history fraction (K2-14 question on the strong base, fraction only)",
        "phenomenon": "HB formation (strong base, K06(b) conditions)",
        "horizon_ns": H,
        "bound_A": BOUND,
        "fraction": "fraction of frames with pair distance <= 3.5 A",
        "halves": {"first_frames": FIRST, "second_frames": SECOND,
                   "all_frames": ALL, "frame_5_excluded_from_halves": True},
        "full_mean_vs_stored_max_abs_diff": {"cal": d_cal, "test": d_test},
        "base_gain": {"pooled": float(gp), "macro": float(gm)},
        "full_frac_gain": {"pooled": float(gp_frac), "macro": float(gm_frac)},
        "metrics": m_h,
        "fractions": frac,
        "judgment": judgment,
        "per_protein_shapes": shapes,
        "per_protein_fractions": per_protein_frac,
        "per_protein_auprc": pp_h,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", OUT / "results.json", OUT / "test_predictions.npz")


if __name__ == "__main__":
    main()
