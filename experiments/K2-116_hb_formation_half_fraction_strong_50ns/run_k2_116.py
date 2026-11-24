"""K2-116: K2-14 の問い（割合の前半と後半）を強い土台の 50 ns先で
測る（水素結合の形成、50 ns先）。

K2-14 は資料Dの条件（テスト 2 タンパク質）で、割合の前半と後半を
測った。90 ns先では前半 0.816・後半 0.676 と、どちらにも当てはまら
なかった。強い土台での割合の前半と後半は、1 ns先（K2-111）は前半
0.619・後半 0.698、5 ns先（K2-113）は前半 0.687・後半 0.437、10 ns先
（K2-114）は前半 0.695・後半 0.545、90 ns先（K2-112）は前半 0.710・
後半 0.575（合算、全部入りの利得に対する割合）で、どれもどちらにも
当てはまらなかった。50 ns先の平均の前半と後半（K2-76）は前半 0.730・
後半 0.581（合算）で、どちらにも当てはまらなかった。ここでは、割合
の形で 50 ns先を測る。
静的な特徴量に、つぎのどれか 1 つを足した形で比べる:
- 前半の割合だけ: フレーム [0..4] で距離が 3.5 Å 以下の割合
- 後半の割合だけ: フレーム [6..10] で距離が 3.5 Å 以下の割合
- 前半と後半の両方: 前半の割合と後半の割合の 2 つ
- 全部の割合だけ: フレーム [0..10] 全部で距離が 3.5 Å 以下の割合
フレーム 5（0.5 ns）は K2-01 と同じく前半にも後半にも入らない
（前半と後半をきれいに分けるためで、「両方」の条件でも使わない）。

読み方（K2-01 と同じく先に固定）:
- 後半の利得が全部入りの 8 割以上で、前半が 5 割以下なら
  「今の状態が効いている」。
- 前半も後半も 8 割以上なら「続いている状態が効いている」。
- どちらでもなければ、数値をそのまま記録する（無理に結論を付けない）。
 - 割合の分母は全部入りの利得（静的＋実履歴 10 特徴の利得）を主とし、
  あわせて全部の割合の利得に対する割合も記録する（分母の確認のため、
  K2-01・K2-05 と同じ）。
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
 - 土台の再現の確認: 50 ns先で静的のみ・実履歴の AUPRC（合算 / タンパ
  ク質ごとの平均）が k06b_results.json の記録（静的 0.4470 / 0.4407、
  全部 0.4825 / 0.4752、利得 +0.0354 / +0.0344）と 5e-4 以内で一致
  すること。全部の割合だけの回復率の既存の記録はないため、照合は
  しない。なお、50 ns先は土台の利得が小さい（合算 +0.0354）ため、
  割合の読みは大きく出やすく、AUPRC の差の絶対値と併せて見るべきです。
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
H = 50  # 50 ns先

orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]

# k06b_results.json の記録値（再現の確認用、50 ns先）
REF = dict(static_pooled=0.4470, static_macro=0.4407,
           full_pooled=0.4825, full_macro=0.4752)
# 全部の割合だけの回復率の既存の記録はない（照合なし）

FIRST = [0, 1, 2, 3, 4]
SECOND = [6, 7, 8, 9, 10]
ALL = list(range(11))


def half_mean(series, idx):
    """series: (n_frames, n_candidates)。指定フレームの平均を返す (n,)。"""
    return series[idx].mean(axis=0).astype(np.float32)


def half_frac(series, idx, bound=3.5):
    """series: (n_frames, n_candidates)。指定フレームで距離が bound 以下
    だった割合を返す (n,)。"""
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

    # 実行前の確認: 11フレーム全部の割合が保存値と一致すること
    # （保存された実履歴10特徴の列 [7] が 3.5 Å 以下の割合。
    #  列の並びは [最初, 平均, 標準偏差, 最小, 最大, 最後−最初, ?, 3.5 Å以下割合,
    #  4.5 Å以下割合, ?] で、列 5 は 3.5 Å 以下の割合ではない。
    #  2026-11-24 に 1a62_A で確かめた：列 7 が 3.5 Å 以下の割合と 2.71e-08 で一致）
    d_cal = float(np.abs(half_frac(cal["series"], ALL) - cal["hist"][:, 7]).max())
    d_test = float(np.abs(half_frac(test["series"], ALL) - test["hist"][:, 7]).max())
    print(f"full frac vs stored history col 7: cal max diff {d_cal:.2e}, test {d_test:.2e}", flush=True)
    assert d_cal < 1e-4 and d_test < 1e-4

    # per-protein spans for macro AUPRC (G141 と同じ)
    spans, cur = [], 0
    for p in TEST:
        z = np.load(K06B / f"{p}.npz")
        n = sum(len(z[f"r{r}_label"]) for r in (1, 2, 3))
        spans.append((p, cur, cur + n)); cur += n

    y_cal, y_test = cal["label"][:, 4], test["label"][:, 4]  # 50 ns先（予測幅は 1/5/10/30/50/90 ns で 0〜5）
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
    partial_path = OUT / "results_k2_116_partial.json"
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
        partial_path.write_text(json.dumps(
            {"metrics_so_far": m_h}, indent=2, ensure_ascii=False))

    # 土台の再現の確認（k06b_results.json の記録値）
    assert abs(m_h["static_only"]["auprc"] - REF["static_pooled"]) < 5e-4, m_h["static_only"]
    assert abs(m_h["static_only"]["macro_auprc"] - REF["static_macro"]) < 5e-4, m_h["static_only"]
    assert abs(m_h["full_history"]["auprc"] - REF["full_pooled"]) < 5e-4, m_h["full_history"]
    assert abs(m_h["full_history"]["macro_auprc"] - REF["full_macro"]) < 5e-4, m_h["full_history"]
    print("  base reproduced (k06b_results.json, 50 ns)", flush=True)

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

    # 全部の割合だけの回復率の既存の記録はないため、照合はしない
    rec_full_frac = frac["full_frac"]["vs_full_history"]["pooled"]
    print(f"  full-frac recovery (no prior value): {rec_full_frac:.3f}", flush=True)

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

    out = {
        "horizon_ns": H,
        "halves": {"first_frames": FIRST, "second_frames": SECOND,
                   "all_frames": ALL, "frame_5_excluded_from_halves": True},
        "full_frac_vs_stored_max_abs_diff": {"cal": d_cal, "test": d_test},
        "base_gain": {"pooled": float(gp), "macro": float(gm)},
        "full_frac_gain": {"pooled": float(gp_frac), "macro": float(gm_frac)},
        "metrics": m_h,
        "fractions": frac,
        "full_frac_recovery_vs_full_history_pooled": rec_full_frac,
        "judgment": judgment,
        "per_protein_shapes": shapes,
        "per_protein_fractions": per_protein_frac,
        "per_protein_auprc": pp_h,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", OUT / "results.json", flush=True)
    # 予測確率は float32 に落として圧縮して保存する（2026-11-24 の
    # ディスククォータ超過のため。float64 のままでは 9.1 MB になり、
    # クォータに収まらない。AUPRC は小数第4位まで一致することを
    # results.json の値と照らして確かめる。保存に失敗しても
    # results.json は残る順番にしておく）
    try:
        np.savez_compressed(OUT / "test_predictions.npz",
                            y_test=y_test.astype(np.uint8),
                            **{f"pred_{k}": v.astype(np.float32)
                               for k, v in preds.items()})
        print("saved", OUT / "test_predictions.npz", flush=True)
    except OSError as e:
        print(f"WARNING: test_predictions.npz not saved ({e}); "
              "predictions are regenerable by re-running this script", flush=True)


if __name__ == "__main__":
    main()
