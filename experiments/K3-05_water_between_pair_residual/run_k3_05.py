"""K3-05: ペアの間にある水と、ずれ（平均の残差）の関係を測る。

強い土台（K06(b)、水素結合の形成）で、ずれを K39/K42 と同じ作り方で
出し（実の平均 − 静的な特徴だけからの予測。予測の当てはめは較正用の
8タンパク質の候補だけ）、抽出した水の量との関係を見る。

測ること:
- 各水の量とずれの相関（Pearson と Spearman、テスト全体と
  タンパク質ごとの平均）。
- 静的な予測に水の量を足したときに、ずれのばらつき（標準偏差）が
  どれだけ減るか。当てはめは較正用だけ、評価はテスト側。
- 対照: テスト側の水の量を別のペアのものと入れ替えた（並べ替えた）
  場合に、相関とばらつきの減りが消えるか。

判定（先に固定）:
- 相関の絶対値 0.3 以上を「関係あり」の目安にする（K2-03 と同じ）。
- 静的な予測に足して、ずれのばらつきが 0.30 Å 以下になれば
  「ずれの大部分を説明できた」とする。下限は 0.249 Å。

補足（2026-11-25、統括からの連絡）:
- 「初期構造」の指すフレームを統一する。静的な予測の入力は
  起点フレーム（フレーム10、1 ns の時点）なので、ペアの間にある水は
  まず起点フレームでの有無と数を測る（n_bridge_f100、has_bridge_f100）。
  最初の 0.1 ns での滞在時間（フレーム0と1の量）は、そのあとの
  比較用に残す。最初の版ではフレーム0と1だけを測っていたが、
  その読み方は正しくなかったため、起点フレームの量を足して測り直す。
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
SAVED = Path("/home/hatch/workspace/k06b/run/k06b_out")
WATER = Path("/home/hatch/workspace/kinescope/work/k3_05_water")
ORIG = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
SEED_MODEL = 20261121
QUANTITIES = ("n_bridge_f0", "n_bridge_f1", "n_bridge_mean",
              "has_bridge_f0", "frac_frames_bridge", "persistence")
F100_QUANTITIES = ("n_bridge_f100", "has_bridge_f100")  # 起点フレームの量
ALL_QUANTITIES = QUANTITIES + F100_QUANTITIES
FRAME_OF = {q: "first_0.1ns" for q in QUANTITIES}
FRAME_OF.update({q: "origin_frame" for q in F100_QUANTITIES})


def clean_nan(o):
    if isinstance(o, dict):
        return {k: clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_nan(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def load_base(proteins):
    S, M = [], []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            for r in (1, 2, 3):
                S.append(z[f"r{r}_static"].astype(np.float64))
                M.append(z[f"r{r}_history"][:, 1].astype(np.float64))
    return np.concatenate(S), np.concatenate(M)


def load_water(proteins):
    W = {q: [] for q in ALL_QUANTITIES}
    groups = []  # (protein, replica) per row block, for per-protein stats
    dist_check_max = 0.0
    for p in proteins:
        with np.load(WATER / f"{p}__head.npz") as z:
            for r in (1, 2, 3):
                for q in QUANTITIES:
                    W[q].append(z[f"r{r}_{q}"].astype(np.float64))
                n = len(z[f"r{r}_n_bridge_f0"])
                groups += [(p, r)] * n
                dist_check_max = max(dist_check_max, float(z[f"r{r}_dist_check_f0"]),
                                     float(z[f"r{r}_dist_check_f1"]))
        with np.load(WATER / f"{p}__f100.npz") as z:
            for r in (1, 2, 3):
                for q in F100_QUANTITIES:
                    W[q].append(z[f"r{r}_{q}"].astype(np.float64))
                dist_check_max = max(dist_check_max, float(z[f"r{r}_dist_check_f100"]))
    return {q: np.concatenate(v) for q, v in W.items()}, groups, dist_check_max


def gbm():
    return HistGradientBoostingRegressor(max_iter=160, learning_rate=.065,
                                         max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=.15, random_state=SEED_MODEL)


def corr(x, y):
    return {"pearson": float(np.corrcoef(x, y)[0, 1]),
            "spearman": float(stats.spearmanr(x, y).statistic)}


def main():
    Str, Mtr = load_base(CAL)
    Ste, Mte = load_base(TEST)
    Wtr, _, _ = load_water(CAL)
    Wte, groups, dist_check_max = load_water(TEST)
    assert len(Mte) == 163074, len(Mte)
    assert all(len(Wte[q]) == len(Mte) for q in ALL_QUANTITIES)
    print(f"n_cal {len(Mtr)} n_test {len(Mte)} dist_check_max {dist_check_max:.4f}", flush=True)

    # ずれ（K39/K42 の作り方。当てはめは較正用だけ）
    ols = LinearRegression().fit(Str, Mtr)
    gb = gbm().fit(Str, Mtr)
    resid = {"ols": Mte - ols.predict(Ste), "gbm": Mte - gb.predict(Ste)}
    resid_cal = {"ols": Mtr - ols.predict(Str), "gbm": Mtr - gb.predict(Str)}
    print("residual SD test: ols", round(float(resid['ols'].std()), 4),
          "gbm", round(float(resid['gbm'].std()), 4), flush=True)

    out = {
        "experiment": "K3-05",
        "base": "K06(b) strong base, HB formation; residual = real mean - static-only prediction (K39/K42 construction, fits on calibration candidates only)",
        "n_cal": int(len(Mtr)), "n_test": int(len(Mte)),
        "dist_check_max_abs_diff_vs_G05_series_A": dist_check_max,
        "residual_sd_test": {k: float(v.std()) for k, v in resid.items()},
        "residual_sd_cal": {k: float(v.std()) for k, v in resid_cal.items()},
        "water_quantity_means_test": {q: float(Wte[q].mean()) for q in ALL_QUANTITIES},
        "criteria_fixed_in_advance": {
            "correlation_abs_threshold": 0.3,
            "residual_sd_line_A": 0.30, "residual_sd_lower_bound_A": 0.249},
        "correlations": {}, "sd_reduction": {}, "shuffle_control": {},
        "per_protein_pearson_gbm": {},
    }

    # 相関（テスト全体）
    for q in ALL_QUANTITIES:
        out["correlations"][q] = {k: corr(Wte[q], resid[k]) for k in ("gbm", "ols")}
        print(q, "corr vs resid_gbm:", {k: round(v, 4) for k, v in out["correlations"][q]["gbm"].items()}, flush=True)

    # タンパク質ごとの Pearson（GBM のずれ、タンパク質×3レプリカをまとめて）
    groups = np.asarray([g[0] for g in groups])
    per_protein_rows = []
    for q in ALL_QUANTITIES:
        vals = {}
        for p in TEST:
            m = groups == p
            if m.sum() > 10 and Wte[q][m].std() > 0:
                vals[p] = float(np.corrcoef(Wte[q][m], resid["gbm"][m])[0, 1])
        arr = np.array(list(vals.values()))
        out["per_protein_pearson_gbm"][q] = {
            "n_proteins": int(len(arr)), "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "frac_abs_ge_0.3": float((np.abs(arr) >= 0.3).mean())}
        for p, v in vals.items():
            per_protein_rows.append({"protein": p, "quantity": q, "frame": FRAME_OF[q],
                                    "pearson_vs_resid_gbm": v})

    # 静的な予測に水の量を足したときの、ずれのばらつき
    def sd_with(extra_tr, extra_te, model_fn):
        m = model_fn().fit(np.column_stack([Str, extra_tr]), Mtr)
        return float((Mte - m.predict(np.column_stack([Ste, extra_te]))).std())

    for q in ALL_QUANTITIES:
        out["sd_reduction"][q] = {
            "gbm": sd_with(Wtr[q], Wte[q], gbm),
            "ols": sd_with(Wtr[q], Wte[q], LinearRegression)}
    all_tr = np.column_stack([Wtr[q] for q in QUANTITIES])
    all_te = np.column_stack([Wte[q] for q in QUANTITIES])
    out["sd_reduction"]["all_quantities"] = {
        "gbm": sd_with(all_tr, all_te, gbm), "ols": sd_with(all_tr, all_te, LinearRegression)}
    all8_tr = np.column_stack([Wtr[q] for q in ALL_QUANTITIES])
    all8_te = np.column_stack([Wte[q] for q in ALL_QUANTITIES])
    out["sd_reduction"]["all_quantities_including_f100"] = {
        "gbm": sd_with(all8_tr, all8_te, gbm), "ols": sd_with(all8_tr, all8_te, LinearRegression)}
    print("sd with water (gbm):", {k: round(v["gbm"], 4) for k, v in out["sd_reduction"].items()}, flush=True)

    # 対照: テスト側の水の量を別のペアのものと入れ替える（全体の並べ替え）
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(Mte))
    for q in ALL_QUANTITIES:
        sh = Wte[q][perm]
        out["shuffle_control"][q] = {
            "corr_gbm": corr(sh, resid["gbm"]),
            "sd_gbm_with_shuffled": sd_with(Wtr[q], sh, gbm)}
    sh_all = all_te[perm]
    out["shuffle_control"]["all_quantities"] = {
        "sd_gbm_with_shuffled": sd_with(all_tr, sh_all, gbm)}
    sh_all8 = all8_te[perm]
    out["shuffle_control"]["all_quantities_including_f100"] = {
        "sd_gbm_with_shuffled": sd_with(all8_tr, sh_all8, gbm)}
    print("shuffle control done", flush=True)

    (HERE / "results.json").write_text(json.dumps(clean_nan(out), indent=2, ensure_ascii=False) + "\n")
    import csv
    with open(HERE / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "frame", "pearson_gbm", "spearman_gbm", "pearson_ols", "spearman_ols",
                    "sd_gbm_static_plus_water", "sd_ols_static_plus_water",
                    "shuffle_pearson_gbm", "shuffle_sd_gbm"])
        for q in ALL_QUANTITIES:
            c = out["correlations"][q]
            w.writerow([q, FRAME_OF[q], c["gbm"]["pearson"], c["gbm"]["spearman"], c["ols"]["pearson"], c["ols"]["spearman"],
                        out["sd_reduction"][q]["gbm"], out["sd_reduction"][q]["ols"],
                        out["shuffle_control"][q]["corr_gbm"]["pearson"],
                        out["shuffle_control"][q]["sd_gbm_with_shuffled"]])
        w.writerow(["all_quantities", "first_0.1ns", "", "", "", "",
                    out["sd_reduction"]["all_quantities"]["gbm"], out["sd_reduction"]["all_quantities"]["ols"],
                    "", out["shuffle_control"]["all_quantities"]["sd_gbm_with_shuffled"]])
        w.writerow(["all_quantities_including_f100", "both", "", "", "", "",
                    out["sd_reduction"]["all_quantities_including_f100"]["gbm"], out["sd_reduction"]["all_quantities_including_f100"]["ols"],
                    "", out["shuffle_control"]["all_quantities_including_f100"]["sd_gbm_with_shuffled"]])
    with open(HERE / "per_protein.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["protein", "quantity", "frame", "pearson_vs_resid_gbm"])
        w.writeheader()
        w.writerows(per_protein_rows)
    print("saved results.json / results.csv / per_protein.csv", flush=True)


if __name__ == "__main__":
    main()
