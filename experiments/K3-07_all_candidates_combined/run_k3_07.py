"""K3-07: K3-01〜K3-06 で測った候補の量を全部合わせると、ずれをどこまで説明できるか。

土台は K3 系列共通（強い土台 K06(b)、水素結合の形成）。ずれは K39・K42 の作り方
（静的特徴5個から実の平均を予測し、実の平均との差を取る。当てはめは較正8タンパク質
の候補 13,808件だけ、評価は未見100タンパク質×3レプリカの候補 163,074件）。
予測器は HistGradientBoosting 回帰（160回、学習率 0.065、葉 15、葉あたり最小 20、
L2 0.15、種 20261121）で、K39・K3-01〜K3-06 と同じ設定。

使う量（すべて起点フレーム＝フレーム10のもの。各実験の保存値を使う。再計算はしない）:
- K3-01: 側鎖の χ 角（供与体側・受容体側の χ1〜χ4 を sin・cos にした16量）
  出所: ~/workspace/kinescope/work/k3_01/chi/{protein}.npz（K3-01 の抽出値）
- K3-02: 周りの水素結合の数 11 種（フレーム10で数え直したもの）
  出所: ~/workspace/k3_02/counts_f10/{protein}.npz
- K3-03: 電場 2 量（ペアを結ぶ方向の成分、中点での大きさ）
  出所: ~/workspace/kinescope/work/k3_data/{protein}.npz（K3-03・K3-04 共通の抽出）
- K3-04: 背骨の φ/ψ（供与体・受容体の φ・ψ を cos・sin にした8量）
  出所: ~/workspace/kinescope/work/k3_data/{protein}.npz
- K3-05: ペアの間の水 8 量（フレーム10の量を含む、測った量すべて）
  出所: ~/workspace/kinescope/work/k3_05_water/{protein}__head.npz と __f100.npz
- K3-06: 相互作用のエネルギー 10 量（フレーム10で測り直したもの、記録した10個）
  出所: ~/workspace/kinescope/k3work/energy_out/{protein}.npz
追加枠: K3-02 のフレーム0の 11 種（~/workspace/k3_02/counts/）。

測り方:
 1. 土台だけ（静的特徴5個）の再現。
 2. 全量を足したモデル（土台＋6グループ全部）。フレーム0の追加枠も足した形も別に測る。
 3. グループごとの追加（6通り）。
 4. 1グループずつ抜いた場合（6通り）。
 5. 入れ替え対照: 全量をタンパク質×レプリカの中でまとめて入れ替える。
    (a) 較正側を入れ替えて当てはめ直す形、(b) 評価側（テスト）だけ入れ替える形の両方。

前提の確認:
 - 土台だけのずれのばらつきが K39・K3-01 の記録（GBM 0.6002 Å、線形 0.6076 Å）と合うこと。
 - 組み立てたテスト側のずれと各量が、K3-01〜K3-06 の保存値（test_pairs.npz、
   analysis_data.npz、candidates.npz）と一致すること（行の対応の確認）。
 - グループごとに足した結果が、各実験の元の記録（その実験の量を全部足したときの
   ばらつき）と合うこと。合わなければ前提の問題として止める。
"""
import numpy as np, json, csv, sys
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

ROOT = Path("/home/hatch/workspace")
SAVED = ROOT / "k06b/run/k06b_out"
ORIG = json.loads((ROOT / "k06b/results_100.json").read_text())
CAL, TEST = ORIG["calibration_proteins"], ORIG["test_proteins"]
EXP = ROOT / "kinescope/repo/experiments"
GBM = dict(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
           min_samples_leaf=20, l2_regularization=.15, random_state=20261121)
SHUFFLE_SEED = 20261126

QCOLS_HB = ["n_out_donor_res", "n_in_donor_res", "n_out_acceptor_res", "n_in_acceptor_res",
            "n_hb_total", "n_partner_backbone", "n_partner_sidechain", "n_partner_water",
            "n_competing", "n_hb_donor_res_total", "n_hb_acceptor_res_total"]
WATER_HEAD = ["n_bridge_f0", "n_bridge_f1", "n_bridge_mean", "has_bridge_f0",
              "frac_frames_bridge", "persistence"]
WATER_F100 = ["n_bridge_f100", "has_bridge_f100"]

# 元の記録（各実験の results.json から。グループごとに全部足したときの GBM のばらつき）
REF_GROUP_SD = {
    "k3_01_chi": 0.5973363445561887,        # K3-01 results.json variance_reduction static+chi_f10_DA
    "k3_02_hb_frame10": 0.5959450554987167,  # K3-02 results_f10.json added_value_gbm ALL
    "k3_03_field": 0.5985696800962539,       # K3-03 results.json added_to_static_prediction gbm
    "k3_04_backbone": 0.6025355452249721,    # K3-04 results.json added_to_static_prediction gbm
    "k3_05_water": 0.5787537505836625,       # K3-05 results.json sd_reduction all_quantities_including_f100 gbm
    "k3_06_energy_frame10": 0.596518838872219,  # K3-06 results.json frame10 augmented_fits all10
}
REF_HB0_SD = 0.5828691322652277  # K3-02 results.json (frame0) added_value_gbm ALL
REF_BASE_GBM = 0.6001888358758622
REF_BASE_OLS = 0.6075883590992472


def load_base(proteins):
    S, M, G = [], [], []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            for r in (1, 2, 3):
                S.append(z[f"r{r}_static"].astype(np.float64))
                M.append(z[f"r{r}_history"][:, 1].astype(np.float64))
                G += [(p, r)] * len(S[-1])
    return np.vstack(S), np.concatenate(M), G


def per_protein_blocks(proteins, fn):
    """fn(p, r, npair) -> (n, k) array; rows checked against k06b pair counts."""
    blocks = []
    for p in proteins:
        with np.load(SAVED / f"{p}.npz") as z:
            for r in (1, 2, 3):
                n = len(z[f"r{r}_pair"])
                b = fn(p, r, n)
                assert b.shape[0] == n, (p, r, b.shape, n)
                blocks.append(b)
    return np.vstack(blocks)


def chi_block(p, r, n):
    with np.load(ROOT / f"kinescope/work/k3_01/chi/{p}.npz") as z:
        chi = z[f"r{r}_chi"][:, :, :, 1].astype(np.float64)  # frame index 1 = frame 10
    cols = []
    for s in (0, 1):  # slots D, A
        for k in range(4):
            ang = chi[:, s, k]
            cols.append(np.sin(np.radians(ang)))
            cols.append(np.cos(np.radians(ang)))
    return np.column_stack(cols)


def hb_block_factory(dirname):
    def f(p, r, n):
        with np.load(ROOT / f"k3_02/{dirname}/{p}.npz") as z:
            cols = list(z["columns"])
            qi = [cols.index(q) for q in QCOLS_HB]
            return z[f"r{r}_counts"][:, qi].astype(np.float64)
    return f


def field_block(p, r, n):
    with np.load(ROOT / f"kinescope/work/k3_data/{p}.npz") as z:
        return np.column_stack([z[f"r{r}_e_par"], z[f"r{r}_e_mag"]]).astype(np.float64)


def backbone_block(p, r, n):
    with np.load(ROOT / f"kinescope/work/k3_data/{p}.npz") as z:
        cols = []
        for key in ("phi_d", "psi_d", "phi_a", "psi_a"):
            a = z[f"r{r}_{key}"].astype(np.float64)  # radians
            cols.append(np.cos(a))
            cols.append(np.sin(a))
        return np.column_stack(cols)


def water_block(p, r, n):
    with np.load(ROOT / f"kinescope/work/k3_05_water/{p}__head.npz") as z:
        head = [z[f"r{r}_{k}"].astype(np.float64) for k in WATER_HEAD]
    with np.load(ROOT / f"kinescope/work/k3_05_water/{p}__f100.npz") as z:
        f100 = [z[f"r{r}_{k}"].astype(np.float64) for k in WATER_F100]
    return np.column_stack(head + f100)


def energy_block(p, r, n):
    with np.load(ROOT / f"kinescope/k3work/energy_out/{p}.npz") as z:
        return z[f"r{r}_quant10"].astype(np.float64)


GROUP_DEFS = [
    ("k3_01_chi", "K3-01 側鎖のχ角（供与体・受容体、sin/cos）", chi_block),
    ("k3_02_hb_frame10", "K3-02 周りの水素結合の数11種（フレーム10）", hb_block_factory("counts_f10")),
    ("k3_03_field", "K3-03 電場2量", field_block),
    ("k3_04_backbone", "K3-04 背骨のφ/ψ（cos/sin 8量）", backbone_block),
    ("k3_05_water", "K3-05 ペアの間の水8量", water_block),
    ("k3_06_energy_frame10", "K3-06 相互作用のエネルギー10量（フレーム10）", energy_block),
]

print("loading base ...", flush=True)
Scal, mr_cal, groups_cal = load_base(CAL)
Ste, mr_te, groups_te = load_base(TEST)
print("n_cal", len(mr_cal), "n_test", len(mr_te), flush=True)

groups = {}
for name, label, fn in GROUP_DEFS:
    groups[name] = (per_protein_blocks(CAL, fn), per_protein_blocks(TEST, fn))
    print("group", name, groups[name][0].shape, groups[name][1].shape, flush=True)
hb0_cal = per_protein_blocks(CAL, hb_block_factory("counts"))
hb0_te = per_protein_blocks(TEST, hb_block_factory("counts"))
print("extra frame (K3-02 frame0)", hb0_cal.shape, hb0_te.shape, flush=True)

# ---------- 前提1: 土台だけの再現 ----------
premise = {}
for method in ("ols", "gbm"):
    model = (LinearRegression() if method == "ols"
             else HistGradientBoostingRegressor(**GBM)).fit(Scal, mr_cal)
    resid = mr_te - model.predict(Ste)
    premise[method] = float(resid.std())
print("premise base: gbm", premise["gbm"], "ref", REF_BASE_GBM,
      "| ols", premise["ols"], "ref", REF_BASE_OLS, flush=True)
if abs(premise["gbm"] - REF_BASE_GBM) > 5e-4 or abs(premise["ols"] - REF_BASE_OLS) > 5e-4:
    print("PREMISE FAILED: base reproduction mismatch", flush=True)
    sys.exit(2)
resid_base = mr_te - HistGradientBoostingRegressor(**GBM).fit(Scal, mr_cal).predict(Ste)

# ---------- 前提2: 行の対応（保存値との一致） ----------
align = {}
tp = np.load(EXP / "K3-01_sidechain_chi_residual/test_pairs.npz")
align["resid_vs_k3_01_test_pairs"] = float(np.max(np.abs(resid_base - tp["resid_gbm"])))
align["protein_order_vs_k3_01"] = bool((tp["protein"] == np.array([g[0] for g in groups_te])).all())
cg = np.load(EXP / "K3-06_interaction_energy_residual/candidates.npz")
align["resid_vs_k3_06_candidates"] = float(np.max(np.abs(resid_base - cg["residual_gbm"])))
align["k3_06_quant10_maxdiff"] = float(np.nanmax(np.abs(groups["k3_06_energy_frame10"][1] - cg["quant10"])))
a3 = np.load(EXP / "K3-03_electric_field_residual/analysis_data.npz")
align["resid_vs_k3_03_analysis"] = float(np.max(np.abs(resid_base - a3["residual_gbm"].astype(np.float64))))
align["k3_03_quantities_maxdiff"] = float(np.nanmax(np.abs(groups["k3_03_field"][1] - a3["quantities"].astype(np.float64))))
a4 = np.load(EXP / "K3-04_backbone_angles_residual/analysis_data.npz")
align["k3_04_quantities_maxdiff"] = float(np.nanmax(np.abs(groups["k3_04_backbone"][1] - a4["quantities"].astype(np.float64))))
chi_saved = tp["chi"][:, :, :, 1].astype(np.float64)
chi_mine = groups["k3_01_chi"][1]
d = []
idx = 0
for s in (0, 1):
    for k in range(4):
        ang = chi_saved[:, s, k]
        d.append(np.nanmax(np.abs(chi_mine[:, idx] - np.sin(np.radians(ang)))))
        d.append(np.nanmax(np.abs(chi_mine[:, idx + 1] - np.cos(np.radians(ang)))))
        idx += 2
align["k3_01_chi_sincos_maxdiff"] = float(max(d))
print("alignment:", json.dumps(align, indent=1), flush=True)
bad = [k for k, v in align.items() if isinstance(v, float) and v > 1e-4]
if bad or not align["protein_order_vs_k3_01"]:
    print("PREMISE FAILED: alignment mismatch", bad, flush=True)
    sys.exit(2)


# ---------- 当てはめ ----------
def fit_eval(Xcal, Xte):
    m = HistGradientBoostingRegressor(**GBM).fit(Xcal, mr_cal)
    pred = m.predict(Xte)
    resid = mr_te - pred
    return float(resid.std()), float(r2_score(mr_te, pred)), resid


def with_groups(base_cal, base_te, names, src=None):
    Xc, Xt = [base_cal], [base_te]
    for nm in names:
        c, t = (src or groups)[nm]
        Xc.append(c); Xt.append(t)
    return np.column_stack(Xc), np.column_stack(Xt)


ALL = [nm for nm, _, _ in GROUP_DEFS]
results = {"n_cal": int(len(mr_cal)), "n_test": int(len(mr_te)),
           "premise_base_sd": premise, "alignment_maxdiff": align, "models": {}}
resid_store = {}

def record(tag, Xcal, Xte):
    sd, r2, resid = fit_eval(Xcal, Xte)
    results["models"][tag] = dict(resid_sd=sd, r2=r2, delta_vs_base=sd - REF_BASE_GBM)
    resid_store[tag] = resid
    print(f"{tag:34s} sd {sd:.4f}  R2 {r2:.4f}  delta {sd - REF_BASE_GBM:+.4f}", flush=True)
    return sd

record("base_static5", Scal, Ste)
# グループごとの追加（前提3: 元の記録との照合を含む）
group_alone = {}
for nm in ALL:
    Xc, Xt = with_groups(Scal, Ste, [nm])
    group_alone[nm] = record(f"add_{nm}", Xc, Xt)
mismatch = {nm: abs(group_alone[nm] - REF_GROUP_SD[nm]) for nm in ALL}
results["group_alone_vs_original"] = {
    nm: dict(mine=group_alone[nm], original=REF_GROUP_SD[nm], diff=mismatch[nm]) for nm in ALL}
print("group-alone vs original:", json.dumps(results["group_alone_vs_original"], indent=1), flush=True)
if max(mismatch.values()) > 5e-3:
    print("PREMISE FAILED: group-alone mismatch", mismatch, flush=True)
    Path(ROOT / "kinescope/work/k3_07/premise_failed.json").write_text(
        json.dumps(results, indent=2, default=float))
    sys.exit(2)

# 追加枠の単独確認（K3-02 フレーム0）
sd_hb0 = record("add_k3_02_hb_frame0_only",
                np.column_stack([Scal, hb0_cal]), np.column_stack([Ste, hb0_te]))
results["hb0_alone_vs_original"] = dict(mine=sd_hb0, original=REF_HB0_SD,
                                        diff=abs(sd_hb0 - REF_HB0_SD))

# 全量
Xc_all, Xt_all = with_groups(Scal, Ste, ALL)
record("all_candidates", Xc_all, Xt_all)
record("all_plus_hb_frame0",
       np.column_stack([Xc_all, hb0_cal]), np.column_stack([Xt_all, hb0_te]))
# 1グループずつ抜く
for nm in ALL:
    rest = [x for x in ALL if x != nm]
    Xc, Xt = with_groups(Scal, Ste, rest)
    record(f"all_minus_{nm}", Xc, Xt)

# ---------- 入れ替え対照 ----------
def shuffle_within_groups(F, groups_list, rng):
    F = F.copy()
    idx_by = {}
    for i, g in enumerate(groups_list):
        idx_by.setdefault(g, []).append(i)
    for idx in idx_by.values():
        idx = np.array(idx)
        F[idx] = F[idx][rng.permutation(len(idx))]
    return F

Qcal = np.column_stack([groups[nm][0] for nm in ALL])
Qte = np.column_stack([groups[nm][1] for nm in ALL])
rng = np.random.default_rng(SHUFFLE_SEED)
Qcal_shuf = shuffle_within_groups(Qcal, groups_cal, rng)
sd_a, r2_a, resid_a = fit_eval(np.column_stack([Scal, Qcal_shuf]), Xt_all)
results["models"]["shuffle_refit_calibration"] = dict(
    resid_sd=sd_a, r2=r2_a, delta_vs_base=sd_a - REF_BASE_GBM)
resid_store["shuffle_refit_calibration"] = resid_a
print(f"{'shuffle_refit_calibration':34s} sd {sd_a:.4f}  R2 {r2_a:.4f}", flush=True)

model_all = HistGradientBoostingRegressor(**GBM).fit(Xc_all, mr_cal)
Qte_shuf = shuffle_within_groups(Qte, groups_te, rng)
resid_b = mr_te - model_all.predict(np.column_stack([Ste, Qte_shuf]))
sd_b = float(resid_b.std())
results["models"]["shuffle_eval_test"] = dict(
    resid_sd=sd_b, r2=float(r2_score(mr_te, mr_te - resid_b)), delta_vs_base=sd_b - REF_BASE_GBM)
resid_store["shuffle_eval_test"] = resid_b
print(f"{'shuffle_eval_test':34s} sd {sd_b:.4f}", flush=True)

results["feature_counts"] = dict(base=5, **{nm: groups[nm][0].shape[1] for nm in ALL},
                                 hb_frame0_extra=hb0_cal.shape[1],
                                 all_total=int(Xc_all.shape[1]),
                                 all_plus_hb0_total=int(Xc_all.shape[1] + hb0_cal.shape[1]))
results["judgement"] = dict(
    line_A=0.30,
    all_candidates_sd=results["models"]["all_candidates"]["resid_sd"],
    explains_most=(results["models"]["all_candidates"]["resid_sd"] <= 0.30))

OUT = EXP / "K3-07_all_candidates_combined"
(OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "resid_sd_A", "r2", "delta_vs_base_A"])
    for tag, v in results["models"].items():
        w.writerow([tag, v["resid_sd"], v["r2"], v["delta_vs_base"]])
np.savez_compressed(OUT / "test_residuals.npz",
                    **{k: v.astype(np.float32) for k, v in resid_store.items()})
print("saved results.json / results.csv / test_residuals.npz", flush=True)
print("ALL DONE", flush=True)
