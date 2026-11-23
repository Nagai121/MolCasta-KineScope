"""G18 前提確認: 第22案の生成平均は水素結合の解離の候補に使えるか.

G18 は、0.1 ns の切り取り平均に第22案の生成平均を足す混合を、
水素結合の解離（資料Dの条件、1 ns先）で測るはずだった。
実行前に、生成系列（K05 の generated_series.npz、第22案の5点前方系列）が
解離の候補（K46 のキャッシュ、6,131件）を覆っているか確かめる。

結論（このスクリプトの出力）: 覆っていない。解離の候補 6,131件のうち
生成系列に対応するものは 0件。理由は、生成系列のペアは起点距離が
3.5〜6.0 Å の形成候補だけで、解離の候補は起点距離が 3.5 Å 以下だから。
距離の範囲が重ならないので、原子番号の付け方の違いでは説明できない。

生成器のチェックポイント .pt ファイルは読み込まない（G11 と同じ決め事）。
"""
import json
from pathlib import Path

import numpy as np

CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache_k46")
GEN = Path("/home/hatch/workspace/kinescope/repo/experiments/K05_generated_quality/generated_series.npz")
OUT = Path(__file__).resolve().parent
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")

z = np.load(GEN)
gen_keys = {(str(t), int(p[0]), int(p[1]))
            for t, p in zip(z["tid"], z["pair"])}
gen_d0 = z["pair"][:, 2].astype(np.float64)
print("generated series rows:", len(gen_keys))
print("generated d0 range: %.6f .. %.6f A" % (gen_d0.min(), gen_d0.max()))
print("generated rows with d0 <= 3.5 A:", int((gen_d0 <= 3.5).sum()))

per_traj = {}
total = covered = 0
max_d0_diss = 0.0
for p in PROTEINS:
    for r in (1, 2, 3):
        d = np.load(CACHE / f"{p}_R{r}.npz")
        tid = f"atlas-{p}-analysis-R{r}"
        n = len(d["donor_atom"])
        hit = sum(1 for da, aa in zip(d["donor_atom"], d["acceptor_atom"])
                  if (tid, int(da), int(aa)) in gen_keys)
        per_traj[f"{p}_R{r}"] = {"n": int(n), "covered": int(hit)}
        total += n
        covered += hit
        max_d0_diss = max(max_d0_diss, float(d["hist"][-1].max()))

print("dissociation candidates total:", total, "covered by generated series:", covered)
print("dissociation frame-10 distance max: %.6f A (all <= 3.5 A: %s)"
      % (max_d0_diss, max_d0_diss <= 3.5))

out = {
    "generated_series": "experiments/K05_generated_quality/generated_series.npz (method 22, 5-point forward series)",
    "generated_rows": int(len(gen_keys)),
    "generated_d0_min_A": float(gen_d0.min()),
    "generated_d0_max_A": float(gen_d0.max()),
    "generated_rows_d0_le_3.5_A": int((gen_d0 <= 3.5).sum()),
    "dissociation_candidates_total": int(total),
    "dissociation_candidates_covered": int(covered),
    "dissociation_frame10_distance_max_A": float(max_d0_diss),
    "per_trajectory": per_traj,
    "conclusion": (
        "No dissociation candidate is covered by the method-22 generated series. "
        "The generated series only contains formation candidates (origin distance "
        "3.5-6.0 A); dissociation candidates all have origin distance <= 3.5 A, "
        "so the two sets are disjoint by construction. G18 as specified "
        "(prefix mean + method-22 generated mean) cannot be measured with saved data. "
        "Regenerating method 22 for dissociation candidates would require loading "
        "the generator checkpoint (.pt), which this experiment does not do."
    ),
}
(OUT / "coverage.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("saved", OUT / "coverage.json")
