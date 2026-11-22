"""K05 comparison: generated (proposal 22) vs elastic-only vs real series,
all on the same 5-point grid (origin + 0.1/0.3/0.5/1.0 ns)."""
import json
from pathlib import Path
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/tmp/k05_real_baseline.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
stats = base.series_stats

rows = {}
gen5, coarse5, real5 = [], [], []
finals = {"gen": [], "coarse": [], "real": []}
for path in sorted(Path("/tmp/gen22_out").glob("atlas-*.npz")):
    with np.load(path) as z:
        pair, gen, coarse, real = z["pair"], z["gen_dist"], z["coarse_dist"], z["real_sub"]
        d0 = pair[:, 2].astype(np.float64)
        g = np.column_stack([d0, gen]); c = np.column_stack([d0, coarse]); r = np.column_stack([d0, real])
        gen5.append(g); coarse5.append(c); real5.append(r)
        finals["gen"].append(gen[:, -1]); finals["coarse"].append(coarse[:, -1]); finals["real"].append(real[:, -1])
G = np.concatenate(gen5).T; C = np.concatenate(coarse5).T; R = np.concatenate(real5).T
out = {"n_pairs": int(G.shape[1]),
       "generated_5pt": stats(G), "elastic_only_5pt": stats(C), "real_5pt": stats(R),
       "final_distance_1ns": {k: {"mean": float(np.concatenate(v).mean()), "std": float(np.concatenate(v).std()),
                                  "frac_formed_le3.5": float((np.concatenate(v) <= 3.5).mean())}
                              for k, v in finals.items()}}
Path("/tmp/k05_comparison.json").write_text(json.dumps(out, indent=1))
for name in ("generated_5pt", "elastic_only_5pt", "real_5pt"):
    s = out[name]
    print(name, "| mean %.3f std %.3f | per-pair std %.3f range %.3f | incr std %.3f | lag1 %.3f | below %.3f ever %.3f cross %.3f dwell %s" % (
        s["dist_mean"], s["dist_std"], s["per_pair_std_mean"], s["per_pair_range_mean"],
        s["increment_std"], s["lag1_autocorr_mean"], s["frac_frames_below"],
        s["frac_pairs_ever_below"], s["crossings_per_pair_mean"],
        None if s["dwell_below_frames_mean"] is None else round(s["dwell_below_frames_mean"], 3)))
print("final@1ns:", json.dumps(out["final_distance_1ns"], indent=1))
print("n_pairs:", out["n_pairs"])
