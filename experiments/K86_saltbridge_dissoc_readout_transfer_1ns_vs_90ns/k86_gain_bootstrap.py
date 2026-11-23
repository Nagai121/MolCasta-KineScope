"""K86 follow-up: bootstrap 95% CIs for the transfer GAINS themselves
(not the ratios), from the saved test predictions in k86_predictions.npz.
All gains use the K56 definition (static reference trained at the EVAL
horizon). The ratio CIs are in k86_bootstrap.json (computed by k86_why.py).
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
z = np.load(HERE / "k86_predictions.npz")
Yte = z["Yte"]
preds = {k: z[k] for k in z.files if k.endswith("]")}
H = (1, 5, 10, 30, 50, 90)
rng = np.random.default_rng(20261123)
N = len(Yte)


def ap(feat, h1, h2, Y, P):
    return average_precision_score(Y[:, H.index(h2)], P[f"{feat}_train[{h1}]"])


def gain(feat, h1, h2, Y, P):
    return ap(feat, h1, h2, Y, P) - ap("static", h2, h2, Y, P)


keys = [("history", 1, 90), ("history", 90, 1), ("history", 90, 90),
        ("history", 1, 1), ("mean_only", 1, 90), ("mean_only", 90, 1)]
names = ["history 1->90", "history 90->1", "history matched@90",
         "history matched@1", "mean_only 1->90", "mean_only 90->1"]
point = {n: gain(f, a, b, Yte, preds) for n, (f, a, b) in zip(names, keys)}
B = 2000
G = {n: [] for n in names}
for _ in range(B):
    idx = rng.integers(0, N, N)
    Yb = Yte[idx]
    Pr = {k: v[idx] for k, v in preds.items()}
    for n, (f, a, b) in zip(names, keys):
        G[n].append(gain(f, a, b, Yb, Pr))
out = {}
for n in names:
    v = np.array(G[n])
    out[n] = dict(point=float(point[n]), mean=float(v.mean()),
                  lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                  p_gt_0=float((v > 0).mean()))
    print(f"bootstrap gain {n}: point {point[n]:.4f} mean {out[n]['mean']:.4f} "
          f"95% CI [{out[n]['lo']:.4f}, {out[n]['hi']:.4f}] P(>0)={out[n]['p_gt_0']:.3f}")
p = HERE / "k86_bootstrap.json"
d = json.loads(p.read_text())
d["transfer_gain_bootstrap"] = out
p.write_text(json.dumps(d, indent=2))
print("updated", p)
