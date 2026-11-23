import numpy as np, json
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

CACHE = Path("/home/hatch/workspace/kinescope/work/k15_cache")
REPO = Path("/home/hatch/workspace/kinescope/repo")
SPLIT = json.loads((REPO / "experiments/K15_saltbridge_expansion/split.json").read_text())
TRAIN_P = tuple(SPLIT["orig_train"]) + tuple(SPLIT["train_additions"])
TEST_P = tuple(SPLIT["orig_test"]) + tuple(SPLIT["test_additions"])
BOUND = 4.0

def load(prots):
    parts = []
    for p in prots:
        for r in (1, 2, 3):
            z = np.load(CACHE / f"{p}_R{r}_salt.npz")
            parts.append((p, {k: z[k] for k in z.files}))
    return parts

def cat(parts, key):
    return np.concatenate([d[key] for _, d in parts])

def hist6(s, b=BOUND):
    return np.stack([s[:, 0], s.mean(1), s.std(1), s.min(1),
                     s[:, -1] - s[:, 0], (s <= b).mean(1)], axis=1).astype(np.float32)

def learner(X, y):
    return HistGradientBoostingClassifier(max_iter=120, learning_rate=.075, max_leaf_nodes=31,
                                          min_samples_leaf=15, l2_regularization=.02,
                                          random_state=42).fit(X, y)

train, test = load(TRAIN_P), load(TEST_P)
Xtr, Xte = cat(train, "X"), cat(test, "X")
Htr, Hte = hist6(cat(train, "hist")), hist6(cat(test, "hist"))
Ytr, Yte = cat(train, "y"), cat(test, "y")
prot = np.concatenate([[p] * len(d["y"]) for p, d in test])

m_s1 = learner(Xtr, Ytr[:, 0])
m_h1 = learner(np.concatenate([Xtr, Htr], 1), Ytr[:, 0])
m_h90 = learner(np.concatenate([Xtr, Htr], 1), Ytr[:, 5])
p_s = m_s1.predict_proba(Xte)[:, 1]
p_h1 = m_h1.predict_proba(np.concatenate([Xte, Hte], 1))[:, 1]
p_h90 = m_h90.predict_proba(np.concatenate([Xte, Hte], 1))[:, 1]
y = Yte[:, 0]
print(f"{'protein':10s} {'n':>4} {'pos':>4} {'static':>7} {'hist1':>7} {'hist90':>7}")
for name in TEST_P:
    m = prot == name
    yy = y[m]
    if 0 < yy.sum() < len(yy):
        print(f"{name:10s} {m.sum():4d} {int(yy.sum()):4d} "
              f"{average_precision_score(yy, p_s[m]):7.3f} "
              f"{average_precision_score(yy, p_h1[m]):7.3f} "
              f"{average_precision_score(yy, p_h90[m]):7.3f}")
    else:
        print(f"{name:10s} {m.sum():4d} {int(yy.sum()):4d}  (no positives or all positive)")
