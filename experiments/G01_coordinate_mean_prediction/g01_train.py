"""G01: can a model that learns from the origin-frame coordinates themselves
predict the candidate pair's history mean distance?

Input per candidate (origin frame = frame 10, 1 ns; no future information):
  atoms within 10 A of the donor or acceptor atom (MIC, per-axis box),
  per-atom features = MIC vectors to donor/acceptor, distances, element
  one-hot (C/N/O/S), donor/acceptor flags, same-residue flags.
  Two variants: (A) element only, (B) + learned atom-name embedding.
  Per-candidate coordinates are expressed in a canonical frame built from
  the pair geometry (donor->acceptor axis, cloud-centroid direction), a
  fixed geometric preprocessing with no chemical information.

Stage 1: residual std of predicted vs real mean on the test proteins,
  compared with K23/K40 references (recomputed here on the same cache).
Stage 2: predicted mean as the only history feature in the DocD correction
  model (HistGradientBoostingClassifier, DocD settings); recovery of the
  1 ns gain. Train-side predicted means are leave-one-protein-out (OOF).

Runs are checkpointed per fold/seed under ~/workspace/kinescope/work/g01/
and skipped when their prediction files already exist.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.metrics import average_precision_score

torch.set_num_threads(2)
WORK = Path("/home/hatch/workspace/kinescope/work/g01")
EXP = Path("/home/hatch/workspace/kinescope/repo/experiments/G01_coordinate_mean_prediction")
TRAIN_P = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A")
TEST_P = ("1mix_A", "2qt4_A")
HORIZONS = (1, 5, 10, 30, 50, 90)
DEVICE = "cpu"

# ---------------------------------------------------------------- load data
trajs = []
for p in TRAIN_P + TEST_P:
    for r in (1, 2, 3):
        z = np.load(WORK / f"{p}_R{r}.npz")
        d = {k: z[k] for k in z.files if k not in ("protein", "replica")}
        d["protein"], d["replica"] = p, r
        trajs.append(d)
N_NAME = int(max(t["names"].max() for t in trajs)) + 1

for t in trajs:
    t["split"] = "train" if t["protein"] in TRAIN_P else "test"
    t["n"] = len(t["target"])
off = 0
for t in trajs:
    t["idx"] = np.arange(off, off + t["n"])
    off += t["n"]
N = off
protein_of = np.concatenate([np.full(t["n"], t["protein"]) for t in trajs])
split_of = np.concatenate([np.full(t["n"], t["split"]) for t in trajs])
target_all = np.concatenate([t["target"] for t in trajs])
print("candidates", N, "train", int((split_of == "train").sum()),
      "test", int((split_of == "test").sum()), flush=True)


# Pre-assemble global padded arrays once (per-batch assembly is too slow).
_KMAX = max(int(t["feats"].shape[1]) for t in trajs)
FEATS = np.zeros((N, _KMAX, 16), dtype=np.float32)
NAMES = np.zeros((N, _KMAX), dtype=np.int64)
MASK = np.zeros((N, _KMAX), dtype=np.float32)
GLOB = np.zeros((N, 9), dtype=np.float32)
SIZES = np.zeros(N, dtype=np.int64)
for t in trajs:
    ii = t["idx"]
    k = int(t["feats"].shape[1])
    FEATS[ii, :k] = t["feats"]
    NAMES[ii, :k] = t["names"]
    MASK[ii, :k] = t["mask"]
    GLOB[ii, :8] = t["pair_elem"]
    GLOB[ii, 8] = t["dist_da"] / 6.0
    SIZES[ii] = t["sizes"]
FEATS_T = torch.tensor(FEATS)
NAMES_T = torch.tensor(NAMES)
MASK_T = torch.tensor(MASK)
GLOB_T = torch.tensor(GLOB)
del FEATS, NAMES, MASK, GLOB


def get_batch(idx):
    kmax = int(SIZES[idx].max())
    return (FEATS_T[idx, :kmax], NAMES_T[idx, :kmax],
            MASK_T[idx, :kmax], GLOB_T[idx])


def canonicalize(feats, mask):
    """Express relD/relA in the per-candidate canonical frame.

    e1 = donor->acceptor axis; e2 = cloud-centroid direction with the e1
    component removed; e3 = e1 x e2. feats: (B,K,16) with relD at 0:3 and
    relA at 3:6. The donor->acceptor vector is the acceptor atom's relD,
    found via the isA flag (channel 13)."""
    is_a = feats[:, :, 13:14]                       # (B,K,1)
    v_da = (feats[:, :, 0:3] * is_a).sum(dim=1)      # (B,3)
    e1 = v_da / v_da.norm(dim=1, keepdim=True).clamp_min(1e-9)
    rel_d = feats[:, :, 0:3]
    cnt = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    centroid = (rel_d * mask.unsqueeze(-1)).sum(dim=1) / cnt
    c_perp = centroid - (centroid * e1).sum(dim=1, keepdim=True) * e1
    nrm = c_perp.norm(dim=1, keepdim=True)
    # fallback for (near-)degenerate clouds: any vector perpendicular to e1
    alt = torch.zeros_like(e1)
    alt[:, 0] = 1.0
    alt = alt - (alt * e1).sum(dim=1, keepdim=True) * e1
    alt = alt / alt.norm(dim=1, keepdim=True).clamp_min(1e-9)
    e2 = torch.where(nrm > 1e-6, c_perp / nrm.clamp_min(1e-9), alt)
    e3 = torch.cross(e1, e2, dim=1)
    R = torch.stack([e1, e2, e3], dim=1)             # (B,3,3) rows are axes
    out = feats.clone()
    out[:, :, 0:3] = torch.einsum("bij,bkj->bki", R, rel_d)
    out[:, :, 3:6] = torch.einsum("bij,bkj->bki", R, feats[:, :, 3:6])
    return out


class MeanNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        in_dim = 16
        if arch == "name":
            self.name_emb = nn.Embedding(N_NAME, 24)
            in_dim += 24
        self.atom_mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(256 + 9, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, feats, names, mask, glob):
        x = canonicalize(feats, mask)
        if self.arch == "name":
            x = torch.cat([x, self.name_emb(names)], dim=-1)
        h = self.atom_mlp(x)                          # (B,K,128)
        m = mask.unsqueeze(-1)
        mean_pool = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        h_big = h.masked_fill(m == 0, -1e9)
        max_pool = h_big.max(dim=1).values
        z = torch.cat([mean_pool, max_pool, glob], dim=-1)
        return self.head(z).squeeze(-1)


def train_one(arch, tr_idx, seed, max_epochs=60, patience=10):
    """Train on tr_idx with an internal 15% candidate-level validation
    split (used only for early stopping). Returns (model, y_mean, y_std,
    best_epoch)."""
    rng = np.random.RandomState(1234)
    perm = rng.permutation(len(tr_idx))
    n_val = int(0.15 * len(tr_idx))
    val_idx = tr_idx[perm[:n_val]]
    fit_idx = tr_idx[perm[n_val:]]
    y = target_all
    y_mean, y_std = float(y[fit_idx].mean()), float(y[fit_idx].std())
    torch.manual_seed(seed)
    model = MeanNet(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    lossf = nn.MSELoss()

    def run(idx, train=False):
        outs = []
        for s in range(0, len(idx), 256):
            b = idx[s:s + 128]
            f, nm, mk, gl = get_batch(b)
            pred = model(f, nm, mk, gl)
            tgt = torch.tensor((y[b] - y_mean) / y_std, dtype=torch.float32)
            loss = lossf(pred, tgt)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            outs.append(loss.item())
        return float(np.mean(outs))

    best, best_state, best_ep, bad = float("inf"), None, -1, 0
    for ep in range(max_epochs):
        model.train()
        run(rng.permutation(fit_idx), train=True)
        sched.step()
        model.eval()
        with torch.no_grad():
            v = run(val_idx)
        if v < best - 1e-4:
            best, best_ep, bad = v, ep, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, y_mean, y_std, best_ep


def predict(model, y_mean, y_std, idx):
    outs = []
    with torch.no_grad():
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            f, nm, mk, gl = get_batch(b)
            outs.append(model(f, nm, mk, gl).numpy() * y_std + y_mean)
    return np.concatenate(outs)


def resid_stats(pred, true):
    r = pred - true
    return dict(std=float(r.std()), bias=float(r.mean()),
                rmse=float(np.sqrt((r ** 2).mean())), mae=float(np.abs(r).mean()),
                corr=float(np.corrcoef(pred, true)[0, 1]),
                r2=float(1 - (r ** 2).sum() / ((true - true.mean()) ** 2).sum()))


results = {}
test_idx = np.flatnonzero(split_of == "test")
train_idx = np.flatnonzero(split_of == "train")

import os
if os.environ.get("G01_SMOKE"):
    m, ym, ys, ep = train_one("name", train_idx[:2000], seed=0, max_epochs=3)
    pr = predict(m, ym, ys, test_idx[:500])
    print("smoke ok, best_epoch", ep, "pred head", pr[:5], flush=True)
    raise SystemExit

# ------------------------------------------------------------- references
X_all = np.concatenate([t["X"] for t in trajs]).astype(np.float64)
gbr = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.075,
                                    max_leaf_nodes=31, min_samples_leaf=15,
                                    l2_regularization=0.02, random_state=42)
gbr.fit(X_all[train_idx], target_all[train_idx])
ref_static = resid_stats(gbr.predict(X_all[test_idx]), target_all[test_idx])
ref_origin = resid_stats(X_all[test_idx, 0], target_all[test_idx])
results["references"] = {"static7_histgbr": ref_static,
                         "origin_distance_only": ref_origin,
                         "replica_floor_std": 0.249,
                         "generator_error_std": 0.483,
                         "rich99_std": 0.4698}
print("references:", json.dumps(results["references"], indent=1), flush=True)

# ------------------------------------------------------------- stage 1
stage1 = {}
for arch in ("elem", "name"):
    # leave-one-protein-out over the 4 train proteins -> OOF predictions
    oof_file = WORK / f"oof_{arch}.npz"
    if oof_file.exists():
        z = np.load(oof_file)
        oof_pred, fold_test = z["oof"], z["fold_test"]
        fold_epochs = z["fold_epochs"].tolist()
    else:
        part_file = WORK / f"oof_{arch}_partial.npz"
        done = []
        oof_pred = np.full(len(train_idx), np.nan)
        fold_test_map, fold_epochs = {}, []
        if part_file.exists():
            z = np.load(part_file)
            oof_pred = z["oof"]
            done = [str(x) for x in z["done"]]
            fold_epochs = z["fold_epochs"].tolist()
            for i, p in enumerate(done):
                fold_test_map[p] = z["fold_test"][i]
        for p in TRAIN_P:
            if p in done:
                continue
            f_tr = train_idx[protein_of[train_idx] != p]
            f_te = train_idx[protein_of[train_idx] == p]
            t0 = time.time()
            model, ym, ys, bep = train_one(arch, f_tr, seed=0,
                                           max_epochs=50, patience=8)
            oof_pred[np.isin(train_idx, f_te)] = predict(model, ym, ys, f_te)
            fold_test_map[p] = predict(model, ym, ys, test_idx)
            fold_epochs.append(bep)
            done.append(p)
            np.savez(part_file, oof=oof_pred, done=np.array(done),
                     fold_test=np.stack([fold_test_map[q] for q in done]),
                     fold_epochs=np.array(fold_epochs))
            print(f"[{arch}] LOPO {p}: best_epoch {bep}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        fold_test = np.stack([fold_test_map[q] for q in TRAIN_P])
        np.savez(oof_file, oof=oof_pred, fold_test=fold_test,
                 fold_epochs=np.array(fold_epochs))
    # final model on all 4 train proteins, 2 seeds averaged
    fin_file = WORK / f"final_{arch}.npz"
    if fin_file.exists():
        z = np.load(fin_file)
        test_pred = z["test_pred"]
        final_epochs = z["final_epochs"].tolist()
    else:
        preds, final_epochs = [], []
        for seed in (0, 1):
            t0 = time.time()
            model, ym, ys, bep = train_one(arch, train_idx, seed=seed,
                                           max_epochs=50, patience=8)
            preds.append(predict(model, ym, ys, test_idx))
            final_epochs.append(bep)
            print(f"[{arch}] final seed {seed}: best_epoch {bep}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        test_pred = np.mean(preds, axis=0)
        np.savez(fin_file, test_pred=test_pred,
                 final_epochs=np.array(final_epochs))
    np.savez(WORK / f"pred_{arch}.npz", oof_train=oof_pred, test=test_pred)
    st = {"test_all": resid_stats(test_pred, target_all[test_idx]),
          "test_lopo_ensemble": resid_stats(fold_test.mean(axis=0),
                                            target_all[test_idx]),
          "fold_epochs": fold_epochs, "final_epochs": final_epochs}
    for p in TEST_P:
        m = protein_of[test_idx] == p
        st[f"test_{p}"] = resid_stats(test_pred[m], target_all[test_idx][m])
    stage1[arch] = st
    print(f"[{arch}] stage1 test residual std = {st['test_all']['std']:.4f}",
          flush=True)
results["stage1"] = stage1

# ------------------------------------------------------------- stage 2
hist6_all = np.concatenate([
    np.stack([t["hist"][0], t["hist"].mean(axis=0), t["hist"].std(axis=0),
              t["hist"].min(axis=0), t["hist"][-1] - t["hist"][0],
              (t["hist"] <= 3.5).mean(axis=0)], axis=1) for t in trajs])
realmean_all = target_all
y_all = np.concatenate([t["y"] for t in trajs])
predmean = {}
for arch in ("elem", "name"):
    z = np.load(WORK / f"pred_{arch}.npz")
    pm = np.empty(N)
    pm[train_idx] = z["oof_train"]
    pm[test_idx] = z["test"]
    predmean[arch] = pm

GBC = dict(max_iter=120, learning_rate=0.075, max_leaf_nodes=31,
           min_samples_leaf=15, l2_regularization=0.02, random_state=42)


def auprcs(feat_fn):
    out = {}
    Xtr, Xte = feat_fn(train_idx), feat_fn(test_idx)
    for hi, h in enumerate(HORIZONS):
        clf = HistGradientBoostingClassifier(**GBC)
        clf.fit(Xtr, y_all[train_idx, hi])
        out[str(h)] = float(average_precision_score(
            y_all[test_idx, hi], clf.predict_proba(Xte)[:, 1]))
    return out


conds = {
    "static": lambda idx: X_all[idx],
    "static_real6": lambda idx: np.column_stack([X_all[idx], hist6_all[idx]]),
    "static_realmean": lambda idx: np.column_stack(
        [X_all[idx], realmean_all[idx]]),
    "static_predmean_elem": lambda idx: np.column_stack(
        [X_all[idx], predmean["elem"][idx]]),
    "static_predmean_name": lambda idx: np.column_stack(
        [X_all[idx], predmean["name"][idx]]),
}
stage2 = {name: auprcs(fn) for name, fn in conds.items()}
gain = {name: stage2[name]["1"] - stage2["static"]["1"] for name in stage2}
rec = {name: (gain[name] / gain["static_real6"] if name != "static" else 0.0)
       for name in gain}
results["stage2"] = {"auprc": stage2, "gain_1ns": gain,
                     "recovery_vs_real6_1ns": rec,
                     "recovery_vs_realmean_1ns": {
                         name: gain[name] / gain["static_realmean"]
                         for name in gain}}
print("stage2 gains @1ns:", json.dumps(gain), flush=True)
print("stage2 recovery vs real6 @1ns:", json.dumps(rec), flush=True)

json.dump(results, open(EXP / "results.json", "w"), indent=1, ensure_ascii=False)
print("wrote", EXP / "results.json", flush=True)
