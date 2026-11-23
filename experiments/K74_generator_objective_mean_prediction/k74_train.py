"""K74 retraining: #22's correction head, mean-distance objective, CPU.

Same network (110->112->96->424) and same forward pass as proposal 22
(elastic coarse + head-2 correction -> constrain -> 4-point distance
series). Only the objective changes: loss = MSE between the predicted
mean (origin distance + 4 predicted distances, /5) and the target real
mean (mean of the 11-point backward history, K08 definition).

Training rows: teacher 8 proteins, DocD HB candidates at origin frame 10
(k74_build.py outputs). From scratch, seed 722153, AdamW lr 0.0015,
16 epochs, batch 2048, mirroring fit() in hb_learned_coordinate_paths.py
except CUDA (this VM has none) and the loss.
"""
import sys, json, time, hashlib
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
from pathlib import Path
import numpy as np
import torch
from tseries.hb_learned_coordinate_paths import network, constrain, SEED

OUTDIR = Path("/home/hatch/workspace/k74/out")
TEACHER = ("1u84_A", "4u5h_G", "5hub_A", "1x6j_A", "2y1y_A", "1krn_A", "2y4x_B", "4xqm_A")
EPOCHS = 16
BATCH = 2048
LR = 0.0015

def load_rows():
    xs, cs, ss, ts, keys = [], [], [], [], []
    for p in TEACHER:
        for r in (1, 2, 3):
            f = OUTDIR / f"atlas-{p}-analysis-R{r}.npz"
            assert f.is_file(), f"missing training file {f}"
            with np.load(f) as z:
                xs.append(z["descriptors"].astype(np.float32))
                cs.append(z["coarse"].astype(np.float32))
                ss.append(z["sites"].astype(np.float32))
                ts.append(z["real_mean"].astype(np.float32))
                keys.append(np.array([f"{p}-R{r}"] * len(ts[-1])))
    return (np.concatenate(xs), np.concatenate(cs), np.concatenate(ss),
            np.concatenate(ts), np.concatenate(keys))

def predict_mean(model, norm, coarse, sites):
    out = model(norm)
    h2 = out[:, :360].reshape(-1, 6, 4, 5, 3)[:, 2]
    corrected = torch.as_tensor(coarse) + h2
    paths = constrain(torch.as_tensor(sites), corrected[:, None])
    d = torch.linalg.vector_norm(paths[:, :, :, 1], dim=-1)[:, 0, :]
    origin = torch.linalg.vector_norm(torch.as_tensor(sites)[:, 1], dim=-1)
    return (origin + d.sum(dim=1)) / 5.0, d

def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    t0 = time.time()
    x, coarse, sites, target, keys = load_rows()
    print("training rows:", len(x), "target mean %.4f std %.4f" % (target.mean(), target.std()), flush=True)
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-4)
    norm = np.clip((x - center) / scale, -12., 12.).astype(np.float32)
    xt = torch.from_numpy(norm)
    tt = torch.from_numpy(target)
    model = network(norm.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    losses = []
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(x))
        ep_loss = 0.0
        for ids in perm.split(BATCH):
            pred, _ = predict_mean(model, xt[ids], coarse[ids.numpy()], sites[ids.numpy()])
            loss = torch.nn.functional.mse_loss(pred, tt[ids])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(ids)
        ep_loss /= len(x)
        losses.append(ep_loss)
        print(f"epoch {ep+1}/{EPOCHS} mse {ep_loss:.5f} rmse {ep_loss ** 0.5:.4f} sec {time.time()-t0:.0f}", flush=True)
    model.eval()
    with torch.no_grad():
        preds = []
        for ids in torch.arange(len(x)).split(BATCH):
            p, _ = predict_mean(model, xt[ids], coarse[ids.numpy()], sites[ids.numpy()])
            preds.append(p.numpy())
    pred = np.concatenate(preds)
    err = pred - target
    record = dict(seed=SEED, epochs=EPOCHS, batch=BATCH, lr=LR, rows=int(len(x)),
                  epoch_mse=losses, final_rmse=float(np.sqrt(losses[-1])),
                  seconds=round(time.time() - t0, 1),
                  train_error=dict(mean=float(err.mean()), std=float(err.std()),
                                   mae=float(np.abs(err).mean()),
                                   corr=float(np.corrcoef(pred, target)[0, 1])))
    bundle_path = Path("/home/hatch/workspace/k74/retrained_generator.pt")
    torch.save(dict(state=model.state_dict(), center=torch.from_numpy(center),
                    scale=torch.from_numpy(scale)), bundle_path)
    record["sha256"] = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    Path("/home/hatch/workspace/k74/k74_train_log.json").write_text(json.dumps(record, indent=1))
    print("TRAIN DONE", json.dumps(record["train_error"]), "sha256", record["sha256"][:16], flush=True)

if __name__ == "__main__":
    main()
