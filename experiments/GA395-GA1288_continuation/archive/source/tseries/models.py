"""Fit CUDA classifiers and reload checkpoints for CPU scoring."""
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

PILOT = {
    "T1I01": ("logistic", 118, "environment"),
    "T1I02": ("mlp", 142, "competition"),
    "T1I03": ("xgb", 166, "interaction"),
    "T1I04": ("xgb", 118, "environment"),
    "T1I05": ("xgb", 142, "competition"),
    "T1I06": ("mlp", 166, "environment"),
    "T1I07": ("xgb", 142, "environment"),
    "T1I08": ("xgb", 142, "interaction"),
    "T1I09": ("xgb", 166, "environment"),
    "T1I10": ("mlp", 118, "environment"),
}
CONFIRMATION = {f"T1I{index:02}": spec for index, spec in enumerate(PILOT.values(), 11)}
TRIAL_SETS = {"pilot": PILOT, "confirmation": CONFIRMATION}
TRIALS = PILOT | CONFIRMATION


def network(kind, n):
    import torch.nn as nn
    if kind == "logistic":
        return nn.Linear(n, 1)
    return nn.Sequential(nn.Linear(n, 96), nn.SiLU(), nn.Linear(96, 48),
                         nn.SiLU(), nn.Linear(48, 1))


def fit(name, x, y, directory):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model fitting")
    torch.manual_seed(27182)
    kind, width, representation = TRIALS[name]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = dict(name=name, algorithm=kind, width=width, representation=representation,
                device=torch.cuda.get_device_name(), rows=len(y), positives=int(y.sum()),
                seed=27182, torch_version=str(torch.__version__), cuda_runtime=torch.version.cuda)
    started = time.perf_counter()
    if kind == "xgb":
        import xgboost as xgb
        if not xgb.build_info().get("USE_CUDA"):
            raise RuntimeError("XGBoost has no CUDA support")
        config = dict(device="cuda:0", tree_method="hist", objective="binary:logistic",
                      max_depth=5, eta=0.05, subsample=0.85, colsample_bytree=0.85,
                      nthread=4, seed=27182)
        rounds = 240
        samples = []
        done = threading.Event()

        def sample_gpu():
            while not done.is_set():
                try:
                    reading = subprocess.check_output(
                        ["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                         "--format=csv,noheader,nounits"], text=True, timeout=2)
                    samples.append(int(reading.strip()))
                except (OSError, ValueError, subprocess.SubprocessError):
                    return
                done.wait(0.2)

        monitor = threading.Thread(target=sample_gpu, daemon=True)
        monitor.start()
        try:
            train = xgb.QuantileDMatrix(x, label=y, nthread=4)
            booster = xgb.Booster(config, [train])
            for step in range(rounds):
                booster.update(train, step)
                if step == 0:
                    actual = json.loads(booster.save_config())["learner"]
                    active = actual["generic_param"]["device"]
                    updater = actual["gradient_booster"]["gbtree_train_param"]["updater_seq"]
                    if not active.startswith("cuda") or "gpu_hist" not in updater:
                        raise RuntimeError(f"Unexpected XGBoost training path {active}, {updater}")
        finally:
            done.set()
            monitor.join()
        booster.save_model(directory / f"{name}.ubj")
        meta.update(rounds=rounds, config=config, active_device=active, updater=updater,
                    xgboost_version=xgb.__version__,
                    gpu_device_memory_start_mib=samples[0] if samples else None,
                    gpu_device_memory_sampled_peak_mib=max(samples) if samples else None,
                    gpu_memory_samples=len(samples))
    else:
        torch.cuda.reset_peak_memory_stats()
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(x.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
        train = torch.as_tensor((x - mean) / scale, device="cuda")
        target = torch.as_tensor(y, dtype=torch.float32, device="cuda")
        model = network(kind, x.shape[1]).to("cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
        epochs, batch_size = 30, 4096
        assert train.is_cuda and target.is_cuda and all(p.is_cuda for p in model.parameters())
        for _ in range(epochs):
            indices = torch.randperm(len(target), device="cuda")
            for batch in indices.split(batch_size):
                opt.zero_grad(set_to_none=True)
                logits = model(train[batch]).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target[batch])
                loss.backward()
                opt.step()
        torch.cuda.synchronize()
        meta.update(epochs=epochs, batch_size=batch_size, learning_rate=0.002, weight_decay=0.01,
                    active_device=str(next(model.parameters()).device),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        torch.save(dict(state=model.cpu().state_dict(), mean=torch.from_numpy(mean),
                        scale=torch.from_numpy(scale)), directory / f"{name}.pt")
    meta["fit_seconds"] = time.perf_counter() - started
    (directory / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_cpu(name, directory):
    directory = Path(directory)
    meta = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    kind = meta["algorithm"]
    if kind == "xgb":
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(directory / f"{name}.ubj")
        booster.set_param({"device": "cpu", "nthread": 4})
        return meta, booster.inplace_predict
    import torch
    torch.set_num_threads(4)
    state = torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True)
    model = network(kind, len(state["mean"])).eval()
    model.load_state_dict(state["state"])

    @torch.inference_mode()
    def predict(x):
        xs = torch.from_numpy(x)
        normalized = (xs - state["mean"]) / state["scale"]
        return torch.sigmoid(model(normalized).squeeze(1)).numpy()

    return meta, predict
