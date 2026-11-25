"""CUDA exploration for salt-bridge and residue-contact formation."""
import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .data import REPLICAS, TRAIN, VALIDATION, TEST
from .events import TASK_NAMES, feature_view, shard
from .legacy import load_models

TARGET_RECALL = 0.80
WORK_REDUCTION = 3.0
EXTRA_TRAINING = ("1ifg_A", "2b0a_A", "2qev_A", "2w0g_A")

# First wave varies the selector, width, representation and optional legacy score.
CONFIGS = (
    ("xgb", 12, "geometry", False, 3, 1.0),
    ("xgb", 20, "geometry", False, 4, 1.0),
    ("xgb", 28, "geometry", False, 5, 1.0),
    ("xgb", 28, "competition", False, 5, 1.0),
    ("xgb", 28, "packing", False, 5, 1.0),
    ("xgb", 28, "packing", False, 7, 1.0),
    ("xgb", 28, "geometry", True, 4, 1.0),
    ("xgb", 28, "competition", True, 5, 1.0),
    ("xgb", 28, "packing", True, 6, 1.0),
    ("xgb", 20, "geometry", True, 3, 1.0),
    ("xgb", 28, "competition", False, 4, 2.0),
    ("xgb", 28, "packing", True, 4, 2.0),
    ("mlp", 12, "geometry", False, 32, 1.0),
    ("mlp", 28, "geometry", False, 64, 1.0),
    ("mlp", 28, "competition", False, 96, 1.0),
    ("mlp", 28, "packing", False, 96, 1.0),
    ("mlp", 28, "geometry", True, 64, 1.0),
    ("mlp", 28, "competition", True, 96, 1.0),
    ("mlp", 28, "packing", True, 96, 1.0),
    ("mlp", 28, "packing", True, 64, 2.0),
)
SCREENS = (
    (None, None), (7.0, None), (9.0, None), (12.0, None), (16.0, None),
    (None, 8), (None, 16), (None, 32), (None, 64), (None, 128),
)
DENSE_CONFIGS = {
    "salt": (
        ("mlp", 28, "competition", True, 96, 1.0),
        ("mlp-local", 28, "competition", True, 96, 1.5),
        ("mlp-local", 28, "geometry", False, 64, 1.0),
        ("mlp-local", 28, "packing", True, 128, 1.0),
        ("mlp", 28, "competition", True, 128, 0.75),
        ("xgb-local", 20, "geometry", True, 2, 1.0),
        ("xgb-local", 28, "competition", True, 3, 1.0),
        ("xgb-local", 28, "packing", True, 4, 0.75),
        ("xgb", 28, "geometry", False, 3, 1.5),
        ("xgb", 28, "competition", True, 5, 1.0),
    ),
    "contact": (
        ("xgb-local", 28, "competition", True, 4, 1.0),
        ("xgb-local", 28, "geometry", True, 5, 1.5),
        ("xgb-local", 28, "packing", True, 7, 1.0),
        ("xgb", 28, "geometry", True, 7, 1.0),
        ("xgb", 28, "competition", True, 5, 1.0),
        ("xgb", 28, "packing", False, 6, 1.0),
        ("mlp-local", 28, "competition", True, 128, 1.0),
        ("mlp-local", 28, "geometry", True, 128, 1.5),
        ("mlp", 28, "geometry", True, 192, 1.0),
        ("mlp", 28, "packing", False, 192, 0.75),
    ),
}
HISTORY_CONFIGS = (
    ("xgb", 28, "motion", False, 4, 1.0),
    ("xgb-local", 28, "motion", True, 5, 1.0),
    ("xgb-local", 28, "occupancy", True, 5, 1.0),
    ("xgb", 28, "occupancy", False, 5, 1.0),
    ("xgb", 28, "history-all", True, 5, 1.0),
    ("xgb-local", 28, "history-all", True, 7, 1.0),
    ("xgb-local", 28, "history-all", False, 4, 1.5),
    ("xgb", 12, "history-all", True, 3, 0.75),
    ("mlp-local", 28, "history-all", True, 64, 1.0),
    ("xgb-local", 28, "history-all", True, 4, 0.75),
    ("xgb", 28, "occupancy", True, 3, 1.0),
    ("xgb", 28, "history-all", True, 6, 1.0),
)
KINETIC_CONFIGS = (
    ("xgb-local", 28, "kinetics", True, 4, 1.0),
    ("xgb", 28, "kinetics", False, 5, 1.0),
    ("xgb", 28, "kinetics-all", True, 4, 1.0),
    ("xgb-local", 28, "kinetics-all", True, 5, 1.0),
    ("xgb-local", 28, "kinetics-all", True, 6, 1.0),
    ("xgb-local", 12, "kinetics-all", True, 3, 1.0),
    ("xgb", 28, "kinetics-all", False, 6, 1.5),
    ("xgb-local", 28, "kinetics-all", True, 4, 0.75),
    ("mlp-local", 28, "kinetics-all", True, 64, 1.0),
    ("mlp-local", 28, "kinetics", True, 64, 1.0),
)
EXTENDED_CONFIGS = (
    ("xgb-local", 28, "kinetics-all", True, 3, 1.0),
    ("xgb-local", 28, "kinetics-all", True, 4, 0.75),
    ("xgb", 28, "kinetics-all", True, 5, 1.0),
    ("xgb-local", 28, "kinetics-all", False, 5, 1.5),
    ("xgb-local", 28, "history-all", True, 5, 1.0),
    ("xgb", 12, "occupancy", True, 3, 1.0),
)
SOFT_CONFIGS = (
    ("xgb-soft-local", 28, "kinetics-all", True, 3, 1.0),
    ("xgb-soft-local", 28, "kinetics-all", True, 5, 1.0),
    ("xgb-soft-wide-local", 28, "kinetics-all", True, 4, 1.0),
    ("xgb-soft-wide-local", 28, "kinetics-all", False, 5, 1.0),
    ("xgb-soft-local", 12, "kinetics-all", True, 3, 1.0),
    ("xgb-soft-local", 28, "history-all", True, 5, 1.0),
)
TAIL_CONFIGS = (
    ("mlp-tail", 28, "kinetics-all", True, 96, 1.0),
    ("mlp-tail-strong", 28, "kinetics-all", True, 96, 1.0),
    ("mlp-tail", 12, "kinetics-all", True, 64, 1.0),
    ("mlp-tail", 28, "history-all", False, 128, 1.0),
)
REFINE_CONFIGS = tuple(
    (f"xgb-soft-tune-{bandwidth}-{mix}-{rounds}-local", 28, "kinetics-all", True, depth, 1.0)
    for depth, (bandwidth, mix), rounds in itertools.product(
        (2, 3, 4), ((0.15, 0.15), (0.25, 0.40), (0.40, 0.35)), (120, 240, 400))
)


def dump(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def get_data(root, proteins, stride, models, task, history=False, kinetics=False, targets=False):
    result = []
    for protein in proteins:
        for replica in REPLICAS:
            with np.load(shard(root, protein, replica, stride, models)) as source:
                part = {k: source[f"{task}_{k}"] for k in ("pair", "base", "y", "old", "ends")}
                part["protein"] = protein
                part["meta"] = json.loads(source["meta"].item())
                if history:
                    from .event_history import history_shard
                    with np.load(history_shard(root, protein, replica, stride, models)) as past:
                        part["history"] = past[task]
                    if len(part["history"]) != len(part["y"]):
                        raise ValueError("History and candidate rows differ")
                if kinetics:
                    from .event_kinetics import kinetics_shard
                    with np.load(kinetics_shard(root, protein, replica, stride, models)) as past:
                        part["kinetics"] = past[task]
                    if len(part["kinetics"]) != len(part["y"]):
                        raise ValueError("Kinetics and candidate rows differ")
                if targets:
                    from .event_targets import training_distances
                    with np.load(training_distances(root, protein, replica, stride, models)) as target:
                        part["future_distance"] = target[task]
                    part["threshold"] = 4.0 if task == "salt" else 4.5
                result.append(part)
    return result


def view(part, config):
    _, width, representation, inherit, _, _ = config
    if representation.startswith("hb-origin-"):
        from .data import inputs
        kind = representation.removeprefix("hb-origin-")
        if kind == "union":
            local, motion, motif, exposed = (part[name] for name in
                                             ("local","directional","motif","exposure"))
            extra = np.column_stack((local[:, :3], motion[:, [0, 3]],
                                     motif[:, [4, 8]], exposed[:, 7]))
            extended = np.column_stack((part["pair"], extra))
            assert extended.dtype == np.float32 and extended.shape[1] == 16
            x = np.column_stack((part["base"][:, :width], extended[:, 8:],
                                 local[:, 3:], motion[:, [1,2,4,5,6,7]],
                                 motif[:, [0,1,2,3,5,6,7,*range(9,18)]],
                                 exposed[:, [*range(7),*range(8,12)]]))
        elif kind == "context":
            from .hb_origin_context import context
            extended=np.column_stack((part["pair"],context(part)))
            assert extended.dtype == np.float32 and extended.shape[1] == 16
            x=np.column_stack((part["base"][:,:width],extended[:,8:],part["local"]))
        elif kind.startswith("stacked-"):
            selected=kind.removeprefix("stacked-")
            block=part[selected]
            extended=np.column_stack((part["pair"],block[:,:8]))
            assert extended.dtype == np.float32 and extended.shape[1] == 16
            remaining=[part[name] for name in ("local","directional","cloud","occupied") if name!=selected]
            x=np.column_stack((part["base"][:,:width],extended[:,8:],block[:,8:],*remaining))
        elif kind in ("local", "flexibility", "directional", "motif", "exposure", "cloud", "charge",
                      "occupied", "stiffness", "torsional", "explicit", "peptide"):
            extra = part[kind][:, :8]
            extended = np.column_stack((part["pair"], extra))
            assert extended.dtype == np.float32 and extended.shape[1] == 16
            local = [part[kind][:, 8:]]
            if kind != "local":
                local.append(part["local"])
            x = np.column_stack((part["base"][:, :width], extended[:, 8:], *local))
        else:
            x = inputs(part["pair"], part["base"], width, kind)
        return np.column_stack((x, part["old"])) if inherit else x
    if representation.startswith("hb-"):
        from .data import inputs
        if representation == "hb-static":
            return inputs(part["pair"], part["base"], width, "environment")
        past = part["history"]
        if representation == "hb-angular":
            extra = part["angles"]
        elif representation == "hb-motion":
            extra = past[:, :8]
        else:
            extra = past[:, 8:]
        extended = np.column_stack((part["pair"], extra))
        assert extended.dtype == np.float32 and extended.shape[1] == 16
        fields = [part["base"][:, :width], extended[:, 8:]]
        if representation == "hb-dynamics":
            fields.append(past[:, :8])
        if representation == "hb-angular":
            fields.append(past)
        if inherit:
            fields.append(part["old"][:, None])
        return np.column_stack(fields)
    if representation in ("kinetics", "kinetics-all"):
        statistics = part["kinetics"]
        core = (0, 1, 2, 3, 8, 9, 10, 11)
        extended = np.column_stack((part["pair"], statistics[:, core]))
        assert extended.dtype == np.float32 and extended.shape[1] == 16
        parts = [part["base"][:, :width], extended[:, 8:]]
        if representation == "kinetics-all":
            parts.extend((statistics[:, 4:8], part["history"]))
        if inherit:
            parts.append(part["old"][:, None])
        return np.column_stack(parts)
    if representation in ("motion", "occupancy", "history-all"):
        past = part["history"]
        motion = (0, 1, 2, 3, 7, 12, 13, 15)
        occupancy = (4, 5, 6, 8, 9, 10, 11, 14)
        columns = motion if representation == "motion" else occupancy
        extended = np.column_stack((part["pair"], past[:, columns]))
        assert extended.dtype == np.float32 and extended.shape[1] == 16
        parts = [part["base"][:, :width], extended[:, 8:]]
        if representation == "history-all":
            parts.append(past[:, motion])
        if inherit:
            parts.append(part["old"][:, None])
        return np.column_stack(parts)
    return feature_view(part["pair"], part["base"], part["old"], representation, width, inherit)


def net(n, hidden):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(n, hidden), nn.SiLU(), nn.Linear(hidden, hidden//2),
                         nn.SiLU(), nn.Linear(hidden//2, 1))


def fit(name, out, train, config):
    import torch
    import xgboost as xgb

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training device is unavailable")
    selector, _, _, _, capacity, weight = config
    x = np.concatenate([view(part, config) for part in train])
    y = np.concatenate([part["y"] for part in train])
    # CPU feature values and a scalar class weight; optimizer updates are CUDA-only.
    torch.manual_seed(27182)
    started = time.perf_counter()
    cuda = "cuda:0"
    path = out / name
    local_weight = np.where(x[:, 0] <= 9.0, 1.0, 0.25).astype(np.float32) if selector.endswith("-local") else None
    if "training_weight" in train[0]:
        weight_by_row = np.concatenate([part["training_weight"] for part in train])
        local_weight = weight_by_row if local_weight is None else weight_by_row*local_weight
    if selector.startswith("xgb"):
        params = dict(device=cuda, tree_method="hist", objective="binary:logistic",
                      max_depth=capacity, eta=0.05, subsample=0.85, colsample_bytree=0.9,
                      min_child_weight=4, scale_pos_weight=weight, seed=27182, nthread=4)
        objective_labels = y
        if "training_soft_label" in train[0]:
            objective_labels = np.concatenate([part["training_soft_label"] for part in train])
            if not np.isfinite(objective_labels).all() or np.any(
                    (objective_labels < -1e-6) | (objective_labels > 1+1e-6)):
                raise ValueError("Training soft labels must lie in [0, 1]")
            objective_labels = np.clip(objective_labels, 0, 1).astype(np.float32)
        rounds = 240
        if selector.startswith("xgb-rounds-"):
            rounds = int(selector.removeprefix("xgb-rounds-"))
            if rounds < 1:
                raise ValueError("Boosting rounds must be positive")
        if "-soft" in selector:
            distances = np.concatenate([part["future_distance"] for part in train])
            bandwidth = .50 if "-wide" in selector else .25
            fraction = .25
            if "-tune-" in selector:
                bandwidth, fraction = map(float, selector.split("-")[3:5])
                rounds = int(selector.split("-")[5])
            proximity = 1/(1+np.exp(np.clip((distances-train[0]["threshold"])/bandwidth, -30, 30)))
            objective_labels = (1-fraction)*y + fraction*proximity
        matrix = xgb.QuantileDMatrix(x, label=objective_labels, weight=local_weight, nthread=4)
        del x
        booster = xgb.Booster(params, [matrix])
        booster.update(matrix, 0)
        actual = json.loads(booster.save_config())["learner"]
        cuda = actual["generic_param"]["device"]
        updater = actual["gradient_booster"]["gbtree_train_param"]["updater_seq"]
        if cuda != "cuda:0" or "gpu_hist" not in updater:
            raise RuntimeError(f"Expected CUDA XGBoost updates, received {cuda} {updater}")
        for step in range(1, rounds):
            booster.update(matrix, step)
            if rounds >= 400 and (step + 1) % 100 == 0:
                print(name, "CUDA boosting", step + 1, "/", rounds, flush=True)
        checkpoint = path.with_suffix(".ubj")
        booster.save_model(checkpoint)
        peak = None
    else:
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(1e-5, x.std(axis=0, dtype=np.float64)).astype(np.float32)
        torch.cuda.reset_peak_memory_stats()
        features = torch.as_tensor((x-mean)/scale, device=cuda)
        target = torch.as_tensor(y, device=cuda, dtype=torch.float32)
        sample_weight = (torch.as_tensor(local_weight, device=cuda)
                         if local_weight is not None else torch.ones_like(target))
        model = net(x.shape[1], capacity).to(cuda)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
        positive_weight = torch.tensor(weight, device=cuda)
        tail = selector.startswith("mlp-tail")
        if tail:
            positives = torch.as_tensor(np.flatnonzero(y), device=cuda)
            negatives = torch.as_tensor(np.flatnonzero(~y & (x[:, 0] <= 10)), device=cuda)
            if not len(positives) or not len(negatives):
                raise ValueError("Tail ranking requires both classes")
            rank_weight = .035 if selector == "mlp-tail" else .1
        for _ in range(12 if tail else 24):
            for batch in torch.randperm(len(y), device=cuda).split(4096):
                optimizer.zero_grad(set_to_none=True)
                logits = model(features[batch]).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target[batch], weight=sample_weight[batch], pos_weight=positive_weight)
                if tail:
                    pos_ids = positives[torch.randint(len(positives), (64,), device=cuda)]
                    neg_ids = negatives[torch.randint(len(negatives), (512,), device=cuda)]
                    low_positives = model(features[pos_ids]).flatten().sort().values[8:24]
                    high_negatives = model(features[neg_ids]).flatten().topk(32).values
                    ranking = torch.nn.functional.softplus(high_negatives[:, None] - low_positives).mean()
                    loss = loss + rank_weight * ranking
                loss.backward()
                optimizer.step()
        torch.cuda.synchronize()
        cuda = str(next(model.parameters()).device)
        peak = torch.cuda.max_memory_allocated()
        updater = "AdamW"
        checkpoint = path.with_suffix(".pt")
        torch.save(dict(state=model.cpu().state_dict(), mean=torch.from_numpy(mean),
                        scale=torch.from_numpy(scale)), checkpoint)
    seconds = time.perf_counter() - started
    memory = psutil.Process().memory_info()
    return dict(checkpoint=checkpoint.name, sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                fit_seconds=seconds, cuda=cuda, updater=updater, train_rows=len(y),
                train_positives=int(y.sum()), cuda_allocated_peak_bytes=peak,
                process_peak_working_set_bytes=getattr(memory, "peak_wset", None))


def cpu_predict(checkpoint, config):
    import torch
    import xgboost as xgb
    if config[0].startswith("xgb"):
        model = xgb.Booster()
        model.load_model(checkpoint)
        model.set_param({"device": "cpu", "nthread": 4})
        predict = model.inplace_predict
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = net(len(state["mean"]), config[4]).eval()
        model.load_state_dict(state["state"])

        @torch.inference_mode()
        def predict(features):
            x = (torch.from_numpy(features) - state["mean"]) / state["scale"]
            return torch.sigmoid(model(x).squeeze(1)).numpy()

    weight = config[-1]

    def predict_adjusted(features):
        p = predict(features).astype(np.float64)
        return p / (weight - (weight - 1) * p)

    return predict_adjusted


def count_to_recall(y, scores, retained):
    positives = int(y.sum())
    target = math.ceil(TARGET_RECALL * positives)
    if target == 0 or int(y[retained].sum()) < target:
        return None
    selected = np.flatnonzero(retained)
    order = selected[np.argsort(-scores[selected], kind="stable")]
    reached = np.searchsorted(np.cumsum(y[order]), target)
    # All candidates tied with the cutoff are counted.
    return int(np.count_nonzero(scores[selected] >= scores[order[reached]]))


def metrics(y, scores):
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return dict(auc=float(roc_auc_score(y, scores)),
                auprc=float(average_precision_score(y, scores)),
                brier=float(brier_score_loss(y, scores)),
                logloss=float(log_loss(y, scores, labels=[0, 1])))


def gate(part, p, screen):
    distance, cap = screen
    keep = np.ones(len(p), dtype=bool) if distance is None else part["base"][:, 0] <= distance
    if cap is None:
        return keep
    for start, end in zip(np.r_[0, part["ends"][:-1]], part["ends"]):
        eligible = np.flatnonzero(keep[start:end]) + start
        if len(eligible) > cap:
            order = eligible[np.argsort(-p[eligible], kind="stable")]
            keep[order[cap:]] = False
    return keep


def bench(parts, scores, screens=SCREENS, keep_override=None):
    y = np.concatenate([part["y"] for part in parts])
    old = np.concatenate([part["old"] for part in parts]).astype(np.float64)
    p = np.concatenate(scores)
    common = old >= 0
    previous = metrics(y[common], old[common])
    candidate = metrics(y[common], p[common])
    if previous is None or candidate is None:
        raise ValueError("Common cohort must contain positives and negatives")
    deltas = {name: (candidate[name] - previous[name]) * direction
              for name, direction in (("auprc", 1), ("auc", 1),
                                      ("brier", -1), ("logloss", -1))}
    needed_old = count_to_recall(y, old, common)
    oracle_review = math.ceil(TARGET_RECALL * int(y.sum()))
    screens_report = []
    for screen in screens:
        keep = (np.concatenate(keep_override) if keep_override is not None else
                np.concatenate([gate(part, prediction, screen)
                                for part, prediction in zip(parts, scores)]))
        needed_new = count_to_recall(y, p, keep)
        reduction = needed_old / needed_new if needed_old is not None and needed_new else None
        screens_report.append(dict(screen=screen, review_at_80=needed_new, reduction=reduction,
                                   retained=int(keep.sum()), retained_positives=int(y[keep].sum()),
                                   passes_work=reduction is not None and reduction >= WORK_REDUCTION))
    return dict(rows=len(y), positives=int(y.sum()), common_rows=int(common.sum()),
                common_positives=int(y[common].sum()), previous=previous, candidate=candidate,
                improvement=deltas, old_review_at_80=needed_old, screens=screens_report,
                oracle_review_at_80=oracle_review,
                oracle_max_reduction=needed_old / oracle_review if needed_old is not None else None,
                passes_accuracy=deltas["auprc"] > 0 and all(value >= 0 for value in deltas.values()))


def evaluate(parts, predict, config):
    start = time.perf_counter()
    scores = [predict(view(part, config)) for part in parts]
    elapsed = time.perf_counter() - start
    report = bench(parts, scores)
    report["cpu_inference_seconds"] = elapsed
    return report


def best_screen(report):
    return max(report["screens"], key=lambda row: row["reduction"] or -1.0)


def test_report(parts, scores, screen, keeps=None):
    if keeps is None:
        keeps = [gate(part, p, screen) for part, p in zip(parts, scores)]
    report = bench(parts, scores, (screen,), keeps)
    report["passed"] = report["passes_accuracy"] and report["screens"][0]["passes_work"]
    proteins = sorted({part["protein"] for part in parts})
    grouped = []
    for name in proteins:
        rows = [(part["y"], part["old"], p, keep)
                for part, p, keep in zip(parts, scores, keeps) if part["protein"] == name]
        grouped.append(tuple(np.concatenate(columns) for columns in zip(*rows)))
    ratios = {}
    draws = []
    for draw in itertools.product(range(len(proteins)), repeat=len(proteins)):
        composition = tuple(sorted(draw))
        if composition not in ratios:
            y, old, new, keep = (np.concatenate(values) for values in
                                 zip(*(grouped[i] for i in composition)))
            a = count_to_recall(y, old, old >= 0)
            b = count_to_recall(y, new, keep)
            ratios[composition] = a/b if a is not None and b else 0.0
        draws.append(ratios[composition])
    report["protein_bootstrap_reduction_95"] = list(map(float, np.quantile(draws, (.025, .975))))
    report["bootstrap_passed"] = report["passed"] and (
        report["protein_bootstrap_reduction_95"][0] >= WORK_REDUCTION)
    return report


def explore(root, out, models, stride, task, dense=False, history=False, kinetics=False,
            extended=False, soft=False, tail=False, refine=False):
    task_out = out / task
    task_out.mkdir(parents=True, exist_ok=True)
    if (task_out / "selection.json").exists():
        raise FileExistsError("Selection already frozen")
    uses_soft_targets = soft or refine
    train_stride = 25 if dense or history or kinetics or extended or uses_soft_targets or tail else stride
    use_history = history or kinetics or extended or uses_soft_targets or tail
    use_kinetics = kinetics or extended or uses_soft_targets or tail
    train = get_data(root, TRAIN, train_stride, models, task, use_history, use_kinetics, uses_soft_targets)
    valid = get_data(root, VALIDATION, stride, models, task, use_history, use_kinetics)
    expanded_training = extended or uses_soft_targets or tail
    if expanded_training:
        assert not set(EXTRA_TRAINING).intersection(TRAIN + VALIDATION + TEST)
        train.extend(get_data(root, EXTRA_TRAINING, 50, models, task, True, True, uses_soft_targets))
    results = {}
    model_start = (81 if task == "salt" else 91) if dense else (21 if task == "salt" else 51)
    initial = DENSE_CONFIGS[task] if dense else CONFIGS
    if history:
        model_start = 151 if task == "salt" else 171
        initial = HISTORY_CONFIGS
    if kinetics:
        model_start = 191 if task == "salt" else 211
        initial = KINETIC_CONFIGS
    if extended:
        model_start = 231 if task == "salt" else 241
        initial = EXTENDED_CONFIGS
    if soft:
        model_start = 311 if task == "salt" else 321
        initial = SOFT_CONFIGS
    if tail:
        model_start = 331 if task == "salt" else 341
        initial = TAIL_CONFIGS
    if refine:
        model_start = 351 if task == "salt" else 381
        initial = REFINE_CONFIGS if task == "salt" else REFINE_CONFIGS[::3]
    configs = list(initial)
    adaptive = not (dense or use_history)
    for index in range(len(initial) + (10 if adaptive else 0)):
        if adaptive and index == len(initial):
            # Additional trials use validation feedback only.
            leaders = sorted(results.values(), key=lambda r:
                             (r["validation"]["passes_accuracy"],
                              best_screen(r["validation"])["reduction"] or -1,
                              r["validation"]["improvement"]["auprc"]), reverse=True)
            for leader in leaders[:5]:
                sel, width, rep, inherit, cap, weight = leader["config"]
                alternatives = (1.0, 1.5) if weight != 1 else (0.75, 1.5)
                configs.extend((sel, width, rep, inherit, cap, w) for w in alternatives)
        name = f"T1I{model_start+index:02}"
        config = configs[index]
        fit_record = fit(name, task_out, train, config)
        predict = cpu_predict(task_out / fit_record["checkpoint"], config)
        report = evaluate(valid, predict, config)
        results[name] = dict(config=config, fit=fit_record, validation=report,
                             training_proteins=TRAIN + EXTRA_TRAINING if expanded_training else TRAIN)
        dump(task_out / "validation.json", results)
        screen = best_screen(report)
        print(task, name, "AUPRC change", report["improvement"]["auprc"], "review reduction",
              screen["reduction"], "accuracy pass", report["passes_accuracy"], flush=True)
        # Complete the initial grid before considering the next wave.
        qualifying = [n for n,r in results.items()
                      if r["validation"]["passes_accuracy"] and best_screen(r["validation"])["passes_work"]]
        if index >= len(initial)-1 and qualifying:
            winner = max(qualifying, key=lambda n: best_screen(results[n]["validation"])["reduction"])
            record = results[winner]
            dump(task_out / "selection.json", dict(name=winner, stride=stride, train_stride=train_stride,
                 history=use_history, kinetics=use_kinetics, task=task,
                 train=TRAIN + EXTRA_TRAINING if expanded_training else TRAIN,
                 extra_train_stride=50 if expanded_training else None, validation=VALIDATION, test=TEST,
                 legacy_sha256={k:v[-1] for k,v in models.items()},
                 screen=best_screen(record["validation"])["screen"], **record))
            print("FROZEN", task, winner, flush=True)
            return
    print("NO_VALIDATION_PASS", task, flush=True)


def final(root, out, models, task):
    task_out = out / task
    if (task_out / "final.json").exists():
        raise FileExistsError("Final evaluation already exists")
    selection = json.loads((task_out / "selection.json").read_text(encoding="utf-8"))
    if selection["task"] != task or selection["test"] != list(TEST):
        raise ValueError("Frozen test split changed")
    if selection["legacy_sha256"] != {k:v[-1] for k,v in models.items()}:
        raise ValueError("Frozen reference changed")
    checkpoint = task_out / selection["fit"]["checkpoint"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection["fit"]["sha256"]:
        raise ValueError("Frozen checkpoint changed")
    predict = cpu_predict(checkpoint, selection["config"])
    parts = get_data(root, TEST, selection["stride"], models, task, selection.get("history", False),
                     selection.get("kinetics", False))
    probabilities = [predict(view(part, selection["config"])) for part in parts]
    report = test_report(parts, probabilities, selection["screen"])
    dump(task_out / "final.json", report)
    print("FINAL", task, selection["name"], report, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("explore", "final"))
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--data", type=Path, default=Path("atlas-workspace"))
    parser.add_argument("--legacy", type=Path, default=Path("runs/legacy"))
    parser.add_argument("--out", type=Path, default=Path("runs/events"))
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--dense", action="store_true")
    schedule.add_argument("--history", action="store_true")
    schedule.add_argument("--kinetics", action="store_true")
    schedule.add_argument("--extended-training", action="store_true")
    schedule.add_argument("--soft-targets", action="store_true")
    schedule.add_argument("--tail-ranking", action="store_true")
    schedule.add_argument("--refine-soft", action="store_true")
    args = parser.parse_args()
    models = load_models(args.legacy)
    if args.mode == "explore":
        explore(args.data, args.out, models, args.stride, args.task, args.dense, args.history,
                args.kinetics, args.extended_training, args.soft_targets, args.tail_ranking,
                args.refine_soft)
    else:
        final(args.data, args.out, models, args.task)


if __name__ == "__main__":
    main()
