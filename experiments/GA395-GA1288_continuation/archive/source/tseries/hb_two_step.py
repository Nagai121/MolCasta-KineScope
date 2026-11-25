"""Marginalize a predicted intermediate state for 1 ns HB formation."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xgboost as xgb

from .data import TRAIN, VALIDATION
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, load, validation_report
from .hb_transition import auxiliary_labels
from .search_events import EXTRA_TRAINING, dump, view

CONFIG = ("xgb", 118, "hb-origin-environment", False, 6, 1.)


def train(folder, name, x, y, classes=2):
    matrix = xgb.QuantileDMatrix(x, label=y, nthread=4)
    params = dict(device="cuda:0", tree_method="hist", max_depth=6, eta=.05,
                  subsample=.85, colsample_bytree=.9, min_child_weight=4, seed=27182, nthread=4)
    params.update(objective="binary:logistic" if classes == 2 else "multi:softprob")
    if classes > 2:
        params["num_class"] = classes
    start = time.perf_counter()
    model = xgb.train(params, matrix, num_boost_round=240)
    seconds = time.perf_counter() - start
    learner = json.loads(model.save_config())["learner"]
    device = learner["generic_param"]["device"]
    updater = learner["gradient_booster"]["gbtree_train_param"]["updater_seq"]
    if device != "cuda:0" or "gpu_hist" not in updater:
        raise RuntimeError("CUDA histogram training was not used")
    path = folder / f"{name}.ubj"
    model.save_model(path)
    result = dict(checkpoint=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  fit_seconds=seconds, train_rows=len(y), cuda=device, updater=updater)
    return result


def scorer(folder, fit):
    path = folder / fit["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != fit["sha256"]:
        raise ValueError("Checkpoint checksum differs")
    model = xgb.Booster()
    model.load_model(path)
    model.set_param({"device": "cpu", "nthread": 4})
    return model.inplace_predict


def predict_two_step(parts, folder, state_fit, conditionals, config=CONFIG):
    state_score = scorer(folder, state_fit)
    endpoint_scores = [scorer(folder, fit) for fit in conditionals]
    scores = []
    for part in parts:
        features = view(part, config)
        weights = state_score(features)
        endpoint = np.column_stack([score(features) for score in endpoint_scores])
        scores.append((weights * endpoint).sum(axis=1))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacked", action="store_true")
    parser.add_argument("--near-expert", action="store_true")
    parser.add_argument("--volume", action="store_true")
    args = parser.parse_args()
    if args.near_expert and not args.stacked:
        parser.error("--near-expert requires --stacked")
    if args.volume and not args.near_expert:
        parser.error("--volume requires --near-expert")
    folder = OUT / ("two_step_near_volume" if args.volume else "two_step_near" if args.near_expert
                    else "two_step_stacked" if args.stacked else "two_step")
    folder.mkdir(exist_ok=True)
    middle = EXTRA_TRAINING + ADDITIONAL_TRAINING
    early, late = load(TRAIN, 25, geometry=args.stacked), load(middle, 50, geometry=args.stacked)
    groups = [(early, 25), (late, 50)]
    allowed = TRAIN + middle
    config = CONFIG
    panel = VALIDATION
    if args.volume:
        from .hb_validation10 import panel10
        panel = panel10()
    valid = load(panel, 100, geometry=args.stacked)
    start = 1521
    if args.stacked:
        from .hb_cloud import add_cloud
        from .hb_directional import add_directional
        from .hb_occupied import add_occupied

        extra = tuple(json.loads((OUT / "diverse_training_panel.json").read_text(encoding="utf-8"))["proteins"])
        if args.volume:
            extra += tuple(json.loads((OUT / "volume_training_panel.json").read_text(encoding="utf-8"))["proteins"])
        allowed += extra
        if set(allowed) & set(panel):
            raise ValueError("Validation overlaps training")
        groups.append((load(extra, 100, geometry=True), 100))
        config = ("xgb", 118, "hb-origin-stacked-directional", False, 6, 1.)
        start = 1531
        for rows, stride in [*groups, (valid, 100)]:
            for part in rows:
                add_cloud(part, stride)
                add_directional(part, stride)
                add_occupied(part, stride)
    with ThreadPoolExecutor(3) as executor:
        jobs = [executor.submit(auxiliary_labels, part, stride, allowed)
                for parts, stride in groups for part in parts]
        for job in jobs:
            job.result()
    train_rows = [part for rows, _ in groups for part in rows]
    x = np.concatenate([view(part, config) for part in train_rows])
    y = np.concatenate([part["y"] for part in train_rows])
    middle_distance = np.concatenate([part["aux"][:, 2] for part in train_rows])
    if args.near_expert:
        start = 1631 if args.volume else 1541
        eligible = np.concatenate([part["pair"][:, 2] < 4. for part in train_rows])
        x, y, middle_distance = x[eligible], y[eligible], middle_distance[eligible]
    boundaries = [3.5, 4.25]
    state = np.searchsorted(boundaries, middle_distance, side="left")
    state_fit = train(folder, f"T1I{start}", x, state, classes=3)
    conditionals = []
    for index in range(3):
        keep = state == index
        conditionals.append(train(folder, f"T1I{start + 1 + index}", x[keep], y[keep]))
    scores = predict_two_step(valid, folder, state_fit, conditionals, config)
    if args.near_expert:
        scores = [np.where(part["pair"][:, 2] < 4., score, part["old"])
                  for part, score in zip(valid, scores)]
    report = validation_report(valid, scores)
    dump(folder / "comparison.json", dict(name=f"T1I{start + 4}", config=config, intermediate_ns=.5, endpoint_ns=1.,
                                            validation_proteins=panel,
                                            boundaries=boundaries, state_fit=state_fit,
                                            conditional_fits=conditionals, validation=report))
    print(report["candidate"], report["screens"][0]["review_at_80"], flush=True)


if __name__ == "__main__":
    main()
