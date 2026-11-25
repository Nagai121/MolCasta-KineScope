"""Compare symmetric-tree GPU boosting on the fixed ten-protein panel."""
import argparse
import hashlib
import json
import time

import catboost
import numpy as np

from .data import TRAIN
from .hb_cloud import add_cloud
from .hb_directional import add_directional
from .hb_occupied import add_occupied
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, load, validation_report
from .hb_validation10 import panel10
from .search_events import EXTRA_TRAINING, dump, view


def cpu_scores(parts, folder, record):
    path = folder / record["fit"]["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["fit"]["sha256"]:
        raise ValueError(f"Checkpoint changed: {path}")
    model = catboost.CatBoostClassifier()
    if "calibration" in record["fit"]:
        model = catboost.CatBoost()
        model.load_model(path)
        scale, bias = record["fit"]["calibration"]
        return [1 / (1 + np.exp(-np.clip(scale * model.predict(view(p, record["config"]),
                                                               task_type="CPU", thread_count=4) + bias, -40, 40)))
                for p in parts]
    model.load_model(path)
    return [model.predict_proba(view(p, record["config"]), task_type="CPU", thread_count=4)[:, 1]
            for p in parts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", action="store_true")
    args = parser.parse_args()
    panel = panel10()
    extra = tuple(json.loads((OUT / "diverse_training_panel.json").read_text(encoding="utf-8"))["proteins"])
    middle = EXTRA_TRAINING + ADDITIONAL_TRAINING
    if set(panel) & set(TRAIN + middle + extra):
        raise ValueError("Validation overlaps training")
    groups = [(load(names, stride, geometry=True), stride)
              for names, stride in ((TRAIN, 25), (middle, 50), (extra, 100))]
    valid = load(panel, 100, geometry=True)
    calibration = []
    if args.rank:
        names = tuple(json.loads((OUT / "ap10_calibration_panel.json").read_text(encoding="utf-8"))["proteins"])
        if set(names) & set(panel + TRAIN + middle + extra):
            raise ValueError("Rank calibration overlaps model fitting or validation")
        calibration = load(names, 100, geometry=True)
    for parts, stride in [*groups, (valid, 100), (calibration, 100)]:
        for part in parts:
            add_cloud(part, stride)
            add_directional(part, stride)
            add_occupied(part, stride)
    train = [part for parts, _ in groups for part in parts]
    labels = np.concatenate([part["y"] for part in train])
    folder = OUT / ("ap10_catboost_rank" if args.rank else "ap10_catboost")
    folder.mkdir(exist_ok=True)
    results = {}
    if args.rank:
        sizes = np.concatenate([np.diff(np.r_[0, p["ends"]]) for p in train])
        group_ids = np.repeat(np.arange(len(sizes)), sizes)
    for index, (width, depth, iterations) in enumerate(((118, 6, 450), (166, 8, 600)), 1641 if args.rank else 1601):
        name = f"T1I{index}"
        config = ("catboost", width, "hb-origin-stacked-directional", False, depth, 1.)
        features = np.concatenate([view(part, config) for part in train])
        model_type = catboost.CatBoostRanker if args.rank else catboost.CatBoostClassifier
        objective = "PairLogit:max_pairs=512" if args.rank else "Logloss"
        model = model_type(task_type="GPU", devices="0", iterations=iterations, learning_rate=.045,
                           depth=depth, loss_function=objective, thread_count=4, verbose=150,
                           gpu_ram_part=.7, allow_writing_files=False, random_seed=27182)
        started = time.perf_counter()
        if args.rank:
            model.fit(catboost.Pool(features, labels, group_id=group_ids))
        else:
            model.fit(features, labels)
        fit_seconds = time.perf_counter() - started
        if model.get_all_params().get("task_type") != "GPU":
            raise RuntimeError("CatBoost did not report GPU training")
        path = folder / f"{name}.cbm"
        model.save_model(path)
        record = dict(config=config, fit=dict(checkpoint=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                              cuda="cuda:0", updater="CatBoost GPU", fit_seconds=fit_seconds,
                                              train_rows=len(labels), objective=objective, catboost=catboost.__version__))
        if args.rank:
            import torch

            cpu = catboost.CatBoost()
            cpu.load_model(path)
            raw = np.concatenate([cpu.predict(view(p, config), task_type="CPU", thread_count=4)
                                  for p in calibration])
            xx = torch.as_tensor(raw, dtype=torch.float32, device="cuda")
            yy = torch.as_tensor(np.concatenate([p["y"] for p in calibration]), dtype=torch.float32, device="cuda")
            parameters = torch.nn.Parameter(torch.tensor([0., -2.], device="cuda"))
            optimizer = torch.optim.Adam([parameters], lr=.03)
            for _ in range(360):
                optimizer.zero_grad()
                logits = parameters[0].exp() * xx + parameters[1]
                torch.nn.functional.binary_cross_entropy_with_logits(logits, yy).backward()
                optimizer.step()
            record["fit"]["calibration"] = [parameters[0].exp().item(), parameters[1].item()]
            record["fit"]["calibration_proteins"] = names
        record["validation"] = validation_report(valid, cpu_scores(valid, folder, record))
        results[name] = record
        dump(folder / "validation.json", results)
        print(name, record["validation"]["candidate"], flush=True)


if __name__ == "__main__":
    main()
