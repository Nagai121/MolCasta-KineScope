"""Replay a saved AP mixture on the CPU and optionally evaluate a frozen model."""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, log_loss

from .data import TEST
from .hb_ap_search import (SWEEP_SUFFIXES, calibrate_group, candidate_roles,
                           prepare_archived_geometry, recipe_weights, save_qualified, scan_calibration,
                           trial_history, write_report)
from .hb_cloud import add_cloud
from .hb_directional import add_directional
from .hb_ensemble_calibration import assemble, members
from .hb_explicit_context import add_explicit_context
from .hb_explicit_hydrogen import add_explicit
from .hb_occupied import add_occupied
from .hb_single_structure import OUT, load, predict, validation_report
from .hb_two_step import predict_two_step
from .hb_validation10 import panel10
from .search_events import dump


def checkpoints():
    found = {}
    for path in sorted(OUT.glob("*/validation.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, dict):
            continue
        for name, row in records.items():
            if isinstance(row, dict) and isinstance(row.get("fit"), dict) and "checkpoint" in row["fit"]:
                found.setdefault(name, (path.parent, row))
    return found


def score_checkpoint(parts, folder, row):
    fit = row["fit"]
    model_path = folder / fit["checkpoint"]
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != fit["sha256"]:
        raise ValueError(f"Checkpoint changed: {model_path}")
    if folder.name.startswith("ap10_residual"):
        from .hb_ap_residual import cpu_scores
        return cpu_scores(parts, folder, row)
    if folder.name.startswith("ap10_catboost"):
        from .hb_ap_catboost import cpu_scores
        return cpu_scores(parts, folder, row)
    prepare_archived_geometry(parts, row["config"][2])
    return predict(parts, folder, row)


def replay(parts, report):
    components = members(report["parent"])
    parent = [sum(weight * matrix[:, i] for i, (weight, _, _) in enumerate(components))
              for matrix in assemble(parts, components)]
    base = np.concatenate(parent).astype(np.float64)
    distance = np.concatenate([part["pair"][:, 2] for part in parts])
    roles = candidate_roles(parts)
    regions = np.digitize(distance, report["distance_edges"])
    models = checkpoints()
    two_step = {}
    for stage in ("two_step_stacked", "two_step_near", "two_step_near_volume"):
        folder = OUT / stage
        path = folder / "comparison.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            two_step[record["name"]] = (folder, record)
    scores = {}

    def score(name):
        if name not in scores:
            if name in two_step:
                folder, row = two_step[name]
                values = predict_two_step(parts, folder, row["state_fit"], row["conditional_fits"], row["config"])
            else:
                folder, row = models[name]
                values = score_checkpoint(parts, folder, row)
            scores[name] = np.concatenate(values).astype(np.float64)
        return scores[name]

    recipe = report["best"]
    steps = []
    while "add_model" in recipe:
        steps.append(recipe)
        recipe = recipe["parent"]
    output = base
    if recipe["near_model"]:
        change = score(report["outer_model"]) - base
        first = regions == 0
        change[first] = (score(recipe["near_model"]) - base)[first]
        output = base + np.take(recipe["weights"], regions) * change
    for step in reversed(steps):
        if "calibrate" in step:
            output = calibrate_group(output, step["calibrate"], regions, roles, distance)
        else:
            output += recipe_weights(step, distance, regions, roles) * (score(step["add_model"]) - output)
    breaks = np.cumsum([len(part["y"]) for part in parts])[:-1]
    return parent, np.split(output, breaks)


def scan_new_model(parts, name, anchor, seed, truth, regions, roles, guard):
    folder, row = checkpoints()[name]
    difference = np.concatenate(score_checkpoint(parts, folder, row)).astype(np.float64) - anchor
    trials = []
    winner, scores = seed, anchor
    grids = (("distance", regions, (0., .0625, .125, .25, .375, .5), 3),
             ("role", roles, (0., .0625, .125, .25, .5), 4))
    for axis, partition, weights_grid, group_count in grids:
        for weights in itertools.product(weights_grid, repeat=group_count):
            if not any(weights):
                continue
            p = anchor + np.take(weights, partition) * difference
            loss = log_loss(truth, p, labels=[0, 1])
            ap = average_precision_score(truth, p)
            trial = dict(add_model=name, weights=weights, weight_axis=axis, parent=seed, auprc=float(ap),
                         logloss=float(loss), meets_guard=bool(loss <= guard))
            trials.append(trial)
            if loss <= guard and ap > winner["auprc"]:
                winner, scores = trial, p
    partition = roles * 3 + regions
    surface = np.zeros((4, 3))
    if winner is not seed:
        weights = np.asarray(winner["weights"])
        surface[:] = weights[:, None] if winner["weight_axis"] == "role" else weights[None, :]
    weights_grid = (0., .0625, .125, .25, .375, .5)
    for _ in range(3):
        old_winner = winner
        reference = surface.ravel()
        for cell, weight in itertools.product(range(12), weights_grid):
            if weight == reference[cell]:
                continue
            weights = reference.copy()
            weights[cell] = weight
            p = anchor + weights[partition] * difference
            loss = log_loss(truth, p, labels=[0, 1])
            ap = average_precision_score(truth, p)
            trial = dict(add_model=name, weights=weights.tolist(), weight_axis="distance_role",
                         parent=seed, auprc=float(ap), logloss=float(loss), meets_guard=bool(loss <= guard))
            trials.append(trial)
            if loss <= guard and ap > winner["auprc"]:
                winner, scores = trial, p
        if winner is old_winner:
            break
        surface = np.asarray(winner["weights"]).reshape(4, 3)
    return scores, winner, trials


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--test", action="store_true")
    modes.add_argument("--calibrate", action="store_true")
    modes.add_argument("--blend-model", action="append")
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()
    fixed = panel10()
    paths = [args.recipe] if args.recipe else [
        OUT / f"ap_search_validation10_extended_sweep{suffix}.json" for suffix in SWEEP_SUFFIXES]
    available = []
    for path in paths:
        if path.is_file():
            data = path.read_bytes()
            available.append((path, json.loads(data), hashlib.sha256(data).hexdigest()))
    path, report, digest = max(available, key=lambda item: item[1]["best"]["auprc"])
    if tuple(report["validation_proteins"]) != fixed:
        raise ValueError("Saved recipe used a different validation panel")
    if args.test and not report["reached"]:
        raise ValueError("The recipe has not reached its fixed validation target")
    destination = OUT / ("ap10_final_test.json" if args.test else "ap10_validation_replay.json")
    if args.test and destination.is_file():
        if json.loads(destination.read_text(encoding="utf-8"))["recipe_sha256"] != digest:
            raise ValueError("A different recipe has already used the final test panel")
    panels = TEST if args.test else fixed
    start = time.perf_counter()
    parts = load(panels, 100, geometry=True)
    for part in parts:
        add_cloud(part, 100)
        add_directional(part, 100)
        add_occupied(part, 100)
        add_explicit(part, 100)
        add_explicit_context(part)
    start_inference = time.perf_counter()
    baseline, proposed = replay(parts, report)
    runtime = time.perf_counter() - start_inference
    measured = validation_report(parts, proposed)
    if not args.test and not np.isclose(measured["candidate"]["auprc"], report["best"]["auprc"], atol=1e-12, rtol=0):
        raise ValueError("Replayed validation score differs from the selected recipe")
    output = dict(recipe=path.name, recipe_sha256=digest,
                  proteins=panels, baseline=validation_report(parts, baseline), selected=measured,
                  cpu_preparation_seconds=start_inference - start, cpu_scoring_seconds=runtime)
    dump(destination, output)
    print(json.dumps(dict(metrics=measured["candidate"], cpu_scoring_seconds=runtime)), flush=True)
    if args.calibrate or args.blend_model:
        truth = np.concatenate([part["y"] for part in parts])
        distance = np.concatenate([part["pair"][:, 2] for part in parts])
        regions = np.digitize(distance, report["distance_edges"])
        roles = candidate_roles(parts)
        guard = report["baseline"]["candidate"]["logloss"]
        seed = {key: value for key, value in report["best"].items() if key != "validation"}
        scores = np.concatenate(proposed)
        winner, trials = seed, []
        for model_name in args.blend_model or ():
            if winner["auprc"] >= report["target"]:
                break
            scores, winner, model_trials = scan_new_model(
                parts, model_name, scores, winner, truth, regions, roles, guard)
            trials.extend(model_trials)
            print("AP_MODEL", model_name, winner["auprc"], flush=True)
        scores, winner, calibration_trials = scan_calibration(
            truth, scores, winner, guard, regions, roles, report["target"], 8, distance)
        trials.extend(calibration_trials)
        parts_scores = np.split(scores, np.cumsum([len(part["y"]) for part in parts])[:-1])
        per_protein = {}
        for protein in panels:
            selected = [i for i, part in enumerate(parts) if part["protein"] == protein]
            labels = np.concatenate([parts[i]["y"] for i in selected])
            per_protein[protein] = dict(
                parent_auprc=float(average_precision_score(labels, np.concatenate([baseline[i] for i in selected]))),
                winner_auprc=float(average_precision_score(labels, np.concatenate([parts_scores[i] for i in selected]))))
        metrics = validation_report(parts, parts_scores)
        trial_rows, anchors = trial_history(trials)
        refined = dict(report, best=dict(winner, validation=metrics), trials=trial_rows, trial_anchors=anchors,
                       per_protein=per_protein, reached=winner["auprc"] >= report["target"])
        suffix = "multi" if len(args.blend_model or ()) > 1 else "incremental" if args.blend_model else "calibration"
        write_report(OUT / f"ap_search_validation10_extended_sweep_{suffix}.json", refined)
        save_qualified(refined)
        print(json.dumps(dict(metrics=metrics["candidate"], reached=refined["reached"])), flush=True)


if __name__ == "__main__":
    main()
