"""Select an origin-only blend by validation AP subject to a log-loss guard."""
import argparse
import hashlib
import itertools
import json
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.metrics import average_precision_score, log_loss

from .data import VALIDATION
from .hb_cloud import add_cloud
from .hb_directional import add_directional
from .hb_ensemble_calibration import assemble, members
from .hb_occupied import add_occupied
from .hb_single_structure import OUT, load, predict, validation_report
from .hb_two_step import predict_two_step


SWEEP_SUFFIXES = ("", "_focused", "_focused_bands", "_geometry", "_calibration",
                  "_incremental", "_multi", "_qualified")
GEOMETRY_BUILDERS = {
    "charge": ("hb_charge", "add_charge"),
    "exposure": ("hb_exposure", "add_exposure"),
    "flexibility": ("hb_flexibility", "add_flexibility"),
    "motif": ("hb_motif", "add_motif"),
    "peptide": ("hb_peptide_motion", "add_peptide_motion"),
    "stiffness": ("hb_atom_stiffness", "add_stiffness"),
    "torsional": ("hb_torsional", "add_torsional"),
}


def prepare_archived_geometry(parts, representation):
    kind = representation.removeprefix("hb-origin-")
    features = ("motif", "exposure") if kind == "union" else (kind,)
    for name in features:
        if name not in GEOMETRY_BUILDERS:
            continue
        module, builder = GEOMETRY_BUILDERS[name]
        add = getattr(import_module("." + module, __package__), builder)
        for part in parts:
            if name not in part:
                add(part, 100)


def candidate_roles(parts):
    return np.concatenate([(2 * (part["base"][:, 3] == 1) + (part["base"][:, 5] == 1))
                           for part in parts])


def recipe_weights(recipe, distance, regions, roles):
    if recipe.get("weight_axis") == "distance_role":
        return np.take(recipe["weights"], roles * 3 + regions)
    if recipe.get("weight_axis") == "role":
        return np.take(recipe["weights"], roles)
    if "role" in recipe:
        return np.where(roles == recipe["role"], recipe["weights"][0], 0.)
    if "band" not in recipe:
        return np.take(recipe["weights"], regions)
    start, stop = recipe["band"]
    return np.where((distance >= start) & (distance < stop), recipe["weights"][0], 0.)


def calibrate_group(scores, selection, regions, roles, distance=None):
    axis, group, shift, *operation = selection
    if axis == "distance_window":
        start, stop = group
        mask = (distance >= start) & (distance < stop)
    else:
        partition = roles * 3 + regions if axis == "distance_role" else roles if axis == "role" else regions
        mask = partition == group
    if operation == ["slope"]:
        probability = np.clip(scores, 1e-9, 1 - 1e-9)
        midpoint = np.log(.2 / .8)
        margin = midpoint + shift * (np.log(probability / (1. - probability)) - midpoint)
        calibrated = 1. / (1. + np.exp(-margin))
    else:
        factor = np.exp(shift)
        calibrated = factor * scores / (1. + (factor - 1.) * scores)
    return np.where(mask, calibrated, scores)


def scan_calibration(y, scores, winner, guard, regions, roles, target, rounds, distance=None):
    trials = []
    best_score = winner["auprc"]
    groupings = [("role", group) for group in range(4)] + [("distance", group) for group in range(3)]
    groupings += [("distance_role", group) for group in range(12)]
    if distance is not None:
        windows = ((3.5, 3.75), (3.75, 4.15), (3.85, 4.35), (4.15, 4.6),
                   (4.35, 4.75), (4.6, 5.1), (5.1, 6.1))
        groupings.extend(("distance_window", list(window)) for window in windows)
    offsets = [[axis, group, shift] for (axis, group), shift in
               itertools.product(groupings, (-.20, -.10, -.04, -.02, .02, .04, .10, .20))]
    slopes = [[axis, group, slope, "slope"] for (axis, group), slope in
              itertools.product(groupings, (.88, .94, .97, 1.03, 1.06, 1.12))]
    for _ in range(rounds):
        if best_score >= target:
            break
        anchor, seed = scores, winner
        for selection in itertools.chain(offsets, slopes):
            p = calibrate_group(anchor, selection, regions, roles, distance)
            loss = log_loss(y, p, labels=[0, 1])
            ap = average_precision_score(y, p)
            trial = dict(add_model=None, calibrate=selection, parent=seed, auprc=float(ap),
                         logloss=float(loss), meets_guard=bool(loss <= guard))
            trials.append(trial)
            if loss <= guard and ap > best_score:
                winner, best_score, scores = trial, ap, p
        if winner is seed:
            break
        print("AP_CALIBRATION_ROUND", float(best_score), winner["calibrate"], flush=True)
    return scores, winner, trials


def trial_history(trials):
    anchors = []
    anchor_index = {}
    rows = []
    for trial in trials:
        anchor = trial.get("parent")
        if anchor is None:
            rows.append(trial)
            continue
        identity = id(anchor)
        if identity not in anchor_index:
            anchor_index[identity] = len(anchors)
            anchors.append(anchor)
        row = {key: val for key, val in trial.items() if key != "parent"}
        row["parent_ref"] = anchor_index[identity]
        rows.append(row)
    return rows, anchors


def write_report(path, result):
    temporary = NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8")
    try:
        with temporary:
            json.dump(result, temporary, indent=2, allow_nan=False)
            temporary.write("\n")
        Path(temporary.name).replace(path)
    finally:
        Path(temporary.name).unlink(missing_ok=True)


def save_qualified(result):
    if result.get("reached") and result.get("target") == .500:
        qualified = OUT / "ap_search_validation10_extended_sweep_qualified.json"
        if not qualified.is_file():
            write_report(qualified, result)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validation10", dest="validation10", action="store_true")
    group.add_argument("--legacy-validation4", dest="validation10", action="store_false")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--focused", action="store_true",
                        help="Reuse the saved recipe and screen its components and new nominated models")
    parser.add_argument("--bands", action="store_true",
                        help="Compare overlapping distance windows for the shortlisted mixture components")
    parser.add_argument("--archive-geometry", action="store_true",
                        help="Include archived origin-frame geometry representations")
    parser.add_argument("--force-search", action="store_true",
                        help="Continue optimization after the saved validation target has been met")
    parser.add_argument("--rounds", type=int, default=2)
    parser.set_defaults(validation10=True)
    args = parser.parse_args()
    if args.sweep and not args.extended:
        parser.error("--sweep requires --extended")
    if args.focused and not args.sweep:
        parser.error("--focused requires --sweep")
    if args.bands and not args.focused:
        parser.error("--bands requires --focused")
    if args.archive_geometry and (not args.sweep or args.focused):
        parser.error("--archive-geometry requires the full --sweep")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    panel = VALIDATION
    if args.validation10:
        from .hb_validation10 import panel10
        panel = panel10()
    qualified_path = OUT / "ap_search_validation10_extended_sweep_qualified.json"
    if args.validation10 and args.sweep and not args.force_search and qualified_path.is_file():
        qualified = json.loads(qualified_path.read_text(encoding="utf-8"))
        if (tuple(qualified["validation_proteins"]) == panel and qualified["target"] == .500
                and qualified["best"]["auprc"] >= qualified["target"]
                and qualified["best"]["logloss"] <= qualified["baseline"]["candidate"]["logloss"]):
            print(json.dumps({"metrics": qualified["best"]["validation"]["candidate"],
                              "reached": True, "trials": 0}), flush=True)
            return
    parts = load(panel, 100, geometry=True)
    for part in parts:
        add_cloud(part, 100)
        add_directional(part, 100)
        add_occupied(part, 100)
    components = members("T1I2006")
    parent = [sum(weight * matrix[:, i] for i, (weight, _, _) in enumerate(components))
              for matrix in assemble(parts, components)]
    rows = [len(part["y"]) for part in parts]
    y = np.concatenate([part["y"] for part in parts])
    distance = np.concatenate([part["pair"][:, 2] for part in parts])
    roles = candidate_roles(parts)
    base = np.concatenate(parent).astype(np.float64)
    baseline = validation_report(parts, parent)
    guard = baseline["candidate"]["logloss"]
    regions = np.digitize(distance, [4., 4.75])

    predictions = {}
    stages = ["two_step_stacked", "two_step_near"]
    volume_near = OUT / "two_step_near_volume" / "comparison.json"
    if args.extended and volume_near.is_file():
        stages.append("two_step_near_volume")
    for stage in stages:
        folder = OUT / stage
        record = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
        predictions[record["name"]] = np.concatenate(predict_two_step(
            parts, folder, record["state_fit"], record["conditional_fits"], record["config"]))
    trials = []
    best_score = average_precision_score(y, base)
    best_prob = base
    winner = dict(near_model=None, weights=[0., 0., 0.], auprc=best_score, logloss=guard)
    for near_model in predictions:
        change = predictions["T1I1535"] - base
        change[regions == 0] = (predictions[near_model] - base)[regions == 0]
        for weights in itertools.product((0., .125, .25, .375, .5, .75, 1.),
                                         (0., .125, .25, .5), (0., .125, .25, .5)):
            p = base + np.take(weights, regions) * change
            loss = log_loss(y, p, labels=[0, 1])
            ap = average_precision_score(y, p)
            trial = dict(near_model=near_model, weights=list(weights), auprc=float(ap),
                         logloss=float(loss), meets_guard=bool(loss <= guard))
            trials.append(trial)
            if loss <= guard and ap > best_score:
                winner, best_score, best_prob = trial, ap, p
    if args.extended:
        candidates = (("stacked", "T1I1271"), ("stacked_volume", "T1I1281"),
                      ("stacked_volume_96", "T1I1301"), ("stacked_volume_128", "T1I1441"),
                      ("volume_96", "T1I1132"), ("volume_128", "T1I1392"),
                      ("transition_volume", "T1I1343"), ("distilled_trajectory_1463", "T1I1463"))
        candidates += tuple((f"ap10_train_{n}", f"T1I{n}") for n in range(1551, 1554)
                            if (OUT / f"ap10_train_{n}" / "selection.json").is_file())
        candidates += tuple((f"ap10_window_{n}", f"T1I{n}")
                            for n in (1651, 1652, 1653, 1661, 1662, 1663)
                            if (OUT / f"ap10_window_{n}" / "selection.json").is_file())
        explicit_numbers = (1581, 1582, 1671, 1672, 1681, 1682, 1691, 1692, 1701, 1702,
                            1731, 1732, 1761, 1762, 1763, 1764, 1781, 1782, 1783,
                            1791, 1792, 1801, 1802, 1851)
        explicit = tuple((f"ap10_train_{n}", f"T1I{n}") for n in explicit_numbers
                         if (OUT / f"ap10_train_{n}" / "selection.json").is_file())
        if explicit:
            from .hb_explicit_context import add_explicit_context
            from .hb_explicit_hydrogen import add_explicit
            for part in parts:
                add_explicit(part, 100)
                add_explicit_context(part)
            candidates += explicit
        for stage in ("ap10_portfolio", "ap10_portfolio_expanded", "ap10_portfolio_deep"):
            portfolio_path = OUT / stage / "validation.json"
            if portfolio_path.is_file():
                candidates += tuple((stage, name)
                                    for name in json.loads(portfolio_path.read_text(encoding="utf-8")))
        values = {}
        for stage, name in candidates:
            folder = OUT / stage
            record = json.loads((folder / "validation.json").read_text(encoding="utf-8"))[name]
            values[name] = np.concatenate(predict(parts, folder, record)).astype(np.float64)
        for stage in ("ap10_residual", "ap10_residual_context", "ap10_residual_incumbent",
                      "ap10_residual_relational", "ap10_residual_deep"):
            folder = OUT / stage
            if (folder / "validation.json").is_file():
                from .hb_ap_residual import cpu_scores
                for name, record in json.loads((folder / "validation.json").read_text(encoding="utf-8")).items():
                    values[name] = np.concatenate(cpu_scores(parts, folder, record)).astype(np.float64)
        for stage in ("ap10_catboost", "ap10_catboost_rank"):
            folder = OUT / stage
            if (folder / "validation.json").is_file():
                from .hb_ap_catboost import cpu_scores
                for name, record in json.loads((folder / "validation.json").read_text(encoding="utf-8")).items():
                    values[name] = np.concatenate(cpu_scores(parts, folder, record)).astype(np.float64)
        if args.sweep:
            supported = {"environment", "competition", "interaction", "local", "directional", "cloud", "occupied",
                         "stacked-directional", "stacked-occupied", "stacked-cloud"}
            if "explicit" in parts[0]:
                supported.add("explicit")
            supported.update("stacked-" + name for name in ("explicit", "explicit_context") if name in parts[0])
            archived_geometry = {*GEOMETRY_BUILDERS, "union", "context"}
            direct = {"xgb", "xgb-soft-local", "xgb-soft-wide-local", "mlp", "mlp-tail",
                      "hb-transition", "hb-distribution", "hb-gnn", "hb-experts", "xgb-rank"}
            shortlist = []
            old = None
            required = set()
            if args.validation10:
                previous = [OUT / f"ap_search_validation10_extended_sweep{suffix}.json"
                            for suffix in SWEEP_SUFFIXES]
                saved = [json.loads(path.read_text(encoding="utf-8")) for path in previous if path.is_file()]
                if any(tuple(report["validation_proteins"]) != panel for report in saved):
                    raise ValueError("Previous sweep used a different panel")
                old = max(saved, key=lambda report: report["best"]["auprc"], default=None)
            if old is not None:
                recipe = old["best"]
                while "add_model" in recipe:
                    if recipe["add_model"] is not None:
                        required.add(recipe["add_model"])
                    recipe = recipe["parent"]
            nominated = {"ap10_graph_expanded", "ap10_graph_weighted", "ap10_distribution_expanded",
                         "ap10_distribution_explicit", "ap10_rank_expanded", "ap10_experts_expanded",
                         "ap10_graph_motion", "ap10_experts_long"}
            archive = {}
            for path in sorted(OUT.glob("*/validation.json")):
                records = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(records, dict):
                    continue
                for name, record in records.items():
                    if name in values or name in archive or not isinstance(record, dict) or not {"config", "fit"} <= record.keys():
                        continue
                    if args.focused and name not in required and path.parent.name not in nominated:
                        continue
                    config = record["config"]
                    if not isinstance(config, (list, tuple)) or len(config) < 3:
                        continue
                    algo, _, representation, *_ = config
                    kind = representation.removeprefix("hb-origin-")
                    if (algo not in direct and not algo.startswith("xgb-soft-tune-")) or kind == representation:
                        continue
                    if kind not in supported:
                        if kind not in archived_geometry or not (args.archive_geometry or name in required):
                            continue
                        prepare_archived_geometry(parts, representation)
                        supported.add(kind)
                        print("AP_ARCHIVE_GEOMETRY", kind, flush=True)
                    archive[name] = (path.parent, record)

            def score_checkpoint(folder, record):
                fit = record["fit"]
                checkpoint = folder / fit["checkpoint"]
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != fit["sha256"]:
                    raise ValueError(f"Checkpoint changed: {checkpoint}")
                return np.concatenate(predict(parts, folder, record)).astype(np.float64)

            for name in required - values.keys():
                if name not in archive:
                    raise ValueError(f"Previous mixture cannot find checkpoint {name}")
                values[name] = score_checkpoint(*archive[name])
            if old is not None:
                def restore(recipe):
                    if "add_model" in recipe:
                        anchor = restore(recipe["parent"])
                        if "calibrate" in recipe:
                            return calibrate_group(anchor, recipe["calibrate"], regions, roles, distance)
                        selected = recipe["add_model"]
                        return anchor + recipe_weights(recipe, distance, regions, roles) * (values[selected] - anchor)
                    if not recipe["near_model"]:
                        return base
                    change = predictions["T1I1535"] - base
                    change[regions == 0] = (predictions[recipe["near_model"]] - base)[regions == 0]
                    return base + np.take(recipe["weights"], regions) * change

                restored = restore(old["best"])
                ap = average_precision_score(y, restored)
                loss = log_loss(y, restored, labels=[0, 1])
                if loss <= guard and ap > best_score:
                    winner = {key: value for key, value in old["best"].items() if key != "validation"}
                    best_score, best_prob = ap, restored
            screen_weights = ((.125,)*3, (.25,)*3, (.25,0.,0.), (0.,.25,0.), (0.,0.,.25))
            for name, reference in archive.items():
                score = values[name] if name in values else score_checkpoint(*reference)
                best_screen = -1.
                for weights in screen_weights:
                    p = best_prob + np.take(weights, regions) * (score - best_prob)
                    loss = log_loss(y, p, labels=[0, 1])
                    ap = average_precision_score(y, p)
                    trials.append(dict(screen_model=name, weights=weights, auprc=float(ap), logloss=float(loss),
                                       meets_guard=bool(loss <= guard)))
                    if loss <= guard:
                        best_screen = max(ap, best_screen)
                if best_screen >= 0.:
                    shortlist.append((best_screen, name, score))
                    shortlist.sort(key=lambda item: item[0], reverse=True)
                    del shortlist[16:]
            values.update({name: score for _, name, score in shortlist})
            print("AP_ARCHIVE_SCREEN", len(archive), "archive models,", len(values), "blend candidates", flush=True)
        for _ in range(args.rounds):
            anchor, seed = best_prob, winner
            for name, extra in values.items():
                for weights in ((.03125,)*3, (.0625,)*3, (.125,)*3, (.25,)*3, (.5,)*3,
                                (.25, 0., 0.), (0., .25, 0.), (0., 0., .25),
                                (.25, .125, .5), (.5, .125, .25)):
                    p = anchor + np.take(weights, regions) * (extra - anchor)
                    loss = log_loss(y, p, labels=[0, 1])
                    ap = average_precision_score(y, p)
                    trial = dict(add_model=name, weights=list(weights), parent=seed, auprc=float(ap),
                                 logloss=float(loss), meets_guard=bool(loss <= guard))
                    trials.append(trial)
                    if loss <= guard and ap > best_score:
                        winner, best_score, best_prob = trial, ap, p
            if winner is seed or best_score >= .500:
                break
            print("AP_ROUND", float(best_score), winner["add_model"], winner["weights"], flush=True)
        if args.bands and best_score < .500:
            ordered_trials = sorted((trial for trial in trials if "add_model" in trial and trial["meets_guard"]),
                                    key=lambda trial: trial["auprc"], reverse=True)
            preferred = list(dict.fromkeys(trial["add_model"] for trial in ordered_trials))[:16]
            windows = ((3.5, 3.8), (3.5, 4.2), (3.75, 4.15), (3.8, 4.5), (4.0, 4.45),
                       (4.2, 4.75), (4.45, 5.0), (4.6, 6.1), (4.75, 5.3), (5.0, 6.1))
            for _ in range(4):
                anchor, seed = best_prob, winner
                for name in preferred:
                    extra = values[name] - anchor
                    for (start, stop), weight in itertools.product(windows, (.0625, .125, .25)):
                        p = anchor + np.where((distance >= start) & (distance < stop), extra * weight, 0.)
                        loss = log_loss(y, p, labels=[0, 1])
                        ap = average_precision_score(y, p)
                        trial = dict(add_model=name, weights=[weight], band=[start, stop], parent=seed,
                                     auprc=float(ap), logloss=float(loss), meets_guard=bool(loss <= guard))
                        trials.append(trial)
                        if loss <= guard and ap > best_score:
                            winner, best_score, best_prob = trial, ap, p
                if winner is seed:
                    break
                print("AP_BAND_ROUND", float(best_score), winner["add_model"], winner["band"], flush=True)
                if best_score >= .500:
                    break
            for _ in range(3):
                if best_score >= .500:
                    break
                anchor, seed = best_prob, winner
                for name in preferred:
                    extra = values[name] - anchor
                    for role, weight in itertools.product(range(4), (.0625, .125, .25)):
                        p = anchor + np.where(roles == role, weight * extra, 0.)
                        loss = log_loss(y, p, labels=[0, 1])
                        ap = average_precision_score(y, p)
                        trial = dict(add_model=name, weights=[weight], role=role, parent=seed,
                                     auprc=float(ap), logloss=float(loss), meets_guard=bool(loss <= guard))
                        trials.append(trial)
                        if loss <= guard and ap > best_score:
                            winner, best_score, best_prob = trial, ap, p
                if winner is seed:
                    break
                print("AP_ROLE_ROUND", float(best_score), winner["add_model"], winner["role"], flush=True)
            best_prob, winner, calibration_trials = scan_calibration(
                y, best_prob, winner, guard, regions, roles, .500, 4, distance)
            trials.extend(calibration_trials)
            best_score = winner["auprc"]
    scores = np.split(best_prob, np.cumsum(rows)[:-1])
    per_protein = {}
    for protein in panel:
        selected = [i for i, part in enumerate(parts) if part["protein"] == protein]
        truth = np.concatenate([parts[i]["y"] for i in selected])
        per_protein[protein] = {
            "parent_auprc": float(average_precision_score(truth, np.concatenate([parent[i] for i in selected]))),
            "winner_auprc": float(average_precision_score(truth, np.concatenate([scores[i] for i in selected]))),
        }
    trial_rows, anchors = trial_history(trials)
    result = dict(parent="T1I2006", outer_model="T1I1535", distance_edges=[4., 4.75],
                  baseline=baseline, best=dict(winner, validation=validation_report(parts, scores)),
                  per_protein=per_protein, trials=trial_rows, trial_anchors=anchors)
    name = "ap_search_validation10.json" if args.validation10 else "ap_search.json"
    if args.validation10:
        target = .500 if args.extended else .480
        result.update(validation_proteins=panel, target=target, reached=bool(best_score >= target))
    if args.extended:
        name = name.replace(".json", "_extended.json")
    if args.sweep:
        name = name.replace(".json", "_sweep.json")
    if args.focused:
        name = name.replace(".json", "_focused.json")
    if args.bands:
        name = name.replace(".json", "_bands.json")
    if args.archive_geometry:
        name = name.replace(".json", "_geometry.json")
    write_report(OUT / name, result)
    save_qualified(result)
    print(json.dumps({"metrics": result["best"]["validation"]["candidate"],
                      "reached": result.get("reached"), "trials": len(trial_rows)}), flush=True)


if __name__ == "__main__":
    main()
