"""Fixed-checkpoint horizon audit on identical 1mix_A origins and labels."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .data import REPLICAS, cell, frame, inputs, registry, topology, trajectory
from .event_kinetics import distances
from .features import _mic
from .legacy import build_layout, load_models
from .models import load_cpu
from .search_events import cpu_predict, get_data, view
from .trial38_features import historical_hb_features

PROTEIN = "1mix_A"
HORIZONS = (1, 5, 10, 30, 50, 90)
STRIDE = 25
TRIAL_BLOB = "a3192700777088a2a97ecfd1caa970fa2b9dad25"


class Trial38Unpickler(pickle.Unpickler):
    ALLOWED = {
        "numpy": {"ndarray", "dtype"},
        "numpy._core.multiarray": {"scalar", "_reconstruct"},
        "numpy.random._pickle": {"__generator_ctor", "__bit_generator_ctor"},
        "numpy.random._pcg64": {"PCG64"},
        "numpy.random.bit_generator": {"__pyx_unpickle_SeedSequence", "SeedSequence"},
        "sklearn.preprocessing._label": {"LabelEncoder"},
        "sklearn._loss.loss": {"HalfBinomialLoss"},
        "sklearn._loss._loss": {"CyHalfBinomialLoss"},
        "sklearn._loss.link": {"LogitLink", "Interval"},
        "sklearn.ensemble._hist_gradient_boosting.gradient_boosting": {"HistGradientBoostingClassifier"},
        "sklearn.ensemble._hist_gradient_boosting.binning": {"_BinMapper"},
        "sklearn.ensemble._hist_gradient_boosting.predictor": {"TreePredictor"},
    }

    def find_class(self, module, name):
        if name not in self.ALLOWED.get(module, ()):
            raise pickle.UnpicklingError(f"Unexpected trial38 symbol {module}.{name}")
        return super().find_class(module, name)


def trial38(path):
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    if hashlib.sha1(header + data).hexdigest() != TRIAL_BLOB:
        raise ValueError("Trial38 checkpoint does not match the Git blob")
    with path.open("rb") as handle:
        return Trial38Unpickler(handle).load()


def scores(y, p):
    y = np.concatenate(y)
    p = np.concatenate(p).astype(np.float64)
    n = len(y)
    positives = int(y.sum())
    return dict(rows=n, positives=positives,
                auc=float(roc_auc_score(y, p)) if 0 < positives < n else None,
                auprc=float(average_precision_score(y, p)) if positives else None,
                brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, p, labels=[0, 1])))


def hb_predictions(position, box, old_length, topo, meta, predict, trial):
    old_ids, x_old = historical_hb_features(position, old_length, topo)
    pairs, x, _, _ = frame(position, box, topo, 8.0)
    n = len(position)
    modern_key = pairs[:, 0].astype(np.int64)*n + pairs[:, 1].astype(np.int64)
    old_key = old_ids[:, 0].astype(np.int64)*n + old_ids[:, 1]
    common, prior, modern = np.intersect1d(old_key, modern_key, return_indices=True)
    if not len(common):
        raise ValueError("No identical HB candidates")
    current = predict(inputs(pairs[modern], x[modern], meta["width"], meta["representation"]))
    original = trial.predict_proba(x_old[prior])[:, 1]
    return old_ids[prior], current, original, len(old_ids) - len(common)


def evaluate(root, runs):
    model = trial38(runs / "legacy/trial38.pkl")
    meta, predict_hb = load_cpu("T1I14", runs / "confirmation")
    legacy = load_models(runs / "legacy")
    identifiers = {"salt":"T1I311", "contact":"T1I325"}
    prepared = {}
    for task, name in identifiers.items():
        folder = runs / "events_soft" / task
        cfg = json.loads((folder / "validation.json").read_text())[name]
        predict = cpu_predict(folder / cfg["fit"]["checkpoint"], cfg["config"])
        rows = get_data(root, [PROTEIN], STRIDE, legacy, task, True, True)
        prepared[task] = [(part, predict(view(part, cfg["config"]))) for part in rows]
    results = {name:{h:dict(y=[], now=[], old=[], origins=0) for h in HORIZONS}
               for name in ("hb", "salt", "contact")}
    unmatched = 0
    for replica_index, replica in enumerate(REPLICAS):
        reg = registry(root, PROTEIN)
        tid = trajectory(PROTEIN, replica)
        blocks = list(reg.blocks(tid))
        positions = np.concatenate([b.positions for b in blocks])
        boxes = np.concatenate([b.box for b in blocks])
        times = np.concatenate([b.time for b in blocks])
        atoms = reg.atoms(tid).to_pydict()
        topo = topology(atoms)
        layout = build_layout(atoms)
        nres = len(layout.ca)
        old_length = float(np.asarray(boxes[0]).reshape(-1)[0])
        states = {}
        for index, origin in enumerate(range(49, len(times)-10, STRIDE)):
            ids, now, old, dropped = hb_predictions(positions[origin], cell(boxes[origin]), old_length,
                                                    topo, meta, predict_hb, model)
            unmatched += dropped
            for horizon in HORIZONS:
                target_ps = times[origin] + 1000*horizon
                k = int(np.searchsorted(times, target_ps - .01))
                if k >= len(times):
                    continue
                if not np.isclose(times[k], target_ps, rtol=0, atol=.01):
                    raise ValueError(f"Horizon {horizon} ns did not match {tid}")
                box = cell(boxes[k])
                separation = positions[k, ids[:, 1]] - positions[k, ids[:, 0]]
                y_hb = np.linalg.norm(_mic(separation, box), axis=1) <= 3.5
                row = results["hb"][horizon]
                row["y"].append(y_hb)
                row["now"].append(now)
                row["old"].append(old)
                row["origins"] += 1
                if k not in states:
                    states[k] = distances(layout, positions[k], box)
                state = states[k]
                for task, threshold in (("salt", 4.0), ("contact", 4.5)):
                    part, prediction = prepared[task][replica_index]
                    subset = slice(0 if index == 0 else part["ends"][index - 1], part["ends"][index])
                    atom_pairs = part["pair"][subset, :2].astype(np.int64)
                    a, b = layout.atom_res[atom_pairs].T
                    y = state[task][np.minimum(a, b)*nres + np.maximum(a,b)] <= threshold
                    if horizon == 1 and not np.array_equal(y, part["y"][subset]):
                        raise ValueError(f"{task} recomputed labels do not match 1 ns cache")
                    row = results[task][horizon]
                    row["y"].append(y)
                    row["now"].append(prediction[subset])
                    row["origins"] += 1
        print("AUDITED", tid, flush=True)
    summary = {}
    for task, horizons in results.items():
        summary[task] = {}
        for h, row in horizons.items():
            summary[task][h] = dict(origins=row["origins"], current=scores(row["y"], row["now"]),
                                    trial38=scores(row["y"], row["old"]) if task == "hb" else None)
    return dict(protein=PROTEIN, stride=STRIDE, training_horizon_ns=1,
                model_names={"hb":"T1I14", **identifiers},
                hb_unmatched_old_rows=unmatched, results=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("atlas-workspace"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    report = evaluate(args.data, args.runs)
    destination = args.runs / "horizon_comparison.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(destination)
