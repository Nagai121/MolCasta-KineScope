"""Hydrogen-bond exploration using only origin-frame inputs."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .compare_horizons import trial38
from .data import HISTORICAL, REPLICAS, TEST, TRAIN, VALIDATION, cell, inputs, registry, shard, topology, trajectory
from .features import _mic
from .models import load_cpu
from .search_events import EXTRA_TRAINING, bench, cpu_predict, dump, fit, test_report, view
from .trial38_features import historical_hb_features

ROOT = Path("atlas-workspace")
RUNS = Path("runs")
OUT = RUNS / "hb_single_structure"
PROTEIN = "1mix_A"
ADDITIONAL_TRAINING = ("3dso_A","2fe5_A","3luu_A","1mk0_A","3zdm_A","2xrh_A","3a39_A","1zt5_A")
BASIC = (
    ("xgb",118,"hb-origin-environment",False,4,1.0),
    ("xgb",118,"hb-origin-environment",False,6,1.0),
    ("xgb",142,"hb-origin-competition",False,4,1.0),
    ("xgb",142,"hb-origin-competition",True,5,1.0),
    ("xgb",166,"hb-origin-interaction",False,5,1.0),
    ("xgb",166,"hb-origin-interaction",True,4,1.5),
    ("xgb",166,"hb-origin-competition",True,6,1.0),
    ("mlp",118,"hb-origin-environment",False,64,1.0),
    ("mlp",142,"hb-origin-competition",True,64,1.0),
    ("mlp-tail",166,"hb-origin-interaction",True,64,1.0),
    ("xgb-soft-local",118,"hb-origin-environment",True,3,1.0),
    ("xgb-soft-local",142,"hb-origin-competition",True,4,1.0),
    ("xgb-soft-tune-0.15-0.15-400-local",166,"hb-origin-interaction",True,5,1.0),
    ("xgb-soft-wide-local",166,"hb-origin-environment",False,6,1.0),
)


def trajectory_parts(protein, replica, stride, targets, baseline):
    path = shard(ROOT, protein, replica, stride, 8.0)
    with np.load(path) as source:
        keep = source["pair"][:,2] <= 6.0
        pair, x, y, origins = (source[k][keep] for k in ("pair","features","y","origin"))
    ends = np.r_[np.flatnonzero(np.diff(origins))+1,len(y)]
    meta, model = baseline
    old = model(inputs(pair, x, meta["width"], meta["representation"]))
    part = dict(pair=pair, base=x, y=y, old=old, origin=origins, ends=ends, protein=protein,
                replica=replica)
    if targets:
        validation_path = OUT / "validation10.json"
        reserved = VALIDATION + TEST + HISTORICAL
        if validation_path.is_file():
            reserved += tuple(json.loads(validation_path.read_text(encoding="utf-8"))["proteins"])
        if protein in reserved:
            raise ValueError("Auxiliary labels are restricted to training proteins")
        if protein not in TRAIN + EXTRA_TRAINING + ADDITIONAL_TRAINING:
            panels=(OUT/"diverse_training_panel.json", OUT/"volume_training_panel.json",
                    OUT/"volume_96_training_panel.json", OUT/"volume_128_training_panel.json",
                    OUT/"ap10_volume_144_training_panel.json")
            extra={name for panel in panels if panel.is_file()
                   for name in json.loads(panel.read_text())["proteins"]}
            if protein not in extra:
                raise ValueError("Auxiliary labels are restricted to training proteins")
        cache = path.with_name(f"{path.stem}.hb-future-distance-v1.npy")
        if cache.is_file():
            future = np.load(cache, allow_pickle=False)
        else:
            reg = registry(ROOT, protein)
            tid = trajectory(protein, replica)
            labels = []
            for start, end in zip(np.r_[0,ends[:-1]], ends):
                t = int(origins[start])
                frame = next(reg.blocks(tid, start=t, stop=t+11, chunk_frames=11))
                if abs(float(frame.time[-1]-frame.time[0])-1000) > .01:
                    raise ValueError("Auxiliary labels must be exactly 1 ns ahead")
                a,b = pair[start:end,:2].astype(np.int64).T
                p = frame.positions[-1]
                labels.append(np.linalg.norm(_mic(p[b]-p[a],cell(frame.box[-1])),axis=1))
            future = np.concatenate(labels)
        if not np.isfinite(future).all() or not np.array_equal(y, future<=3.5):
            raise ValueError("Auxiliary distances do not match the binary labels")
        if not cache.is_file():
            temp = NamedTemporaryFile(dir=cache.parent, delete=False)
            try:
                with temp:
                    np.save(temp, future, allow_pickle=False)
                os.replace(temp.name, cache)
            finally:
                Path(temp.name).unlink(missing_ok=True)
        part.update(future_distance=future, threshold=3.5)
    return part


def load(proteins, stride, targets=False, geometry=False):
    baseline = load_cpu("T1I14", RUNS/"confirmation")
    tasks = [(p,r) for p in proteins for r in REPLICAS]
    def one(item):
        part = trajectory_parts(*item, stride, targets, baseline)
        if geometry:
            from .hb_local_geometry import add_geometry
            add_geometry(part, stride)
        return part
    with ThreadPoolExecutor(3) as pool:
        return list(pool.map(one,tasks))


def historical():
    parts = load((PROTEIN,),25)
    path = OUT/"trial38_scores.npz"
    if path.is_file():
        with np.load(path) as z:
            for part in parts:
                name = str(part["replica"])
                if not np.array_equal(z[f"pair_{name}"],part["pair"][:,:2]):
                    raise ValueError("Historical scores lost candidate order")
                part["trial38"] = z[f"trial_{name}"]
        return parts
    legacy = trial38(RUNS/"legacy/trial38.pkl")
    values = {}
    for part in parts:
        reg = registry(ROOT, PROTEIN)
        tid = trajectory(PROTEIN,part["replica"])
        topo = topology(reg.atoms(tid).to_pydict())
        first = next(reg.blocks(tid,start=0,stop=1))
        box = float(first.box.reshape(-1)[0])
        n = len(first.positions[0])
        batches = []
        for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
            t = int(part["origin"][start])
            past = next(reg.blocks(tid,start=t,stop=t+1))
            pairs, x = historical_hb_features(past.positions[0],box,topo)
            current = part["pair"][start:end,:2].astype(np.int64)
            keys = pairs[:,0]*n+pairs[:,1]
            order = keys.argsort()
            index = np.searchsorted(keys[order],current[:,0]*n+current[:,1])
            if np.any(index == len(order)) or not np.array_equal(pairs[order[index]],current):
                raise ValueError("Historical candidates differ")
            batches.append(legacy.predict_proba(x[order[index]])[:,1])
        part["trial38"] = np.concatenate(batches)
        label = str(part["replica"])
        values[f"trial_{label}"] = part["trial38"]
        values[f"pair_{label}"] = part["pair"][:,:2]
    np.savez(path,**values)
    return parts


def validation_report(parts, probabilities):
    return bench(parts, probabilities, ((None,None),))


def predict(parts, folder, record):
    if record["config"][0] == "hb-calibration":
        from .hb_ensemble_calibration import predict_calibrated
        return predict_calibrated(parts, folder, record)
    if record["config"][0] == "xgb-rank":
        from .hb_rank import rank_predict
        return rank_predict(parts, folder, record)
    if record["config"][0] == "hb-gnn":
        from .hb_graph import read_and_predict
        return read_and_predict(parts, folder, record)
    if record["config"][0] == "hb-transition":
        from .hb_transition import transition_predict
        return transition_predict(parts, folder, record)
    if record["config"][0] == "hb-relative-motion":
        from .hb_relative_motion import predict_motion
        return predict_motion(parts, folder, record)
    if record["config"][0] == "hb-residual-prior":
        from .hb_residual_prior import predict_residual_prior
        return predict_residual_prior(parts, folder, record)
    if record["config"][0] == "hb-experts":
        from .hb_experts import predict_experts
        return predict_experts(parts, folder, record)
    if record["config"][0] == "hb-crossfit":
        from .hb_crossfit import cross_predict
        return cross_predict(parts, folder, record)
    if record["config"][0] == "hb-distribution":
        from .hb_distribution import predict_distribution
        return predict_distribution(parts, folder, record)
    if record["config"][0] == "hb-prototype":
        from .hb_prototype import predict_prototypes
        return predict_prototypes(parts, folder, record)
    if record["config"][0] == "hb-constraint":
        from .hb_constraint import predict_review
        return predict_review(parts, folder, record)
    model = cpu_predict(folder/record["fit"]["checkpoint"],record["config"])
    return [model(view(part,record["config"])) for part in parts]


def key(record):
    v = record["validation"]
    return v["passes_accuracy"],v["screens"][0]["reduction"]


def explore(stage, configs, start, train, valid):
    folder = OUT/stage
    folder.mkdir(parents=True,exist_ok=True)
    result = {}
    for number,cfg in enumerate(configs,start):
        name = f"T1I{number}"
        fit_record = fit(name,folder,train,cfg)
        record = dict(config=cfg,fit=fit_record)
        record["validation"] = validation_report(valid,predict(valid,folder,record))
        result[name] = record
        dump(folder/"validation.json",result)
        row = record["validation"]
        print(name,row["candidate"]["auprc"],row["screens"][0]["reduction"],
              row["passes_accuracy"],flush=True)
    selected = max(result,key=lambda n:key(result[n]))
    dump(folder/"selection.json",dict(name=selected,**result[selected]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("basic","geometry","refine","finalize"),required=True)
    args = parser.parse_args()
    stage = args.stage
    if stage == "finalize":
        from .hb_local_geometry import finalize
        finalize()
        return
    geometry = stage != "basic"
    train = load(TRAIN,25,True,geometry) + load(EXTRA_TRAINING,50,True,geometry)
    valid = load(VALIDATION,100,geometry=geometry)
    print("STATIC DATA",stage,sum(len(p["y"]) for p in train),sum(len(p["y"]) for p in valid),flush=True)
    if not geometry:
        explore(stage,BASIC,541,train,valid)
    else:
        from .hb_local_geometry import geometry_configs
        configurations, start = geometry_configs(stage)
        explore(stage,configurations,start,train,valid)


if __name__ == "__main__":
    main()
