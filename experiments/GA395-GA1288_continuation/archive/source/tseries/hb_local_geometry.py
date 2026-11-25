"""Origin-frame geometry for the single-structure hydrogen-bond benchmark."""
import hashlib
import itertools
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.spatial import cKDTree

from .compare_hb_review import summarize
from .data import TEST, VALIDATION, cell, registry, topology, trajectory
from .features import _mic
from .hb_single_structure import OUT, ROOT, historical, key, load, predict, validation_report
from .search_events import count_to_recall, dump, test_report


def unit(delta):
    return delta/np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-7)


def project(a, b):
    return np.einsum("ij,ij->i", a, b)


def packing(position, box, queries):
    if box.ndim == 1:
        edge = np.nextafter(box.astype(np.float64), 0)
        xyz = np.minimum(np.remainder(position.astype(np.float64),box),edge)
        query = np.minimum(np.remainder(queries.astype(np.float64),box),edge)
        tree = cKDTree(xyz,boxsize=box)
        return [tree.query_ball_point(query,r,return_length=True) for r in (1.8,2.5,3.5)]
    blocks = []
    for idx in range(0,len(queries),64):
        delta = _mic(position[None]-queries[idx:idx+64,None],box)
        d = np.linalg.norm(delta,axis=2)
        blocks.append(np.column_stack([(d <= r).sum(1) for r in (1.8,2.5,3.5)]))
    return np.concatenate(blocks).T


def origin_features(position, box, pair, x, topo):
    maps, _, ctx, atom_type = topo
    donor, donor_anchor = ctx[5], ctx[8]
    acceptor, acceptor_anchor = ctx[9], ctx[12]
    count = len(position)
    da = np.zeros(count,dtype=np.int32)
    aa = np.zeros(count,dtype=np.int32)
    da[donor], aa[acceptor] = donor_anchor, acceptor_anchor
    previous = np.full(count,-1,dtype=np.int32)
    for prior, current in zip(maps[:-1],maps[1:]):
        previous[current["N"]] = prior["C"]
    a,b = pair[:,:2].astype(np.int64).T
    ab = _mic(position[b]-position[a],box)
    donor_out = unit(_mic(position[a]-position[da[a]],box))
    predecessor_out = _mic(position[a]-position[np.maximum(previous[a],0)],box)
    bonded = (previous[a]>=0) & (np.linalg.norm(predecessor_out,axis=1)<1.8)
    direction = unit(donor_out+unit(predecessor_out)*bonded[:,None])
    acceptor_out = unit(_mic(position[b]-position[aa[b]],box))
    ah = ab - direction
    dh_acceptor = np.linalg.norm(ah,axis=1)
    cos_dh = project(direction,unit(ah))
    cos_a = project(acceptor_out,-unit(ah))
    cos_d_ab = project(direction,unit(ab))
    mid = position[a]+direction+.5*ah
    packed = packing(position,box,mid)
    # Remove the scored endpoints themselves from the midpoint counts.
    d_a = np.linalg.norm(mid-position[a],axis=1)
    d_b = np.linalg.norm(.5*ah,axis=1)
    crowded = [c-(d_a<=r)-(d_b<=r) for c,r in zip(packed,(1.8,2.5,3.5))]
    residual = pair[:,2]-dh_acceptor
    out = np.column_stack((
        dh_acceptor, cos_dh, cos_a, cos_d_ab, cos_dh*cos_a,
        residual, crowded[0], crowded[1],
        crowded[2], bonded, abs(cos_d_ab-cos_a),
        np.maximum(0,1-cos_dh)*dh_acceptor,
        np.maximum(0,1-cos_a)*dh_acceptor,
        project(donor_out,acceptor_out), atom_type[a], atom_type[b],
        x[:,12]*cos_dh, x[:,13]*cos_a,
        x[:,138]*cos_dh, x[:,139]*cos_a,
        x[:,136] + dh_acceptor-pair[:,2], x[:,137] + dh_acceptor-pair[:,2],
    )).astype(np.float32)
    assert np.isfinite(out).all()
    return out


def add_geometry(part,stride):
    protein,replica = part["protein"],part["replica"]
    reg = registry(ROOT,protein)
    tid = trajectory(protein,replica)
    topo = topology(reg.atoms(tid).to_pydict())
    version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    directory = ROOT/protein/"trajectories"/tid/"hb-origin-geometry"
    directory.mkdir(exist_ok=True)
    cache = directory/f"{stride}-{version}.npz"
    if cache.exists():
        with np.load(cache) as z:
            if np.array_equal(z["pair"],part["pair"]) and np.array_equal(z["origin"],part["origin"]):
                part["local"] = z["local"]
                return
    rows = []
    for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
        origin = int(part["origin"][start])
        block = next(reg.blocks(tid,start=origin,stop=origin+1))
        rows.append(origin_features(block.positions[0],cell(block.box[0]),part["pair"][start:end],
                                    part["base"][start:end],topo))
    features = np.concatenate(rows)
    tmp = NamedTemporaryFile(dir=directory,delete=False)
    try:
        with tmp:
            np.savez(tmp,local=features,pair=part["pair"],origin=part["origin"])
        os.replace(tmp.name,cache)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    part["local"] = features


def geometry_configs(stage):
    if stage == "geometry":
        configurations = (
            ("xgb",118,"hb-origin-local",False,3,1.0),
            ("xgb",118,"hb-origin-local",True,5,1.0),
            ("xgb",142,"hb-origin-local",False,4,1.0),
            ("xgb",166,"hb-origin-local",True,6,1.0),
            ("xgb-soft-local",118,"hb-origin-local",False,3,1.0),
            ("xgb-soft-wide-local",166,"hb-origin-local",True,4,1.0),
            ("xgb-soft-tune-0.15-0.15-400-local",118,"hb-origin-local",True,5,1.0),
            ("xgb-soft-tune-0.25-0.4-400-local",142,"hb-origin-local",False,5,1.0),
            ("mlp",118,"hb-origin-local",True,64,1.0),
            ("mlp-tail",142,"hb-origin-local",False,64,1.0),
            ("mlp-tail-strong",166,"hb-origin-local",True,96,1.0),
            ("xgb",118,"hb-origin-local",True,7,.75),
        )
        return configurations, 561
    files = [OUT/s/"validation.json" for s in ("basic","geometry")]
    pool = {name:row for path in files for name,row in json.loads(path.read_text()).items()}
    leaders = sorted(pool.values(),key=key,reverse=True)
    configurations = []
    for row in leaders:
        _, width, rep, inherit, depth, _ = row["config"]
        if depth > 12:
            depth = 4
        for strategy,delta in (("xgb-soft-tune-0.15-0.25-160-local",-1),
                               ("xgb-soft-tune-0.35-0.25-320-local",0),
                               ("xgb-soft-tune-0.25-0.4-500-local",1)):
            configuration = (strategy,width,rep,inherit,max(2,depth+delta),1.0)
            if configuration not in configurations:
                configurations.append(configuration)
        if len(configurations)>=12:
            break
    return configurations[:12],581


def mixture_name(weights, trained_names):
    active = {name: value for name, value in weights.items() if value}
    if len(active) == 1:
        return next(iter(active))
    path = OUT/"mixture_ids.json"
    existing = (json.loads(path.read_text()) if path.is_file() else
                {"T1I601": {"T1I542": .875, "T1I589": .125}})
    for name, definition in existing.items():
        if name not in trained_names and definition == active:
            return name
    reserved=set(existing)|trained_names
    number = max(2000, *(int(name.removeprefix("T1I")) for name in reserved))+1
    name = f"T1I{number:02}"
    existing[name] = active
    dump(path,existing)
    return name


def finalize():
    if (OUT/"final.json").is_file():
        raise FileExistsError("Final test has already been evaluated")
    if not (OUT/"ranking/selection.json").is_file():
        from .hb_rank import main
        main()
    if not (OUT/"flexibility/selection.json").is_file():
        from .hb_flexibility import main
        main()
    pool = {}
    stages = ["basic","geometry","refine","ranking","flexibility"]
    for stage in ("graph", "graph_volume", "dense", "directional", "transition", "transition_volume", "experts", "experts_volume",
                  "motif", "exposure", "focus", "focus_volume", "crossfit", "distribution",
                  "distribution_volume", "cohort", "cloud", "charge", "prototype", "diverse",
                  "balanced", "balanced_volume", "balanced_volume_96", "volume", "volume_96", "volume_128",
                  "constraint", "constraint_volume", "ranking_volume", "occupied", "stacked", "stacked_volume",
                  "stacked_volume_96", "stacked_volume_128", "stacked_dense_origins", "stacked_explicit",
                  "stiffness", "origin_context", "ensemble_calibration", "ensemble_gate", "torsional",
                  "explicit_hydrogen", "relative_motion", "peptide_motion", "geometry_teacher", "residual_prior"):
        if (OUT/stage/"selection.json").is_file():
            stages.append(stage)
    stages.extend(p.name for pattern in ("scaling_*", "window_*", "replicated_teacher_*", "distilled_trajectory_*") for p in sorted(OUT.glob(pattern))
                  if (p/"selection.json").is_file())
    for stage in stages:
        folder=OUT/stage
        for name,row in json.loads((folder/"validation.json").read_text()).items():
            if name in pool:
                raise ValueError(f"Duplicate trained model {name}")
            if hashlib.sha256((folder/row["fit"]["checkpoint"]).read_bytes()).hexdigest()!=row["fit"]["sha256"]:
                raise ValueError(f"Changed checkpoint {name}")
            pool[name]=dict(row,folder=folder)
    champions = dict.fromkeys(max(pool,key=lambda n:order(pool[n])) for order in (
        key,lambda r:r["validation"]["candidate"]["auprc"],
        lambda r:-r["validation"]["candidate"]["logloss"],
    ))
    previous = [json.loads(file.read_text()) for file in (OUT/"selections").glob("*/selection.json")]
    incumbent = max(previous, key=key, default=None)
    incumbent_weights = ({n:w for n,w in incumbent["weights"].items() if w}
                         if incumbent is not None else {})
    challengers=sorted(pool,key=lambda n:key(pool[n]),reverse=True)[:12]
    names = tuple(dict.fromkeys((*champions, *incumbent_weights, *challengers)))
    def outputs(parts,stride):
        from .hb_ensemble_calibration import component_representations
        representations=set().union(*(component_representations(pool[n]["folder"],pool[n]) for n in names))
        stacked=any(rep.startswith("hb-origin-stacked-") for rep in representations)
        if representations & {"hb-origin-explicit","hb-origin-stacked-explicit"}:
            from .hb_explicit_hydrogen import add_explicit
            for part in parts:
                add_explicit(part,stride)
        if "hb-origin-peptide" in representations:
            from .hb_peptide_motion import add_peptide_motion
            for part in parts:
                add_peptide_motion(part,stride)
        if "hb-origin-torsional" in representations:
            from .hb_torsional import add_torsional
            for part in parts:
                add_torsional(part,stride)
        if "hb-origin-stiffness" in representations:
            from .hb_atom_stiffness import add_stiffness
            for part in parts:
                add_stiffness(part,stride)
        if stacked or "hb-origin-occupied" in representations:
            from .hb_occupied import add_occupied
            for part in parts:
                add_occupied(part,stride)
        if any(pool[n]["config"][2]=="hb-origin-charge" for n in names):
            from .hb_charge import add_charge
            for part in parts:
                add_charge(part,stride)
        if stacked or "hb-origin-cloud" in representations:
            from .hb_cloud import add_cloud
            for part in parts:
                add_cloud(part,stride)
        if any(pool[n]["config"][2]=="hb-origin-union" for n in names):
            from .hb_crossfit import add_union
            add_union(parts,stride)
        if any(pool[n]["config"][2]=="hb-origin-flexibility" for n in names):
            from .hb_flexibility import add_flexibility
            for part in parts:
                add_flexibility(part,stride)
        if stacked or "hb-origin-directional" in representations:
            from .hb_directional import add_directional
            for part in parts:
                add_directional(part,stride)
        if any(pool[n]["config"][2]=="hb-origin-motif" for n in names):
            from .hb_motif import add_motif
            for part in parts:
                add_motif(part,stride)
        if any(pool[n]["config"][2]=="hb-origin-exposure" for n in names):
            from .hb_exposure import add_exposure
            for part in parts:
                add_exposure(part,stride)
        return [predict(parts,pool[n]["folder"],pool[n]) for n in names]
    def mixture(predictions,weights):
        return [sum(w*p for w,p in zip(weights,values)) for values in zip(*predictions)]
    valid = load(VALIDATION,100,geometry=True)
    pred = outputs(valid,100)
    ledger=[]
    for w in itertools.product(range(9),repeat=len(champions)):
        if sum(w)!=8:
            continue
        current=dict(zip(champions,(i/8 for i in w)))
        weights=[current.get(n,0.) for n in names]
        ledger.append(dict(weights=dict(zip(names,weights)),
                           validation=validation_report(valid,mixture(pred,weights))))
    if incumbent_weights:
        weights=[incumbent_weights.get(n,0.) for n in names]
        ledger.append(dict(weights=dict(zip(names,weights)),
                           validation=validation_report(valid,mixture(pred,weights))))
    for _ in range(3):
        incumbent=max(ledger,key=key)
        previous_key=key(incumbent)
        for challenger in challengers:
            for fraction in (.0625,.125,.25,.5):
                weights=[(1-fraction)*incumbent["weights"].get(n,0.)
                         +fraction*(n==challenger) for n in names]
                ledger.append(dict(weights=dict(zip(names,weights)),
                                   validation=validation_report(valid,mixture(pred,weights))))
        if max(map(key,ledger))<=previous_key:
            break
    dump(OUT/"blends.json",ledger)
    winner=max(ledger,key=lambda r:(*key(r),-r["validation"]["candidate"]["logloss"],
                                   r["validation"]["candidate"]["auprc"],
                                   -sum(w>0 for w in r["weights"].values())))
    weights=list(winner["weights"].values())
    name=mixture_name(winner["weights"],set(pool))
    selection=dict(name=name,**winner)
    dump(OUT/"selection.json",selection)
    archive=OUT/"selections"/name
    archive.mkdir(parents=True,exist_ok=True)
    dump(archive/"selection.json",selection)
    record=archive/"historical.json"
    if record.is_file():
        historical_result=json.loads(record.read_text())
        pooled=historical_result["workload"]
    else:
        parts=historical()
        for part in parts:
            add_geometry(part,25)
        scores=mixture(outputs(parts,25),weights)
        report=validation_report([dict(p,old=p["trial38"]) for p in parts],scores)
        pooled=summarize(np.concatenate([p["y"] for p in parts]),np.concatenate([p["trial38"] for p in parts]),
                         np.concatenate(scores))
        assert (pooled["rows"],pooled["positives"],pooled["trial38_review"])==(119085,7905,19955)
        historical_result=dict(name=name,metrics=report,workload=pooled)
        dump(record,historical_result)
    dump(OUT/"historical.json",historical_result)
    results=winner["validation"]
    if results["passes_accuracy"] and results["screens"][0]["passes_work"]:
        test=load(TEST,100,geometry=True)
        dump(OUT/"final.json",test_report(test,mixture(outputs(test,100),weights),(None,None)))
    print("SELECTED",name,winner["weights"],key(winner),pooled,flush=True)
