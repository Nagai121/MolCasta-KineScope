"""K81 generation step: proposal-22 (idea 22) paths for the
SALT-BRIDGE DISSOCIATION candidates that can be fed to the generator.

Why this exists: K72 measured the error shape of the generated mean
for salt-bridge dissociation with the residue-level method (ideas
11-14). K81 asks whether idea 22 gives the same shape (K76 did the
same comparison for contact dissociation).

Method: the idea-22 pipeline, the same as K05's regeneration, K54's
first attempt (gen_k54.py) and K76 (gen_k76.py), adapted to
salt-bridge dissociation candidates:
- Candidates are K53's salt-bridge dissociation residue pairs
  (origin minimum charged heavy-atom distance <= 4.0 A, K15 expanded
  set, 88 proteins). For each candidate the representative charged
  heavy-atom pair (the pair attaining the minimum distance at the
  origin, frame 10) is used as the input pair.
- Idea 22's descriptors are defined only for (donor, acceptor)
  pairs. Salt-bridge representative pairs are (positive N, negative
  O) pairs, so most are feedable; the exact fraction is measured
  beforehand by check_feedable.py and recorded in the README.
  Non-feedable candidates are not measured; results apply to the
  feedable subset only.
- The pair list is built directly (no frame()/expand() call, since
  expand() selects donor-acceptor pairs by the formation study's
  rule); everything downstream is the same code as K54's first
  attempt / K76 with the same inputs: directional features, local
  sites, elastic modes, descriptors, network, constrain. The
  generator keeps pairs with origin distance <= 6.0 A, which
  includes every feedable dissociation candidate (origin <= 4.0 A).
- Data: atoms.parquet / A.zarr (same path as gen_k65.py: the
  original 6 proteins from gpt-model-check/data/trajectories, the
  other 82 from k15_atlas/<protein>/trajectories). The atoms dict
  for topology() is built from atoms.parquet (it carries an element
  column); the residue maps are verified against layout().

Generator handling, exactly as the user fixed for K54:
- The checkpoint is ~/workspace/k06b/gen22/generator.pt (261,929
  bytes). Its SHA-256 is checked against the record in
  generator.json (research/tseries-models-and-results-20261121
  branch, research/tseries-exploration-20261121/runs/hb_single_structure/
  trajectory_mechanistic/ideas_21_25/generator.json):
  d35b5d92043fc62479a52c2468a8ae50e56d5442846af0f1d8df2ecfd3e88a05.
  The script recomputes the hash and refuses to load the file if it
  does not match.
- The checkpoint is loaded with torch.load(..., weights_only=True),
  the same as K05.
- The tseries reference implementation is used from
  ~/workspace/k06b/pkg (read-only copy).
Caveat: the network was trained on hydrogen-bond formation pairs
only, so applying it to salt-bridge dissociation representative
pairs is an extrapolation outside its training range, twice over
(salt-bridge pairs, dissociation candidates). The generated series
are checked against the real forward continuation (K05-style
quality check) before the error-structure measurement; if they are
not sane, the result measures generator extrapolation failure, not
the error shape.

Per trajectory the output npz holds res_i, res_j, rep_a, rep_b
(representative atom pair), gen22 (origin + 4 forward distances at
0.1/0.3/0.5/1.0 ns), coarse (elastic-only, same 5 points), rep5
(real representative-pair distances at the same 5 points), min5
(real minimum distances at the same 5 points).
"""
import sys, json, time, hashlib
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
import minitopol  # registers the minitopol_agx3 zarr codec
from pathlib import Path
import numpy as np, pandas as pd, zarr
from tseries.data import topology, cell
from tseries.features import _mic, structural, feature_graph, enriched_features
from tseries.hb_elastic_paths import elastic_modes, mode_paths
from tseries.hb_directional import directional_features
from tseries.hb_learned_coordinate_paths import (local_sites, descriptors,
                                                 network, constrain, SEED)

GPT_BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
ATLAS_WS = Path("/home/hatch/workspace/kinescope/work/k15_atlas")
CACHE = Path("/home/hatch/workspace/kinescope/work/k53_cache")
GEN_PATH = Path("/home/hatch/workspace/k06b/gen22/generator.pt")
RECORDED_SHA256 = "d35b5d92043fc62479a52c2468a8ae50e56d5442846af0f1d8df2ecfd3e88a05"
OUTDIR = Path("/home/hatch/workspace/kinescope/work/k81_gen22")
OUTDIR.mkdir(parents=True, exist_ok=True)
_SPLIT = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion/split.json").read_text())
PROTS = tuple(_SPLIT["orig_train"]) + tuple(_SPLIT["train_additions"]) + tuple(_SPLIT["orig_test"]) + tuple(_SPLIT["test_additions"])
FRAMES5 = (10, 11, 13, 15, 20)
ORIGIN = 10


def trajdir(prot, rep):
    tid = f"atlas-{prot}-analysis-R{rep}"
    if prot in tuple(_SPLIT["orig_train"]) + tuple(_SPLIT["orig_test"]):
        return GPT_BASE / tid
    return ATLAS_WS / prot / "trajectories" / tid


def layout(df):
    order = list(dict.fromkeys(df["residue_index"].tolist()))
    maps, names = [], []
    for r in order:
        idx = np.flatnonzero(df["residue_index"].values == r)
        nm = {df["name"].values[i]: int(i) for i in idx}
        if not {"N", "CA", "C", "O"}.issubset(nm):
            continue
        maps.append(nm); names.append(df["residue_name"].values[idx[0]])
    return maps, names


def charged_atoms(maps, names, i):
    r = names[i]
    m = maps[i]
    if r == "LYS" and "NZ" in m:
        return np.array([m["NZ"]], dtype=int)
    if r == "ARG":
        return np.array([m[a] for a in ("NE", "NH1", "NH2") if a in m], dtype=int)
    if r in ("ASP", "GLU"):
        keys = ("OD1", "OD2") if r == "ASP" else ("OE1", "OE2")
        return np.array([m[a] for a in keys if a in m], dtype=int)
    return np.empty(0, dtype=int)


def load_generator():
    import torch
    digest = hashlib.sha256(GEN_PATH.read_bytes()).hexdigest()
    print("generator sha256:", digest, flush=True)
    assert digest == RECORDED_SHA256, "generator hash does not match generator.json"
    bundle = torch.load(str(GEN_PATH), map_location="cpu", weights_only=True)
    model = network(len(bundle["center"]))
    model.load_state_dict(bundle["state"])
    model.eval()
    return model, bundle["center"].numpy(), bundle["scale"].numpy()


def run_one(protein, replica, model, center, scale):
    import torch
    tid = f"atlas-{protein}-analysis-R{replica}"
    d = trajdir(protein, replica)
    df = pd.read_parquet(d / "atoms.parquet")
    atoms = {"residue_index": df["residue_index"].values,
             "name": df["name"].values,
             "residue_name": df["residue_name"].values,
             "element": df["element"].values}
    topo = topology(atoms)
    maps, residue_names, context, atom_types = topo
    maps_l, names_l = layout(df)
    assert len(maps) == len(maps_l), tid
    assert all(maps[i]["CA"] == maps_l[i]["CA"] for i in range(len(maps))), tid
    assert residue_names == names_l, tid
    d_at, a_at = context[5], context[9]
    d_index = {int(at): i for i, at in enumerate(d_at)}
    a_index = {int(at): j for j, at in enumerate(a_at)}
    zz = np.load(CACHE / f"{protein}_R{replica}_salt_diss.npz")
    ri, rj = zz["res_i"], zz["res_j"]
    origin_cache = zz["hist"][:, -1].astype(np.float64)
    if len(ri) == 0:
        # No candidates (e.g. 4cv4_A R2 has an all-zero box recorded) --
        # the real side is empty exactly as in K53; write an empty
        # output and return before touching the box. Same as K20/K65.
        np.savez_compressed(OUTDIR / f"{tid}.npz",
                            res_i=ri, res_j=rj,
                            rep_a=np.empty(0, dtype=int), rep_b=np.empty(0, dtype=int),
                            gen22=np.empty((0, 5), dtype=np.float32),
                            coarse=np.empty((0, 5), dtype=np.float32),
                            rep5=np.empty((0, 5), dtype=np.float32),
                            min5=np.empty((0, 5), dtype=np.float32))
        return dict(tid=tid, n=0)
    z = zarr.open_group(str(d / "A.zarr"), mode='r')
    pos64 = z['positions'][:].astype(np.float64)
    box64 = z['box'][:].astype(np.float64)
    position = np.asarray(pos64[ORIGIN], dtype=np.float32)
    bx = cell(box64[ORIGIN])
    rep_a, rep_b, keep_idx = [], [], []
    min5_list = []
    for zi, (i, j) in enumerate(zip(ri, rj)):
        ia = charged_atoms(maps, residue_names, int(i))
        ib = charged_atoms(maps, residue_names, int(j))
        assert len(ia) > 0 and len(ib) > 0, (tid, i, j)
        dist = np.linalg.norm(_mic(pos64[ORIGIN][np.repeat(ia, len(ib))] - pos64[ORIGIN][np.tile(ib, len(ia))],
                                   cell(box64[ORIGIN])), axis=1)
        kmin = int(np.argmin(dist))
        ra, rb = int(ia[kmin // len(ib)]), int(ib[kmin % len(ib)])
        assert abs(dist[kmin] - origin_cache[zi]) < 1e-5, (tid, zi, dist[kmin], origin_cache[zi])
        if ra in d_index and rb in a_index:
            da, ab = ra, rb
        elif rb in d_index and ra in a_index:
            da, ab = rb, ra
        else:
            continue  # not feedable: representative pair is not donor-acceptor
        keep_idx.append(zi)
        rep_a.append(da); rep_b.append(ab)
        a_all = np.repeat(ia, len(ib)); b_all = np.tile(ib, len(ia))
        series = []
        for f in FRAMES5:
            vals = np.linalg.norm(_mic(pos64[f][b_all] - pos64[f][a_all], cell(box64[f])), axis=1)
            series.append(vals.min())
        min5_list.append(series)
    keep_idx = np.array(keep_idx, dtype=int)
    rep_a = np.array(rep_a, dtype=int); rep_b = np.array(rep_b, dtype=int)
    n = len(keep_idx)
    if n == 0:
        np.savez_compressed(OUTDIR / f"{tid}.npz",
                            res_i=np.empty(0, dtype=int), res_j=np.empty(0, dtype=int),
                            rep_a=rep_a, rep_b=rep_b,
                            gen22=np.empty((0, 5), dtype=np.float32),
                            coarse=np.empty((0, 5), dtype=np.float32),
                            rep5=np.empty((0, 5), dtype=np.float32),
                            min5=np.empty((0, 5), dtype=np.float32))
        return dict(tid=tid, n=0, feedable_of=int(len(ri)))
    delta = _mic(position[rep_b] - position[rep_a], bx)
    dist0 = np.linalg.norm(delta, axis=1)
    pair = np.column_stack((rep_a, rep_b, dist0, delta,
                            atom_types[rep_a], atom_types[rep_b])).astype(np.float32)
    assert np.abs(dist0 - origin_cache[keep_idx]).max() < 1e-5
    ii = np.array([d_index[int(a)] for a in rep_a], dtype=int)
    jj = np.array([a_index[int(b)] for b in rep_b], dtype=int)
    base = structural(position, bx, maps, residue_names, context, ii, jj)
    graph = feature_graph(base, pair[:, 0], pair[:, 1], np.zeros(len(pair)), len(position))
    features = enriched_features(np.column_stack((base, graph)))
    part = dict(pair=pair, base=features)
    part["directional"] = directional_features(position, bx, pair, maps)
    sites, basis, local = local_sites(position, bx, pair, atoms)
    ca = np.array([r["CA"] for r in maps])
    owner = np.full(len(atoms["name"]), -1, dtype=np.intp)
    for i, residue in enumerate(maps):
        owner[list(residue.values())] = i
    values, modes, _, _, _ = elastic_modes(position[ca], bx)
    rng = np.random.default_rng([SEED, ORIGIN, int(replica)])
    signal = mode_paths(modes, values, rng)[:, (0, 2, 4, 7)]
    coarse = np.einsum("ncm,rtm->rtnc", modes, signal)
    coarse = coarse[:, :, np.maximum(owner[sites], 0)]
    coarse -= coarse[:, :, :, :1].copy()
    coarse_local = np.einsum("rtnsc,nck->nrtsk", coarse, basis).mean(axis=1).astype(np.float32)
    part["sites"] = local; part["coarse"] = coarse_local
    x = descriptors(part)
    assert x.shape[1] == len(center), (x.shape, len(center))
    finite = np.isfinite(x).all(axis=1)
    norm = np.clip((x - center) / scale, -12., 12.)
    with torch.inference_mode():
        out = model(torch.from_numpy(norm))
    h = out[:, :360].reshape(-1, 6, 4, 5, 3).numpy()
    correction = h[:, 2]
    tc = torch.from_numpy(local)
    def distances(disp):
        p = constrain(tc, torch.as_tensor(disp[:, None], dtype=torch.float32)).numpy()
        return np.linalg.norm(p[:, :, :, 1], axis=-1)[:, 0]
    gen_dist = distances(coarse_local + correction)
    coarse_dist = distances(coarse_local)
    gen22 = np.column_stack([dist0, gen_dist]).astype(np.float32)
    coarse5 = np.column_stack([dist0, coarse_dist]).astype(np.float32)
    rep5 = np.stack([np.linalg.norm(_mic(pos64[f][rep_b] - pos64[f][rep_a], cell(box64[f])), axis=1)
                     for f in FRAMES5], axis=1)
    min5 = np.array(min5_list, dtype=np.float32)
    np.savez_compressed(OUTDIR / f"{tid}.npz",
                        res_i=ri[keep_idx], res_j=rj[keep_idx],
                        rep_a=rep_a, rep_b=rep_b, gen22=gen22, coarse=coarse5,
                        rep5=rep5.astype(np.float32), min5=min5)
    return dict(tid=tid, n=int(n), feedable_of=int(len(ri)),
                nonfinite=int((~finite).sum()),
                gen_mean=float(gen22.mean()), gen_std=float(gen22.std(axis=1).mean()),
                rep_mean=float(rep5.mean()))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    model, center, scale = load_generator()
    print("generator input width:", len(center), flush=True)
    results = []
    for protein in PROTS:
        for replica in (1, 2, 3):
            tid = f"atlas-{protein}-analysis-R{replica}"
            if only and only not in tid:
                continue
            if (OUTDIR / f"{tid}.npz").is_file():
                print("SKIP", tid, flush=True)
                continue
            t0 = time.time()
            r = run_one(protein, replica, model, center, scale)
            r["seconds"] = round(time.time() - t0, 1)
            results.append(r)
            print(json.dumps(r), flush=True)
    (OUTDIR / "adapter_summary.json").write_text(json.dumps(results, indent=1))
    print("saved to", OUTDIR)


if __name__ == "__main__":
    main()
