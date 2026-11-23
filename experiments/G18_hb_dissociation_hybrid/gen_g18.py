"""G18 generation step: regenerate proposal-22 paths for the
DISSOCIATION candidates, using the saved generator.

This is K54's gen_k54.py run again for the same 18 trajectories, with
the output kept in a G18 directory instead of the K54 directory.
Echizen's decision was option (a): regenerate the method-22 series for
the dissociation candidates. The loading procedure is the one Echizen
approved for the K05 regeneration on 2026-11-23 (and used again for
K74): the checkpoint is loaded with torch.load(..., weights_only=True)
only after its SHA-256 is checked against generator.json.

Generator handling, exactly as the user fixed:
- The checkpoint is ~/workspace/k06b/gen22/generator.pt (261,929
  bytes). Its SHA-256 was checked against the record in
  generator.json (research/tseries-models-and-results-20261121 branch,
  research/tseries-exploration-20261121/runs/hb_single_structure/
  trajectory_mechanistic/ideas_21_25/generator.json):
  d35b5d92043fc62479a52c2468a8ae50e56d5442846af0f1d8df2ecfd3e88a05.
  The script below recomputes the hash and refuses to load the file
  if it does not match.
- The checkpoint is loaded with torch.load(..., weights_only=True),
  the same as K05.
- The tseries reference implementation is used from
  ~/workspace/k06b/pkg (read-only copy); no /tmp stubs are needed for
  these imports (verified before running).

One deviation from K05's path, forced by the data and recorded here:
the reference `frame()`/`expand()` selects only pairs with origin
distance > 3.5 A (`eligible = distance > HB_DISTANCE` in
tseries/rod.py). That filter is the formation study's candidate
definition, not part of the generation method itself (elastic modes +
learned correction + constrain), but it means the pipeline as written
can never produce paths for dissociation candidates (origin <= 3.5
A). The first run of this script confirmed it: the regenerated set
was exactly K05's 14,582 formation pairs, and no dissociation
candidate matched. So `expand` is replaced here by `expand_all`,
which is the same code with only the eligibility line changed (every
cross-residue donor-acceptor pair is eligible; the label rule
future distance <= 3.5 A is unchanged). Everything downstream
(directional features, local sites, elastic modes, descriptors,
network, constrain) is the same code with the same inputs.
Caveat: the network was trained on formation pairs only (the teacher
rows come from the same frame() selection), so applying it to
dissociation candidates is an extrapolation outside its training
range. The generated series are therefore checked against the real
forward continuation (K05-style quality check) before the swap
evaluation in run_g18.py; if they are not sane, the swap result
would measure generator extrapolation failure, not the error shape.

Per trajectory the output npz holds pair (donor atom, acceptor atom,
origin distance), gen_dist (4 forward distances at 0.1/0.3/0.5/1.0
ns), coarse_dist (elastic-only), gen8 (native 8 summary features),
real_sub (real forward distances at the same 4 times), and y.
The generator keeps pairs with origin distance <= 6.0 A, which
includes every dissociation candidate (origin <= 3.5 A); run_g18.py
asserts that every dissociation candidate matches a generated pair.
"""
import sys, json, time, hashlib
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
from pathlib import Path
import numpy as np
from minitopol import Workspace
from minitopol.registry import Registry

import tseries.data as tdata
from tseries.data import frame, topology, cell
from tseries.features import _mic
from tseries.hb_elastic_paths import elastic_modes, mode_paths
from tseries.hb_directional import directional_features
from tseries.hb_learned_coordinate_paths import (local_sites, descriptors,
                                                 network, constrain, summarize,
                                                 FRAMES, SEED)
from tseries.rod import HB_DISTANCE


def expand_all(positions, box, context, atom_types, screening_distance=None, *, future=None,
               future_box=None, chunk_atoms=64):
    """tseries.rod.expand with only the eligibility rule changed.

    The reference expand() keeps only pairs with origin distance >
    3.5 A (the formation study's candidate definition). Everything
    else here is copied verbatim, including the label rule
    (future distance <= 3.5 A). Patching tdata.expand before calling
    frame() makes frame() use this version.
    """
    n_atoms = len(positions)
    if n_atoms > 2**24:
        raise ValueError("Atom indices exceed exact float32 integer precision")
    _, _, _, _, _, d_at, d_res, _, _, a_at, a_res, _, _, _ = context
    donor = np.full(n_atoms, -1, dtype=np.int32)
    acceptor = donor.copy()
    donor[d_at] = np.arange(len(d_at))
    acceptor[a_at] = np.arange(len(a_at))
    atoms = np.arange(n_atoms)
    vectors, donor_indices, acceptor_indices, labels = [], [], [], []
    counts = dict(expanded=0, chemistry_orientations=0, eligible=0, eligible_positives=0,
                  retained=0, retained_positives=0)
    for start in range(0, n_atoms, chunk_atoms):
        left, right = np.where(atoms[start:start + chunk_atoms, None] < atoms)
        left += start
        counts["expanded"] += len(left)
        for first, second in ((left, right), (right, left)):
            i, j = donor[first], acceptor[second]
            typed = (i >= 0) & (j >= 0)
            i, j = i[typed], j[typed]
            cross = d_res[i] != a_res[j]
            i, j = i[cross], j[cross]
            counts["chemistry_orientations"] += len(i)
            a, b = d_at[i], a_at[j]
            delta = _mic(positions[b] - positions[a], box)
            distance = np.linalg.norm(delta, axis=1)
            eligible = np.ones_like(distance, dtype=bool)  # was: distance > HB_DISTANCE
            counts["eligible"] += np.count_nonzero(eligible)
            retained = eligible if screening_distance is None else (
                eligible & (distance <= screening_distance)
            )
            if future is not None:
                future_distance = np.linalg.norm(_mic(future[b] - future[a], future_box), axis=1)
                label = future_distance <= HB_DISTANCE
                counts["eligible_positives"] += np.count_nonzero(label & eligible)
                counts["retained_positives"] += np.count_nonzero(label & retained)
                labels.append(label[retained])
            a, b = a[retained], b[retained]
            vectors.append(np.column_stack((a, b, distance[retained], delta[retained],
                                            atom_types[a], atom_types[b])).astype(np.float32))
            donor_indices.append(i[retained])
            acceptor_indices.append(j[retained])
            counts["retained"] += len(a)
    assert counts["expanded"] == n_atoms * (n_atoms - 1) // 2
    return (np.concatenate(vectors), np.concatenate(donor_indices),
            np.concatenate(acceptor_indices), np.concatenate(labels) if labels else None,
            {key: int(value) for key, value in counts.items()})


tdata.expand = expand_all

ORIGIN = 10
DATA = Path.home() / "workspace/gpt-model-check/data"
GEN_PATH = Path("/home/hatch/workspace/k06b/gen22/generator.pt")
RECORDED_SHA256 = "d35b5d92043fc62479a52c2468a8ae50e56d5442846af0f1d8df2ecfd3e88a05"
OUTDIR = Path("/home/hatch/workspace/kinescope/work/g18_gen22")
OUTDIR.mkdir(parents=True, exist_ok=True)
GEN_SUMMARY = Path(__file__).resolve().parent / "gen_summary.json"


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


def run_one(reg, protein, replica, model, center, scale):
    import torch
    tid = f"atlas-{protein}-analysis-R{replica}"
    atoms = reg.atoms(tid).to_pydict()
    topo = topology(atoms)
    blocks = list(reg.blocks(tid, start=0, stop=21, chunk_frames=21))
    pos = np.concatenate([b.positions for b in blocks], axis=0)
    box = np.concatenate([b.box for b in blocks], axis=0)
    position = np.asarray(pos[ORIGIN]); bx = cell(box[ORIGIN])
    pair, base, y, counts = frame(position, bx, topo, 8.0,
                                  future=np.asarray(pos[20]), future_box=cell(box[20]))
    keep = pair[:, 2] <= 6.0
    pair, base, y = pair[keep], base[keep], y[keep]
    part = dict(pair=pair, base=base, y=y, protein=protein, replica=replica)
    part["directional"] = directional_features(position, bx, pair, topo[0])
    sites, basis, local = local_sites(position, bx, pair, atoms)
    residues = topo[0]
    ca = np.array([r["CA"] for r in residues])
    owner = np.full(len(atoms["name"]), -1, dtype=np.intp)
    for i, residue in enumerate(residues):
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
    norm = np.clip((x - center) / scale, -12., 12.)
    with torch.inference_mode():
        out = model(torch.from_numpy(norm))
    h = out[:, :360].reshape(-1, 6, 4, 5, 3).numpy()
    correction = h[:, 2]
    tc = torch.from_numpy(local)
    def distances(disp):
        p = constrain(tc, torch.as_tensor(disp[:, None], dtype=torch.float32)).numpy()
        return np.linalg.norm(p[:, :, :, 1], axis=-1)[:, 0], p
    gen_dist, gen_path = distances(coarse_local + correction)
    coarse_dist, _ = distances(coarse_local)
    gen8 = summarize(gen_path, pair[:, 2])
    d0_check = np.linalg.norm(local[:, 1], axis=1)
    real_sub = np.stack([np.linalg.norm(_mic(pos[f, pair[:, 1].astype(int)] - pos[f, pair[:, 0].astype(int)], cell(box[f])), axis=1)
                         for f in (11, 13, 15, 20)], axis=1)
    np.savez_compressed(OUTDIR / f"{tid}.npz", pair=pair, y=y,
                        gen_dist=gen_dist.astype(np.float32),
                        coarse_dist=coarse_dist.astype(np.float32),
                        gen8=gen8, real_sub=real_sub.astype(np.float32))
    return dict(tid=tid, n=int(len(pair)), d0_max_err=float(np.abs(d0_check - pair[:, 2]).max()),
                y_rate=float(y.mean()), gen_mean=float(gen_dist.mean()), gen_std=float(gen_dist.std()),
                coarse_mean=float(coarse_dist.mean()), real_sub_mean=float(real_sub.mean()))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    reg = Registry(Workspace(DATA))
    model, center, scale = load_generator()
    print("generator input width:", len(center), flush=True)
    proteins = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
    results = []
    for protein in proteins:
        for replica in (1, 2, 3):
            tid = f"atlas-{protein}-analysis-R{replica}"
            if only and only not in tid:
                continue
            t0 = time.time()
            r = run_one(reg, protein, replica, model, center, scale)
            r["seconds"] = round(time.time() - t0, 1)
            results.append(r)
            print(json.dumps(r), flush=True)
    # Append to the repo-side summary so protein-by-protein progress is
    # committed (the work-dir npz files themselves are not committed).
    prior = []
    if GEN_SUMMARY.exists():
        prior = json.loads(GEN_SUMMARY.read_text())
    seen = {r["tid"] for r in prior}
    prior.extend(r for r in results if r["tid"] not in seen)
    GEN_SUMMARY.write_text(json.dumps(prior, indent=1))
    (OUTDIR / "adapter_summary.json").write_text(json.dumps(prior, indent=1))
    print("saved to", OUTDIR, "and", GEN_SUMMARY)


if __name__ == "__main__":
    main()
