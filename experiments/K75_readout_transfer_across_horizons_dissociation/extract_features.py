"""K75 preparation: extract K06(b)-style features for HB DISSOCIATION.

K06(b) saved only formation candidates (origin distance in (3.5, 6.0] A,
label = distance <= 3.5 A at the horizon). Dissociation is the mirror:
candidates are pairs formed at the origin (distance <= 3.5 A), and the
label is distance > 3.5 A at the horizon. This script mirrors K06(b)
score_b.py's one_replica exactly, except for the candidate rule:

  - the donor-acceptor enumeration in tseries/rod.py expand() applies
    `eligible = distance > 3.5 A` (the formation rule). Here it is flipped
    to `distance <= 3.5 A`; every other line of expand(), frame(),
    predict_snapshot() and one_replica() is copied unchanged, so the
    static columns (frozen T1I2006 probability, origin distance, CA-CA
    distance, donor/acceptor kinds), the history10 formula and the frozen
    model are identical to the formation pipeline.
  - the frozen probability is the same model's P(formed at the horizon)
    for the pair; for a dissociation candidate it is roughly the complement
    of the dissociation probability, and the correction model learns how
    to use it from the calibration proteins.

Trajectory sources, in order (K22's pattern):
 1. K15's workspace (~/workspace/kinescope/work/k15_atlas), which holds
    the same 108 ATLAS proteins. Read-only: nothing is acquired or
    deleted there.
 2. For proteins missing there, acquire into a K75 workspace
    (~/workspace/kinescope/work/k75_atlas) and delete after scoring.
For the 6 weak-base proteins (verification against the K46 cache),
copied MiniTopol zarrs are placed in the K75 workspace first (see
verify_k46.py); acquire() is then a no-op for them.

Run with cwd=~/workspace/k06b/run (prepare_model uses relative paths
"runs/..." from there, as score_b.py did). Output:
~/workspace/kinescope/work/k75_cache/<protein>.npz with keys
r{1,2,3}_pair/_static/_history/_frozen/_label (dissociation labels),
same layout as k06b_out. Checkpointed: skips proteins whose output
npz exists.
"""
import sys, os, shutil, time
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg_doca")
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from tseries.data import registry, trajectory, topology, cell, acquire, inputs
from tseries.t1i2006_initial_history_100 import prepare_model, HORIZONS, TARGETS
from tseries.features import _mic, structural, feature_graph, enriched_features, prepare
from tseries.rod import HB_DISTANCE
from tseries.hb_single_structure import predict
from tseries.hb_ensemble_calibration import members
from tseries.models import load_cpu
from tseries.hb_local_geometry import origin_features
from tseries.hb_directional import directional_features
from tseries.hb_occupied import occupied
from tseries.hb_cloud import cloud

K15_ROOT = Path("/home/hatch/workspace/kinescope/work/k15_atlas")
OWN_ROOT = Path(os.environ.get("K75_OWN_ROOT", "/home/hatch/workspace/kinescope/work/k75_atlas"))
OUTD = Path("/home/hatch/workspace/kinescope/work/k75_cache")
OUTD.mkdir(parents=True, exist_ok=True)
MODEL = None


def expand_mirror(positions, box, context, atom_types, screening_distance=None, chunk_atoms=64):
    """Copy of tseries/rod.py expand() with the eligibility flipped:
    dissociation candidates are pairs at distance <= 3.5 A at the origin.
    Every other line is unchanged."""
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
            eligible = distance <= HB_DISTANCE
            counts["eligible"] += np.count_nonzero(eligible)
            retained = eligible if screening_distance is None else (
                eligible & (distance <= screening_distance)
            )
            a, b = a[retained], b[retained]
            vectors.append(np.column_stack((a, b, distance[retained], delta[retained],
                                            atom_types[a], atom_types[b])).astype(np.float32))
            donor_indices.append(i[retained])
            acceptor_indices.append(j[retained])
            counts["retained"] += len(a)
    assert counts["expanded"] == n_atoms * (n_atoms - 1) // 2
    return (np.concatenate(vectors), np.concatenate(donor_indices),
            np.concatenate(acceptor_indices), None,
            {key: int(value) for key, value in counts.items()})


def frame_mirror(positions, box, topo, screening_distance):
    """Copy of tseries/data.py frame() using expand_mirror."""
    maps, residue_names, context, types = topo
    pair, ii, jj, y, counts = expand_mirror(positions, box, context, types, screening_distance)
    base = structural(positions, box, maps, residue_names, context, ii, jj)
    graph = feature_graph(base, pair[:, 0], pair[:, 1], np.zeros(len(pair)), len(positions))
    features = enriched_features(np.column_stack((base, graph)))
    if not np.isfinite(features).all():
        raise ValueError("Non-finite structural features")
    return pair, features, y, counts


def predict_snapshot_mirror(pos, box, topo, components, old_meta, old_model):
    """Copy of t1i2006_initial_history_100.predict_snapshot() using
    frame_mirror. Returns the screened dissociation pairs, base features,
    frozen probabilities and counts."""
    with threadpool_limits(limits=1, user_api="blas"):
        pair, base, _, counts = frame_mirror(pos, box, topo, 8.0)
        keep = pair[:, 2] <= 6.0
        pair, base = pair[keep], base[keep]
        if not len(pair):
            raise RuntimeError("No screened ROD HB candidates")
        part = dict(pair=pair, base=base, ends=np.asarray([len(pair)]))
        part["old"] = old_model(inputs(pair, base, old_meta["width"], old_meta["representation"]))
        part["local"] = origin_features(pos, box, pair, base, topo)
        part["directional"] = directional_features(pos, box, pair, topo[0])
        part["cloud"] = cloud(pos, box, pair, topo[3])
        part["occupied"] = occupied(pos, box, pair, topo)
        p = np.zeros(len(pair), dtype=np.float64)
        for w, folder, record in components:
            p += float(w) * predict([part], folder, record)[0]
    if not (np.isfinite(p).all() and np.all((0 <= p) & (p <= 1))):
        raise AssertionError("Invalid frozen T1I2006 probabilities")
    return pair, base, p, counts


def one_replica(root, protein, replica):
    reg = registry(root, protein)
    tid = trajectory(protein, replica)
    blocks = list(reg.blocks(tid, start=0, stop=911, chunk_frames=128))
    pos = np.concatenate([b.positions for b in blocks], axis=0)
    boxes = np.concatenate([b.box for b in blocks], axis=0)
    times = np.concatenate([b.time for b in blocks], axis=0)
    assert len(pos) == 911, (tid, len(pos))
    assert np.allclose(times[:11] - times[0], np.arange(11) * 100, rtol=0, atol=0.01), "0..1ns grid mismatch"
    assert np.allclose(times[[TARGETS[h] for h in HORIZONS]] - times[10],
                       np.array(HORIZONS) * 1000, rtol=0, atol=.01), "target grid mismatch"
    topo = topology(reg.atoms(tid).to_pydict())
    pair, base, frozen, counts = predict_snapshot_mirror(pos[10], cell(boxes[10]), topo, *MODEL)
    a, b = pair[:, :2].astype(np.intp).T
    series = np.stack([np.linalg.norm(_mic(pos[k, b] - pos[k, a], cell(boxes[k])), axis=1)
                       for k in range(11)], axis=0)
    np.testing.assert_allclose(series[-1], pair[:, 2], atol=2e-4, rtol=0)
    assert np.all(pair[:, 2] <= 3.5 + 1e-6), "dissociation candidates must be formed at origin"
    history = np.column_stack((series[0], series.mean(axis=0), series.std(axis=0), series.min(axis=0),
                               series.max(axis=0), series[-1] - series[0], series[-1] - series[-2],
                               (series <= 3.5).mean(axis=0), (series <= 4.5).mean(axis=0),
                               (np.diff(series, axis=0) > 0).mean(axis=0))).astype(np.float32)
    static = np.column_stack((frozen, pair[:, 2], base[:, 1], base[:, 3], base[:, 5])).astype(np.float32)
    label = np.column_stack([np.linalg.norm(_mic(pos[TARGETS[h], b] - pos[TARGETS[h], a],
                                                 cell(boxes[TARGETS[h]])), axis=1) > 3.5
                             for h in HORIZONS]).astype(np.uint8)
    return dict(pair=pair, static=static, history=history, frozen=frozen, label=label)


def has_all(root, protein):
    reg = registry(root, protein)
    return all((Path(reg.workspace.trajectory_path(trajectory(protein, r))) / "A.zarr").is_dir()
               for r in (1, 2, 3))


def main():
    global MODEL
    proteins = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    MODEL = prepare_model()
    for p in proteins:
        out = OUTD / f"{p}.npz"
        if out.is_file():
            print("SKIP", p, flush=True)
            continue
        t0 = time.time()
        own = False
        last_err = None
        try:
            roots = []
            if has_all(K15_ROOT, p):
                roots.append(K15_ROOT)
            roots.append(OWN_ROOT)
            reps = None
            for root in roots:
                try:
                    if root is OWN_ROOT:
                        acquire(OWN_ROOT, p)
                        own = True
                    reps = [one_replica(root, p, r) for r in (1, 2, 3)]
                    break
                except Exception as e:
                    last_err = e
                    reps = None
                    print("SOURCE FAILED", p, root.name, type(e).__name__, str(e)[:120], flush=True)
            if reps is None:
                raise last_err
            np.savez_compressed(out, **{f"r{r}_{k}": v for r, rep in enumerate(reps, 1)
                                        for k, v in rep.items()})
            print("SCORED", p, "dissociation candidates", sum(len(r["label"]) for r in reps),
                  "sec", round(time.time() - t0, 1), flush=True)
        except Exception as e:
            print("FAILED", p, type(e).__name__, str(e)[:300], flush=True)
        finally:
            if own:
                shutil.rmtree(OWN_ROOT / p, ignore_errors=True)


if __name__ == "__main__":
    main()
