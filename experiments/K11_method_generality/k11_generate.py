"""K11: regenerate generated-trajectory distance series for methods other
than proposal 22, on exactly the K05 candidate set (Doc D six proteins x
3 replicas, origin frame 10 = 1 ns, 3.5 < d(origin) <= 6.0 A), so the raw
series can be compared with the real trajectory without a predictor.

Methods (mechanistic implementations from the reference branch
research/tseries-models-and-results-20261121, tseries package, read-only
copies placed at K11_PKG, default /tmp/k11_pkg):

- idea 02 (physical integration): all-heavy-atom Langevin dynamics,
  tseries/hb_heavy_dynamics.py. The original features() pools the recorded
  distances into 8 summary values in memory; this adapter runs the same
  integration loop (same constants, same recording every 5 of 40 steps,
  3 replicas) and keeps the donor-acceptor distance at each recorded step.
- ideas 11-14 (elastic modes / fluctuations): tseries/hb_elastic_paths.py.
  The original features() builds four distance tracks and summarises them;
  this adapter rebuilds the same tracks (same rng consumption order:
  mode_paths -> local_paths(omega=0) -> local_paths(omega=.8)) and keeps
  the distances. 11 = low-frequency modes, 12 = response-focused modes,
  13 = local anisotropic fluctuation, 14 = 11 + 0.55 * fast local motion.
- idea 08 (rotamer transitions): tseries/hb_rotamer_paths.py. transitions()
  gives per-atom chi-rotation paths (8 steps each); the adapter rebuilds
  the same moved-position tensor as the original features() and keeps the
  donor-acceptor distance for every path of either endpoint atom.

Random seeds: the original experiment driver (hb_heavy_experiment.prepare)
created rng = default_rng([506166, replica, origin]) per origin group, with
a fresh rng per experiment group. K11 fixes origin = 10 and uses a fresh
rng seeded [506166, replica, 10] per method, matching the group-wise
execution of the original.

Boxes are handled per axis through tseries.data.cell (a 3-vector of box
lengths or a full 3x3 matrix), and all distances use the reference _mic
minimum-image implementation. No per-axis shortcut is taken.

Output: /tmp/k11_out/<tid>.npz per trajectory with the candidate pairs,
origin distances, and the generated future-distance arrays:
  heavy    (N, 3, 8)   replicas x recorded steps (origin not included)
  elas11..14 (N, 4, 8) branches x steps
  rot_ab, rot_ba (N, W, 8) rotamer paths of donor / acceptor atom,
             count_a, count_b (N,) number of paths per endpoint atom
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("K11_PKG", "/tmp/k11_pkg"))

import numpy as np
from minitopol import Workspace
from minitopol.registry import Registry

from tseries.data import topology, cell
from tseries.features import _mic
from tseries.hb_heavy_dynamics import bonded_terms, neighbor_pairs
from tseries.hb_elastic_paths import elastic_modes, mode_paths, local_paths
from tseries.hb_rotamer_paths import transitions

ORIGIN = 10
SEED = 506166
DATA = Path.home() / "workspace/gpt-model-check/data"
K05_NPZ = Path(__file__).resolve().parent.parent / "K05_generated_quality" / "generated_series.npz"
OUTDIR = Path("/tmp/k11_out")
OUTDIR.mkdir(exist_ok=True)
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")


# ---------------------------------------------------------------- idea 02
def gen_heavy(position, box, pair, atoms, rng, steps=40, replicas=3):
    """Same integration as hb_heavy_dynamics.features, recording distances.

    Returns (distances (replicas, steps//5, N), max_bond_shift)."""
    edges, rest, stiffness, mass, adjacency = bonded_terms(position, box, atoms)
    donor, acceptor = pair[:, :2].astype(np.intp).T
    n = len(position)
    excluded = np.sort(edges[:, 0] * n + edges[:, 1])
    distances, shifts = [], []
    dt, friction, kt = .025, .98, .596
    noise = np.sqrt((1 - friction ** 2) * kt / mass)[:, None]
    for _ in range(replicas):
        current = position.astype(np.float64, copy=True)
        velocity = rng.normal(size=current.shape) * np.sqrt(kt / mass)[:, None]
        reference = None
        rep = []
        for step in range(steps):
            if reference is None or np.max(np.linalg.norm(current - reference, axis=1)) > .4:
                pairs = neighbor_pairs(current, box, 3.9)
                repulsive = pairs[~np.isin(pairs[:, 0] * n + pairs[:, 1], excluded, kind="sort")]
                reference = current.copy()
            left, right = np.concatenate((edges, repulsive)).T
            delta = _mic(current[right] - current[left], box)
            length = np.maximum(np.linalg.norm(delta, axis=1), 1e-8)
            stretch = stiffness * (length[:len(edges)] - rest)
            overlap = np.minimum(0., length[len(edges):] - 2.75)
            reaction = delta * (np.r_[stretch, 28. * overlap] / length)[:, None]
            force = np.column_stack([
                np.bincount(left, weights=f, minlength=n) - np.bincount(right, weights=f, minlength=n)
                for f in reaction.T])
            velocity = friction * (velocity + dt * force / mass[:, None])
            velocity += noise * rng.normal(size=current.shape)
            current += dt * velocity
            if (step + 1) % 5:
                continue
            rep.append(np.linalg.norm(_mic(current[acceptor] - current[donor], box), axis=1))
            shifts.append(float(np.max(np.abs(length[:len(edges)] - rest))))
        distances.append(np.stack(rep))
    distance = np.stack(distances)  # (replicas, 8, N)
    if not np.isfinite(distance).all() or max(shifts) > 1.5:
        raise ValueError("Heavy-atom integration produced unstable covalent geometry")
    return distance.transpose(0, 2, 1), max(shifts)  # (replicas, N, 8) -> caller transposes


# ------------------------------------------------------------- ideas 11-14
def gen_elastic(position, box, pair, atoms, rng):
    """Same tracks as hb_elastic_paths.features; returns {11..14: (N, 4, 8)}."""
    residues = topology(atoms)[0]
    ca = position[[residue["CA"] for residue in residues]]
    owner = np.full(len(position), -1, dtype=np.intp)
    for i, residue in enumerate(residues):
        owner[list(residue.values())] = i
    a, b = pair[:, :2].astype(np.intp).T
    ia, ib = owner[a], owner[b]
    if np.any((ia < 0) | (ib < 0)):
        raise ValueError("Candidate endpoint is outside the elastic graph")
    owner = np.maximum(owner, 0)
    values, modes, left, right, blocks = elastic_modes(ca, box)
    if not len(values):
        raise ValueError("No nontrivial elastic modes")
    base = _mic(position[b] - position[a], box)
    difference = modes[ib] - modes[ia]
    signal = mode_paths(modes, values, rng)
    low = np.einsum("ncm,rtm->nrtc", difference, signal)
    response = np.einsum("ncm,nc->nm", difference, base / pair[:, 2, None])
    order = np.argsort(np.abs(response) / np.sqrt(values), axis=1)
    selected = np.zeros_like(response, dtype=bool)
    np.put_along_axis(selected, order[:, -min(3, len(values)):], True, axis=1)
    focused = np.einsum("ncm,rtm->nrtc", difference * selected[:, None, :], signal)
    local = local_paths(ca, position, owner, box, left, right, blocks, rng, omega=0.)
    fast = local_paths(ca, position, owner, box, left, right, blocks, rng, omega=.8)
    tracks = (low, focused, local[b] - local[a], low + .55 * (fast[b] - fast[a]))
    return {number: np.linalg.norm(base[:, None, None] + shift, axis=-1)
            for number, shift in enumerate(tracks, 11)}


# ----------------------------------------------------------------- idea 08
def gen_rotamer(position, box, pair, atoms):
    """Same moved-position tensor as hb_rotamer_paths.features.

    Returns (d_ab (N, W, 8), d_ba (N, W, 8), count_a, count_b) where W is
    the largest number of rotamer paths of any single atom in the protein.
    Atoms without a path keep the origin position in every slot, exactly
    as in the original features()."""
    paths = transitions(position, box, topology(atoms)[0])
    a, b = pair[:, :2].astype(np.intp).T
    width = max(1, max(map(len, paths)))
    moved = np.repeat(position[:, None, None, :], width, axis=1)
    moved = np.repeat(moved, 8, axis=2)
    count = np.array([len(path) for path in paths])
    for atom, path in enumerate(paths):
        if path:
            moved[atom, :len(path)] = path
    d_ab = np.linalg.norm(_mic(position[b, None, None] - moved[a], box), axis=-1)
    d_ba = np.linalg.norm(_mic(position[a, None, None] - moved[b], box), axis=-1)
    return d_ab, d_ba, count[a], count[b]


# -------------------------------------------------------------------- main
def load_candidates():
    with np.load(K05_NPZ) as z:
        return z["tid"].copy(), z["pair"].copy()


def run_one(reg, tid, pair_all, mask):
    atoms = reg.atoms(tid).to_pydict()
    blocks = list(reg.blocks(tid, start=0, stop=ORIGIN + 1, chunk_frames=ORIGIN + 1))
    pos = np.concatenate([b.positions for b in blocks], axis=0)
    box = np.concatenate([b.box for b in blocks], axis=0)
    position = np.asarray(pos[ORIGIN])
    bx = cell(box[ORIGIN])
    pair = pair_all[mask]
    replica = int(tid.rsplit("R", 1)[1])
    d0_check = np.linalg.norm(_mic(position[pair[:, 1].astype(int)] - position[pair[:, 0].astype(int)], bx), axis=1)
    info = {"tid": tid, "n": int(len(pair)),
            "d0_max_err": float(np.abs(d0_check - pair[:, 2]).max())}
    t0 = time.time()
    heavy, max_shift = gen_heavy(position, bx, pair, atoms,
                                 np.random.default_rng([SEED, replica, ORIGIN]))
    info["heavy_seconds"] = round(time.time() - t0, 1)
    info["heavy_max_bond_shift"] = max_shift
    t0 = time.time()
    elas = gen_elastic(position, bx, pair, atoms,
                       np.random.default_rng([SEED, replica, ORIGIN]))
    info["elastic_seconds"] = round(time.time() - t0, 1)
    t0 = time.time()
    d_ab, d_ba, count_a, count_b = gen_rotamer(position, bx, pair, atoms)
    info["rotamer_seconds"] = round(time.time() - t0, 1)
    info["rotamer_width"] = int(d_ab.shape[1])
    info["rotamer_frac_pairs_any_path"] = float(((count_a > 0) | (count_b > 0)).mean())
    np.savez_compressed(
        OUTDIR / f"{tid}.npz", pair=pair,
        heavy=heavy.transpose(1, 0, 2).astype(np.float32),  # (N, replicas, 8)
        elas11=elas[11].astype(np.float32), elas12=elas[12].astype(np.float32),
        elas13=elas[13].astype(np.float32), elas14=elas[14].astype(np.float32),
        rot_ab=d_ab.astype(np.float32), rot_ba=d_ba.astype(np.float32),
        count_a=count_a, count_b=count_b)
    return info


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    tids, pair_all = load_candidates()
    reg = Registry(Workspace(DATA))
    results = []
    for protein in PROTEINS:
        for replica in (1, 2, 3):
            tid = f"atlas-{protein}-analysis-R{replica}"
            if only and only not in tid:
                continue
            out_path = OUTDIR / f"{tid}.npz"
            if out_path.is_file() and not only:
                print("skip (exists)", tid, flush=True)
                continue
            info = run_one(reg, tid, pair_all, tids == tid)
            results.append(info)
            print(json.dumps(info), flush=True)
    (OUTDIR / "generate_summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
