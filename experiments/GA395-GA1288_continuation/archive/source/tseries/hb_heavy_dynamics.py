"""CPU Langevin dynamics for a reduced protein-heavy-atom force field."""
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from .features import _mic


def neighbor_pairs(position, box, cutoff):
    if box.ndim == 1:
        tree = cKDTree(position % box, boxsize=box)
        return tree.query_pairs(cutoff, output_type="ndarray")
    inverse = np.linalg.inv(box)
    wrapped = (position @ inverse) % 1.
    radius = cutoff * np.linalg.norm(inverse, ord=2)
    pairs = cKDTree(wrapped, boxsize=1.).query_pairs(radius, output_type="ndarray")
    delta = _mic(position[pairs[:, 1]] - position[pairs[:, 0]], box)
    return pairs[np.linalg.norm(delta, axis=1) <= cutoff]


def bonded_terms(position, box, atoms):
    """Infer intraresidue bonds, peptide bonds and disulfides from the origin."""
    element = np.char.upper(np.asarray(atoms["element"], dtype=str))
    residue = np.asarray(atoms["residue_index"])
    names = np.asarray(atoms["name"])
    radii = np.array([dict(C=.76, N=.71, O=.66, S=1.05, P=1.07, SE=1.20).get(e, .76)
                      for e in element])
    mass = np.array([dict(C=12., N=14., O=16., S=32., P=31., SE=79.).get(e, 12.)
                     for e in element])
    left, right = neighbor_pairs(position, box, 2.7).T
    distance = np.linalg.norm(_mic(position[right] - position[left], box), axis=1)
    peptide = (np.abs(residue[left] - residue[right]) == 1) & (
        ((names[left] == "C") & (names[right] == "N")) |
        ((names[right] == "C") & (names[left] == "N")))
    disulfide = (names[left] == "SG") & (names[right] == "SG")
    keep = ((residue[left] == residue[right]) | peptide | disulfide)
    keep &= (distance > .85) & (distance <= radii[left] + radii[right] + .32)
    bonds = list(zip(left[keep], right[keep]))
    adjacency = [set() for _ in position]
    for a, b in bonds:
        adjacency[a].add(b)
        adjacency[b].add(a)
    angles = sorted({edge for adjacent in adjacency for edge in combinations(sorted(adjacent), 2)}
                    - set(bonds))
    edges = np.asarray(bonds + angles, dtype=np.intp).reshape(-1, 2)
    if not len(bonds):
        raise ValueError("No covalent bonds inferred")
    rest = np.linalg.norm(_mic(position[edges[:, 1]] - position[edges[:, 0]], box), axis=1)
    stiffness = np.r_[np.full(len(bonds), 185.), np.full(len(angles), 42.)]
    return edges, rest, stiffness, mass, adjacency


def place_hydrogen(position, box, donors, adjacency, partners, initial):
    """Place a virtual donor hydrogen using the current local heavy geometry."""
    vectors = _mic(position[partners] - position[donors, None], box)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=2, keepdims=True), 1e-9)
    direction = -(vectors * adjacency[:, :, None]).sum(axis=1)
    length = np.linalg.norm(direction, axis=1, keepdims=True)
    # Nearly symmetric or undefined directions have no reconstructed hydrogen.
    reliable = (length[:, 0] > 1e-4) & initial
    return position[donors] + direction / np.maximum(length, 1e-9), reliable


def features(position, box, pair, atoms, rng, *, steps=40, replicas=3):
    edges, rest, stiffness, mass, adjacency = bonded_terms(position, box, atoms)
    donor, acceptor = pair[:, :2].astype(np.intp).T
    donors, index = np.unique(donor, return_inverse=True)
    sites = [sorted(adjacency[d]) for d in donors]
    width = max(1, max(map(len, sites), default=0))
    partners = np.repeat(donors[:, None], width, axis=1)
    available = np.zeros_like(partners, dtype=bool)
    for row, sites_row in enumerate(sites):
        partners[row, :len(sites_row)] = sites_row
        available[row, :len(sites_row)] = True
    # A single bisector is appropriate only for donors with at least two anchors.
    initial = available.sum(axis=1) >= 2
    n = len(position)
    excluded = np.sort(edges[:, 0] * n + edges[:, 1])
    distances, geometry, shifts = [], [], []
    dt, friction, kt = .025, .98, .596
    noise = np.sqrt((1 - friction ** 2) * kt / mass)[:, None]
    for _ in range(replicas):
        current = position.astype(np.float64, copy=True)
        velocity = rng.normal(size=current.shape) * np.sqrt(kt / mass)[:, None]
        reference = None
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
            hydrogen, valid = place_hydrogen(current, box, donors, available, partners, initial)
            ha = _mic(current[acceptor] - hydrogen[index], box)
            dh = _mic(hydrogen[index] - current[donor], box)
            d = np.linalg.norm(_mic(current[acceptor] - current[donor], box), axis=1)
            angle = np.sum(ha * dh, axis=1) / np.maximum(np.linalg.norm(ha, axis=1), 1e-8)
            distances.append(d)
            geometry.append(valid[index] & (np.linalg.norm(ha, axis=1) <= 2.5) & (angle >= .5))
            shifts.append(float(np.max(np.abs(length[:len(edges)] - rest))))
    if not np.isfinite(distances).all() or max(shifts) > 1.5:
        raise ValueError("Heavy-atom integration produced unstable covalent geometry")
    distance = np.stack(distances)
    change = distance - pair[:, 2]
    return np.column_stack((change.mean(axis=0), change.std(axis=0), change.min(axis=0),
                            change.max(axis=0), (distance <= 3.5).mean(axis=0),
                            (distance <= 4.).mean(axis=0), np.mean(geometry, axis=0),
                            initial[index])).astype(np.float32)
