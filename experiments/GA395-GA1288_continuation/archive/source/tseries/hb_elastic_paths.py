"""Origin-only elastic and packing-dependent reduced-coordinate trajectories."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from .data import topology
from .features import _mic
from .hb_heavy_dynamics import neighbor_pairs
from .hb_trajectory_screen import summary


def elastic_modes(ca, box):
    left, right = neighbor_pairs(ca, box, 10.).T
    relative = _mic(ca[right] - ca[left], box)
    norm = np.linalg.norm(relative, axis=1)
    unit = relative / np.maximum(norm[:, None], 1e-8)
    springs = np.where(norm < 4.4, 3., 1.) * (4. / np.maximum(norm, 3.))**2
    blocks = springs[:, None, None] * unit[:, :, None] * unit[:, None, :]
    dims = np.arange(3)
    a = 3 * left[:, None] + dims
    b = 3 * right[:, None] + dims
    n = 3 * len(ca)
    row = np.concatenate([np.broadcast_to(x[:, :, None], blocks.shape).ravel()
                          for x in (a, b, a, b)])
    col = np.concatenate([np.broadcast_to(x[:, None, :], blocks.shape).ravel()
                          for x in (a, b, b, a)])
    hessian = coo_matrix((np.stack((blocks, blocks, -blocks, -blocks)).ravel(), (row, col)),
                         shape=(n, n)).tocsc()
    if n <= 72:
        values, vectors = np.linalg.eigh(hessian.toarray())
    else:
        values, vectors = eigsh(hessian, k=min(24, n-1), sigma=-.001, which="LM",
                                v0=np.random.default_rng(52).normal(size=n))
    keep = np.flatnonzero(values > 1e-5)[:10]
    return values[keep], vectors[:, keep].reshape(len(ca), 3, -1), left, right, blocks


def mode_paths(modes, values, rng):
    # Time is a reduced simulation coordinate. Frequencies are relative to the first mode.
    scale = np.sqrt(values[0] / values)
    frequency = np.minimum(4., 1.2 / scale)
    time = np.linspace(0., 1., 9)[1:]
    phase = rng.uniform(0., 2.*np.pi, size=(4, len(values)))
    amplitude = rng.normal(0., 1., size=phase.shape) * scale
    signal = amplitude[:, None] * (np.sin(phase[:, None]+time[None, :, None]*frequency) -
                                    np.sin(phase[:, None]))
    normalization = np.sqrt(np.mean(np.sum(modes**2, axis=1)))
    return .21 * signal / max(normalization, 1e-6)


def local_paths(ca, position, owner, box, left, right, blocks, rng, omega):
    stiffness = np.zeros((len(ca), 3, 3))
    np.add.at(stiffness, left, blocks)
    np.add.at(stiffness, right, blocks)
    stiffness += np.eye(3)
    eigen, axes = np.linalg.eigh(stiffness)
    decay = np.exp(-.2 * eigen)
    width = .3 * np.sqrt((1.-decay**2)/eigen)
    state = np.zeros((4, len(ca), 3))
    angle = np.zeros_like(state)
    rel = _mic(position - ca[owner], box)
    paths = []
    for _ in range(8):
        state = decay * state + width * rng.normal(size=state.shape)
        angle = .35 * angle + omega * width * rng.normal(size=angle.shape)
        shift = np.einsum("nij,rnj->rni", axes, state)
        angular = np.einsum("nij,rnj->rni", axes, angle)
        paths.append(shift[:, owner] + np.cross(angular[:, owner], rel))
    return np.asarray(paths).transpose(2, 1, 0, 3)


def adaptive_contacts(ca, box, rng):
    """Let springs enter or leave the coarse graph as their distances change."""
    frames = []
    fixed_left, fixed_right = neighbor_pairs(ca, box, 4.4).T
    fixed_rest = np.linalg.norm(_mic(ca[fixed_right]-ca[fixed_left], box), axis=1)
    n = len(ca)
    for _ in range(4):
        x = ca.copy()
        v = rng.normal(0., .19, x.shape)
        path = []
        for step in range(24):
            a, b = neighbor_pairs(x, box, 9.5).T
            old = np.linalg.norm(_mic(ca[b]-ca[a], box), axis=1)
            keep = ~np.isin(a*n+b, fixed_left*n+fixed_right)
            a, b, old = a[keep], b[keep], old[keep]
            left = np.r_[fixed_left, a]
            right = np.r_[fixed_right, b]
            rest = np.r_[fixed_rest, np.minimum(old, 9.5)]
            delta = _mic(x[right]-x[left], box)
            d = np.maximum(np.linalg.norm(delta, axis=1), 1e-8)
            strength = np.r_[np.full(len(fixed_left), 8.), np.clip((9.5-d[len(fixed_left):])/1.2, 0., 1.)]
            reaction = strength[:, None]*(d-rest)[:, None]*delta/d[:, None]
            force = np.column_stack([np.bincount(left, q, n)-np.bincount(right, q, n) for q in reaction.T])
            v = .92*v + .11*force + rng.normal(0., .07, v.shape)
            x += .11*v
            if step % 3 == 2:
                path.append(x-ca)
        frames.append(path)
    return np.asarray(frames).transpose(2, 0, 1, 3)


def features(position, box, pair, atoms, rng):
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
    base = _mic(position[b]-position[a], box)
    difference = modes[ib] - modes[ia]
    signal = mode_paths(modes, values, rng)
    low = np.einsum("ncm,rtm->nrtc", difference, signal)
    response = np.einsum("ncm,nc->nm", difference, base / pair[:, 2, None])
    order = np.argsort(np.abs(response) / np.sqrt(values), axis=1)
    selected = np.zeros_like(response, dtype=bool)
    np.put_along_axis(selected, order[:, -min(3, len(values)):], True, axis=1)
    focused = np.einsum("ncm,rtm->nrtc", difference*selected[:, None, :], signal)
    local = local_paths(ca, position, owner, box, left, right, blocks, rng, omega=0.)
    fast = local_paths(ca, position, owner, box, left, right, blocks, rng, omega=.8)
    adaptive = adaptive_contacts(ca, box, rng)
    trajectories = (low, focused, local[b]-local[a], low + .55*(fast[b]-fast[a]),
                    adaptive[ib]-adaptive[ia])
    return {number: summary(np.linalg.norm(base[:, None, None] + shift, axis=-1), pair[:, 2])
            for number, shift in enumerate(trajectories, 11)}
