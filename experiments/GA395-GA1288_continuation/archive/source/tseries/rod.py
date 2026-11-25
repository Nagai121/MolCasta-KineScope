"""Exhaustive CPU ROD expansion for the historical HB formation proxy."""
import numpy as np

from .features import _mic

HB_DISTANCE = 3.5


def expand(positions, box, context, atom_types, screening_distance=None, *, future=None,
           future_box=None, chunk_atoms=64):
    """Enumerate every unordered pair, then apply chemistry and origin-state rules.

    Eligible directions are evaluated independently for dual-role atoms.
    Future coordinates are used exclusively for measuring retention and labels.
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
            eligible = distance > HB_DISTANCE
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
