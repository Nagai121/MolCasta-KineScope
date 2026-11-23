"""G01 prep: build per-candidate atom clouds from the origin frame (1 ns).

For every DocD HB-formation candidate (same cache as K05/K08/K23/K40),
take the origin frame (frame 10) coordinates only and collect the atoms
within RADIUS of the donor or the acceptor atom (minimum image, per-axis
box handling, same convention as base_reproduction/extract.py).

Per-atom features (16):
  relD (3): MIC vector from donor atom to this atom
  relA (3): MIC vector from acceptor atom to this atom
  distD, distA: MIC distances to donor / acceptor
  element one-hot (C, N, O, S)
  isD, isA: this atom is the donor / acceptor atom
  sameResD, sameResA: same residue_index as donor / acceptor atom
Plus per-candidate: atom-name indices (for the name-embedding variant),
pair element one-hots (donor, acceptor), origin distance, target
(real mean of the 11-point history series, from the shared cache).

Output: ~/workspace/kinescope/work/g01/<protein>_R<replica>.npz
Sanity: MIC donor-acceptor distance at frame 10 must equal hist[-1].
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import minitopol  # noqa: F401  (registers the MiniTopol zarr codec)

import numpy as np
import pandas as pd
import zarr

BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path("/home/hatch/workspace/kinescope/work/g01")
OUT.mkdir(parents=True, exist_ok=True)
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
RADIUS = 10.0
CAP = 384
ELEMENTS = ("C", "N", "O", "S")
ORIGIN_FRAME = 10


def mic_vec(v, box):
    box = np.asarray(box, dtype=np.float64)
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-5):
        lengths = np.diag(box)
        return v - lengths * np.rint(v / lengths)
    inv = np.linalg.inv(box)
    fr = v @ inv
    fr -= np.rint(fr)
    return fr @ box


# global atom-name vocabulary over all trajectories
vocab = {}
for p in PROTEINS:
    df = pd.read_parquet(BASE / f"atlas-{p}-analysis-R1" / "atoms.parquet")
    for nm in df["name"].astype(str):
        if nm not in vocab:
            vocab[nm] = len(vocab)
json.dump(vocab, open(OUT / "name_vocab.json", "w"))
print("name vocab size", len(vocab), flush=True)

cloud_sizes = []
n_capped = 0
tot_train = tot_test = 0
for p in PROTEINS:
    for r in (1, 2, 3):
        tid = f"atlas-{p}-analysis-R{r}"
        d = BASE / tid
        df = pd.read_parquet(d / "atoms.parquet")
        z = zarr.open_group(str(d / "A.zarr"), mode="r")
        pos = z["positions"][ORIGIN_FRAME].astype(np.float64)
        box = z["box"][ORIGIN_FRAME].astype(np.float64)
        elem = df["element"].astype(str).values
        elem_idx = np.array([ELEMENTS.index(e) for e in elem], dtype=np.int64)
        name_idx_all = np.array([vocab[nm] for nm in df["name"].astype(str)], dtype=np.int64)
        res_of_atom = df["residue_index"].values.astype(np.int64)

        with np.load(CACHE / f"{p}_R{r}.npz") as zz:
            cache = {k: zz[k] for k in zz.files}
        donors = cache["donor_atom"]
        acceptors = cache["acceptor_atom"]
        hist = cache["hist"]
        target = hist.mean(axis=0)
        n = len(donors)

        # sanity: MIC distance at the origin frame equals hist[-1]
        vda = mic_vec(pos[acceptors] - pos[donors], box)
        dda = np.linalg.norm(vda, axis=1)
        max_dev = float(np.abs(dda - hist[-1]).max())
        assert max_dev < 1e-6, (tid, max_dev)

        feats = np.zeros((n, CAP, 16), dtype=np.float32)
        names = np.zeros((n, CAP), dtype=np.int64)
        mask = np.zeros((n, CAP), dtype=bool)
        sizes = np.zeros(n, dtype=np.int64)
        for c in range(n):
            D, A = int(donors[c]), int(acceptors[c])
            rel_d = mic_vec(pos - pos[D], box)
            rel_a = mic_vec(pos - pos[A], box)
            dist_d = np.linalg.norm(rel_d, axis=1)
            dist_a = np.linalg.norm(rel_a, axis=1)
            sel = np.flatnonzero((dist_d <= RADIUS) | (dist_a <= RADIUS))
            order = np.argsort(np.minimum(dist_d[sel], dist_a[sel]), kind="stable")
            sel = sel[order]
            if len(sel) > CAP:
                sel = sel[:CAP]
                n_capped += 1
            k = len(sel)
            sizes[c] = k
            cloud_sizes.append(k)
            f = np.zeros((k, 16), dtype=np.float32)
            f[:, 0:3] = rel_d[sel]
            f[:, 3:6] = rel_a[sel]
            f[:, 6] = dist_d[sel]
            f[:, 7] = dist_a[sel]
            f[np.arange(k), 8 + elem_idx[sel]] = 1.0
            f[:, 12] = (sel == D)
            f[:, 13] = (sel == A)
            f[:, 14] = (res_of_atom[sel] == res_of_atom[D])
            f[:, 15] = (res_of_atom[sel] == res_of_atom[A])
            feats[c, :k] = f
            names[c, :k] = name_idx_all[sel]
            mask[c, :k] = True

        pair_elem = np.zeros((n, 8), dtype=np.float32)
        pair_elem[np.arange(n), elem_idx[donors]] = 1.0
        pair_elem[np.arange(n), 4 + elem_idx[acceptors]] = 1.0
        np.savez_compressed(
            OUT / f"{p}_R{r}.npz",
            feats=feats, names=names, mask=mask, sizes=sizes,
            target=target.astype(np.float64),
            X=cache["X"], hist=cache["hist"], y=cache["y"],
            donor_atom=donors, acceptor_atom=acceptors,
            pair_elem=pair_elem, dist_da=dda.astype(np.float64),
            protein=p, replica=r,
        )
        if p in ("1mix_A", "2qt4_A"):
            tot_test += n
        else:
            tot_train += n
        print(tid, "cand", n, "cloud mean", float(sizes.mean()),
              "max", int(sizes.max()), flush=True)

cs = np.array(cloud_sizes)
print("total train", tot_train, "test", tot_test)
print("cloud size: mean %.1f p50 %d p99 %d max %d; capped candidates %d"
      % (cs.mean(), np.percentile(cs, 50), np.percentile(cs, 99), cs.max(), n_capped))
