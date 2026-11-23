"""K81 feedability check (no generator involved): for each K53
salt-bridge dissociation candidate (all 88 proteins of the K15
expanded set), find the representative charged heavy-atom pair (the
pair attaining the minimum distance at the origin, frame 10) and
check whether it is a (donor, acceptor) pair in either orientation,
using the tseries topology context. Idea 22's descriptors are
defined only for (donor, acceptor) pairs, so only feedable
candidates can be fed to the generator (the same restriction as
K14/K35/K76).

Charged atoms (K53's definition): positive LYS(NZ)/ARG(NE,NH1,NH2),
negative ASP(OD1,OD2)/GLU(OE1,OE2). Distance = min over charged
heavy-atom pairs of the residue pair.

Data: K53 cache (res_i, res_j per trajectory) + atoms.parquet /
A.zarr (same path as gen_k65.py: the original 6 proteins from
gpt-model-check/data/trajectories, the other 82 from
k15_atlas/<protein>/trajectories). The atoms dict for topology()
is built from atoms.parquet (it carries an element column).
"""
import sys, json
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
import minitopol  # registers the minitopol_agx3 zarr codec
from pathlib import Path
import numpy as np, pandas as pd, zarr
from tseries.data import topology, cell
from tseries.features import _mic

GPT_BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
ATLAS_WS = Path("/home/hatch/workspace/kinescope/work/k15_atlas")
CACHE = Path("/home/hatch/workspace/kinescope/work/k53_cache")
_SPLIT = json.loads(Path("/home/hatch/workspace/kinescope/repo/experiments/K15_saltbridge_expansion/split.json").read_text())
PROTS = tuple(_SPLIT["orig_train"]) + tuple(_SPLIT["train_additions"]) + tuple(_SPLIT["orig_test"]) + tuple(_SPLIT["test_additions"])


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


def main():
    total, feedable = 0, 0
    for protein in PROTS:
        for replica in (1, 2, 3):
            tid = f"atlas-{protein}-analysis-R{replica}"
            d = trajdir(protein, replica)
            df = pd.read_parquet(d / "atoms.parquet")
            atoms = {"residue_index": df["residue_index"].values,
                     "name": df["name"].values,
                     "residue_name": df["residue_name"].values,
                     "element": df["element"].values}
            maps_t, names_t, context, _ = topology(atoms)
            maps_l, names_l = layout(df)
            assert len(maps_t) == len(maps_l), tid
            assert all(maps_t[i]["CA"] == maps_l[i]["CA"] for i in range(len(maps_t))), tid
            assert names_t == names_l, tid
            d_at, a_at = context[5], context[9]
            d_set, a_set = set(int(x) for x in d_at), set(int(x) for x in a_at)
            zz = np.load(CACHE / f"{protein}_R{replica}_salt_diss.npz")
            ri, rj = zz["res_i"], zz["res_j"]
            if len(ri) == 0:
                # No candidates (e.g. 4cv4_A R2 has an all-zero box
                # recorded) -- same handling as K20/K65.
                print(tid, "candidates 0 feedable 0", flush=True)
                continue
            z = zarr.open_group(str(d / "A.zarr"), mode='r')
            pos = z['positions'][:].astype(np.float64)
            box = z['box'][:].astype(np.float64)
            pos10 = np.asarray(pos[10]); boxv = cell(box[10])
            origin_cache = zz["hist"][:, -1].astype(np.float64)
            n_feed = 0
            for zi, (i, j) in enumerate(zip(ri, rj)):
                ia = charged_atoms(maps_t, names_t, int(i))
                ib = charged_atoms(maps_t, names_t, int(j))
                assert len(ia) > 0 and len(ib) > 0, (tid, i, j)
                dist = np.linalg.norm(_mic(pos10[np.repeat(ia, len(ib))] - pos10[np.tile(ib, len(ia))], boxv), axis=1)
                kmin = int(np.argmin(dist))
                ra, rb = int(ia[kmin // len(ib)]), int(ib[kmin % len(ib)])
                assert abs(dist[kmin] - origin_cache[zi]) < 1e-5, (tid, zi, dist[kmin], origin_cache[zi])
                if (ra in d_set and rb in a_set) or (rb in d_set and ra in a_set):
                    n_feed += 1
            total += len(ri); feedable += n_feed
            print(tid, "candidates", len(ri), "feedable", n_feed, flush=True)
    print("TOTAL candidates", total, "feedable", feedable, "fraction %.3f" % (feedable / total))


if __name__ == "__main__":
    main()
