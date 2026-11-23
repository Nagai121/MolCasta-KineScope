"""K76 feedability check (no generator involved): for each K50
contact-dissociation candidate, find the representative heavy-atom
pair (the pair attaining the minimum distance at the origin, frame
10) and check whether it is a (donor, acceptor) pair in either
orientation, using the tseries topology context. Idea 22's
descriptors are defined only for (donor, acceptor) pairs, so only
feedable candidates can be fed to the generator (the same
restriction as K14/K35 for contact formation).

Data: K50 cache (res_i, res_j per trajectory) + the minitopol
Registry (same path as gen_k54.py) for atoms/positions/topology.
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
from pathlib import Path
import numpy as np
from minitopol import Workspace
from minitopol.registry import Registry
from tseries.data import topology, cell
from tseries.features import _mic

DATA = Path.home() / "workspace/gpt-model-check/data"
CACHE = Path("/home/hatch/workspace/kinescope/work/k50_cache")
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")


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


def main():
    import pandas as pd
    reg = Registry(Workspace(DATA))
    total, feedable = 0, 0
    per_traj = {}
    for protein in PROTEINS:
        for replica in (1, 2, 3):
            tid = f"atlas-{protein}-analysis-R{replica}"
            atoms = reg.atoms(tid).to_pydict()
            topo = topology(atoms)
            maps_t, _, context, _ = topo
            d_at, a_at = context[5], context[9]
            d_set, a_set = set(int(x) for x in d_at), set(int(x) for x in a_at)
            blocks = list(reg.blocks(tid, start=0, stop=21, chunk_frames=21))
            pos = np.concatenate([b.positions for b in blocks], axis=0)
            box = np.concatenate([b.box for b in blocks], axis=0)
            pos10 = np.asarray(pos[10]); boxv = cell(box[10])
            # residue maps from the same atoms dict (topology's own order)
            maps = maps_t
            # K50 cache residue indices are in layout() order from
            # atoms.parquet; verify the residue order matches topology's
            df = pd.read_parquet(Path.home() / f"workspace/gpt-model-check/data/trajectories/{tid}/atoms.parquet")
            lay_maps, _ = layout(df)
            assert len(lay_maps) == len(maps), (tid, len(lay_maps), len(maps))
            assert all(lay_maps[i]["CA"] == maps[i]["CA"] for i in range(len(maps))), tid
            z = np.load(CACHE / f"{protein}_R{replica}_contact_diss.npz")
            ri, rj = z["res_i"], z["res_j"]
            n_feed = 0
            for i, j in zip(ri, rj):
                ia = [v for k, v in maps[int(i)].items()
                      if not k.startswith("H") and not k[0].isdigit()]
                ib = [v for k, v in maps[int(j)].items()
                      if not k.startswith("H") and not k[0].isdigit()]
                ia = np.array(ia, dtype=int); ib = np.array(ib, dtype=int)
                d = np.linalg.norm(_mic(pos10[np.repeat(ia, len(ib))] - pos10[np.tile(ib, len(ia))], boxv), axis=1)
                kmin = int(np.argmin(d))
                ra, rb = int(ia[kmin // len(ib)]), int(ib[kmin % len(ib)])
                if (ra in d_set and rb in a_set) or (rb in d_set and ra in a_set):
                    n_feed += 1
            total += len(ri); feedable += n_feed
            per_traj[tid] = (int(len(ri)), int(n_feed))
            print(tid, "candidates", len(ri), "feedable", n_feed, flush=True)
    print("TOTAL candidates", total, "feedable", feedable, "fraction %.3f" % (feedable / total))


if __name__ == "__main__":
    main()
