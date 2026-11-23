"""K74 build: training rows for retraining #22's correction head with a
mean-distance objective. Teacher 8 proteins, DocD HB candidates at origin
frame 10. Per trajectory saves descriptors (110), coarse path (4,5,3),
local sites (5,3), real 11-point history and its mean. No generator is
loaded here (no checkpoint needed for the build)."""
import sys, json, time, shutil
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
from pathlib import Path
import numpy as np
from tseries.data import frame, topology, cell, registry, acquire
from tseries.features import _mic
from tseries.hb_elastic_paths import elastic_modes, mode_paths
from tseries.hb_directional import directional_features
from tseries.hb_learned_coordinate_paths import (local_sites, descriptors,
                                                 FRAMES, SEED)

ORIGIN = 10
ROOT = Path("/home/hatch/workspace/k74/ws")
OUTDIR = Path("/home/hatch/workspace/k74/out"); OUTDIR.mkdir(parents=True, exist_ok=True)
TEACHER = ("1u84_A", "4u5h_G", "5hub_A", "1x6j_A", "2y1y_A", "1krn_A", "2y4x_B", "4xqm_A")

def run_one(reg, protein, replica):
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
    assert x.shape[1] == 110, x.shape
    a = pair[:, 0].astype(int); b = pair[:, 1].astype(int)
    real_hist = np.stack([np.linalg.norm(_mic(pos[f, b] - pos[f, a], cell(box[f])), axis=1)
                          for f in range(11)], axis=0)
    real_mean = real_hist.mean(axis=0)
    np.savez_compressed(OUTDIR / f"{tid}.npz", pair=pair, y=y,
                        descriptors=x.astype(np.float32),
                        coarse=coarse_local,
                        sites=local.astype(np.float32),
                        real_hist=real_hist.astype(np.float32),
                        real_mean=real_mean.astype(np.float64))
    return dict(tid=tid, n=int(len(pair)), y_rate=float(y.mean()),
                real_mean=float(real_mean.mean()))

def main():
    results = []
    for protein in TEACHER:
        outs = [OUTDIR / f"atlas-{protein}-analysis-R{r}.npz" for r in (1, 2, 3)]
        if all(o.is_file() for o in outs):
            print("SKIP", protein, flush=True); continue
        t0 = time.time()
        try:
            acquire(ROOT, protein)
            reg = registry(ROOT, protein)
            for replica in (1, 2, 3):
                out = OUTDIR / f"atlas-{protein}-analysis-R{replica}.npz"
                if out.is_file():
                    print("SKIP", protein, replica, flush=True); continue
                r = run_one(reg, protein, replica)
                r["seconds"] = round(time.time() - t0, 1)
                results.append(r)
                print(json.dumps(r), flush=True)
            print("DONE", protein, "sec", round(time.time() - t0, 1), flush=True)
        except Exception as e:
            print("FAILED", protein, type(e).__name__, str(e)[:300], flush=True)
        finally:
            shutil.rmtree(ROOT / protein, ignore_errors=True)
    (OUTDIR / "k74_build_summary.json").write_text(json.dumps(results, indent=1))
    print("BUILD DONE")

if __name__ == "__main__":
    main()
