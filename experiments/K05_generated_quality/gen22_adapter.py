"""K05/K06 adapter: regenerate proposal-22 (mechanistic, T1I1962 family) paths
at a single fixed origin (frame 10 = 1 ns) for the Doc D six proteins.

Reuses the reference implementation functions (tseries package, read-only
copies) and the saved shared generator checkpoint (weights_only load).
For each trajectory writes an npz with candidate pairs, the generated
distance series at +0.1/+0.3/+0.5/+1.0 ns, the elastic-only (uncorrected)
series, the native 8 summary features, and labels.
"""
import sys, json, time
sys.path.insert(0, "/tmp/stubs")
sys.path.insert(0, "/tmp/tseries_pkg")
from pathlib import Path
import numpy as np
from minitopol import Workspace
from minitopol.registry import Registry

from tseries.data import frame, topology, cell
from tseries.features import _mic
from tseries.hb_elastic_paths import elastic_modes, mode_paths
from tseries.hb_directional import directional_features
from tseries.hb_learned_coordinate_paths import (local_sites, descriptors,
                                                 network, constrain, summarize,
                                                 FRAMES, SEED)

ORIGIN = 10
DATA = Path.home() / "workspace/gpt-model-check/data"
OUTDIR = Path("/tmp/gen22_out"); OUTDIR.mkdir(exist_ok=True)

def load_generator():
    import torch
    bundle = torch.load("/tmp/gen22/generator.pt", map_location="cpu", weights_only=True)
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
    (OUTDIR / "adapter_summary.json").write_text(json.dumps(results, indent=1))

if __name__ == "__main__":
    main()
