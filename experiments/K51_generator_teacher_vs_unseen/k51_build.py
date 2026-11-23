"""K51 build: proposal-22 generated series + real 11-point history means
for the generator's 8 teacher proteins, DocD candidate definitions.

Same generation method as K05 (gen22_adapter.py): saved generator.pt
(weights_only), elastic coarse path + learned correction head, origin
frame 10, rng [SEED, 10, replica]. Paths use the persistent copies under
~/workspace/k06b (the /tmp copies from K05 no longer exist).
Teacher proteins are acquired via tseries acquire() and the workspace
is deleted after each protein (as in K06(b) score_b.py)."""
import sys, json, time, shutil
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
from pathlib import Path
import numpy as np
from minitopol import Workspace
from minitopol.registry import Registry
from tseries.data import frame, topology, cell, registry, acquire, trajectory
from tseries.features import _mic
from tseries.hb_elastic_paths import elastic_modes, mode_paths
from tseries.hb_directional import directional_features
from tseries.hb_learned_coordinate_paths import (local_sites, descriptors,
                                                 network, constrain, summarize,
                                                 FRAMES, SEED)

ORIGIN = 10
ROOT = Path("/home/hatch/workspace/k51/ws")
OUTDIR = Path("/home/hatch/workspace/k51/out"); OUTDIR.mkdir(parents=True, exist_ok=True)
TEACHER = ("1u84_A", "4u5h_G", "5hub_A", "1x6j_A", "2y1y_A", "1krn_A", "2y4x_B", "4xqm_A")

def load_generator():
    import torch
    bundle = torch.load("/home/hatch/workspace/k06b/gen22/generator.pt",
                        map_location="cpu", weights_only=True)
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
    # real 11-point backward history (frames 0..10), same definition as K08/K23
    a = pair[:, 0].astype(int); b = pair[:, 1].astype(int)
    real_hist = np.stack([np.linalg.norm(_mic(pos[f, b] - pos[f, a], cell(box[f])), axis=1)
                          for f in range(11)], axis=0)
    np.savez_compressed(OUTDIR / f"{tid}.npz", pair=pair, y=y,
                        gen_dist=gen_dist.astype(np.float32),
                        coarse_dist=coarse_dist.astype(np.float32),
                        gen8=gen8, real_hist=real_hist.astype(np.float32))
    return dict(tid=tid, n=int(len(pair)), y_rate=float(y.mean()),
                gen_mean=float(gen_dist.mean()), real_mean=float(real_hist.mean(axis=0).mean()))

def main():
    model, center, scale = load_generator()
    print("generator input width:", len(center), flush=True)
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
                r = run_one(reg, protein, replica, model, center, scale)
                r["seconds"] = round(time.time() - t0, 1)
                results.append(r)
                print(json.dumps(r), flush=True)
            print("DONE", protein, "sec", round(time.time() - t0, 1), flush=True)
        except Exception as e:
            print("FAILED", protein, type(e).__name__, str(e)[:300], flush=True)
        finally:
            shutil.rmtree(ROOT / protein, ignore_errors=True)
    (OUTDIR / "k51_build_summary.json").write_text(json.dumps(results, indent=1))
    print("BUILD DONE")

if __name__ == "__main__":
    main()
