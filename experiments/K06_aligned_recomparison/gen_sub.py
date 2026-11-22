"""#22 generated features for one replica of one protein (tseries-branch pkg).
Usage: gen_sub.py <workspace_root> <protein> <replica> <out_npz>
Writes pair atom ids, gen_dist (N,4), gen8 (N,8)."""
import sys
sys.path.insert(0, "/tmp/stubs")
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
import numpy as np
from pathlib import Path
import torch
from tseries.data import registry, trajectory, topology, cell, frame
from tseries.features import _mic
from tseries.hb_elastic_paths import elastic_modes, mode_paths
from tseries.hb_directional import directional_features
from tseries.hb_learned_coordinate_paths import (local_sites, descriptors,
                                                 network, constrain, summarize, SEED)
ORIGIN = 10

def main():
    root, protein, replica, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
    reg = registry(root, protein)
    tid = trajectory(protein, replica)
    atoms = reg.atoms(tid).to_pydict()
    topo = topology(atoms)
    blocks = list(reg.blocks(tid, start=0, stop=21, chunk_frames=21))
    pos = np.concatenate([b.positions for b in blocks], axis=0)
    box = np.concatenate([b.box for b in blocks], axis=0)
    position = np.asarray(pos[ORIGIN]); bx = cell(box[ORIGIN])
    pair, base, y, counts = frame(position, bx, topo, 8.0,
                                  future=np.asarray(pos[20]), future_box=cell(box[20]))
    keep = pair[:, 2] <= 6.0
    pair, base = pair[keep], base[keep]
    part = dict(pair=pair, base=base)
    part["directional"] = directional_features(position, bx, pair, topo[0])
    sites, basis, local = local_sites(position, bx, pair, atoms)
    residues = topo[0]
    ca = np.array([r["CA"] for r in residues])
    owner = np.full(len(atoms["name"]), -1, dtype=np.intp)
    for i, residue in enumerate(residues):
        owner[list(residue.values())] = i
    values, modes, _, _, _ = elastic_modes(position[ca], bx)
    rng = np.random.default_rng([SEED, ORIGIN, replica])
    signal = mode_paths(modes, values, rng)[:, (0, 2, 4, 7)]
    coarse = np.einsum("ncm,rtm->rtnc", modes, signal)
    coarse = coarse[:, :, np.maximum(owner[sites], 0)]
    coarse -= coarse[:, :, :, :1].copy()
    coarse_local = np.einsum("rtnsc,nck->nrtsk", coarse, basis).mean(axis=1).astype(np.float32)
    part["sites"] = local; part["coarse"] = coarse_local
    x = descriptors(part)
    bundle = torch.load("/home/hatch/workspace/k06b/gen22/generator.pt", map_location="cpu", weights_only=True)
    model = network(len(bundle["center"]))
    model.load_state_dict(bundle["state"]); model.eval()
    norm = np.clip((x - bundle["center"].numpy()) / bundle["scale"].numpy(), -12., 12.)
    with torch.inference_mode():
        o = model(torch.from_numpy(norm))
    h = o[:, :360].reshape(-1, 6, 4, 5, 3).numpy()
    corrected = coarse_local + h[:, 2]
    path = constrain(torch.from_numpy(local), torch.as_tensor(corrected[:, None], dtype=torch.float32)).numpy()
    gen_dist = np.linalg.norm(path[:, :, :, 1], axis=-1)[:, 0]
    gen8 = summarize(path, pair[:, 2])
    np.savez_compressed(out, pair=pair, gen_dist=gen_dist.astype(np.float32), gen8=gen8)
    print("GEN OK", protein, replica, len(pair), flush=True)

if __name__ == "__main__":
    main()
