"""K74 evaluation: run a generator bundle (original or retrained) on the
DocD-6 candidates and compare the generated mean with the real mean
(11-point history mean, K08 cache).

Usage: k74_eval.py <bundle_path> <tag>
The original bundle is verified by SHA-256 against the value recorded in
the reference repo's generator.json before loading (weights_only).
Outputs k74_eval_<tag>.json with pooled + per-protein error stats, the
coarse-only (uncorrected) reference, and the generated series' fluctuation.
"""
import sys, json, hashlib
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
from pathlib import Path
import numpy as np
import torch
from minitopol import Workspace
from minitopol.registry import Registry
from tseries.data import frame, topology, cell
from tseries.features import _mic
from tseries.hb_elastic_paths import elastic_modes, mode_paths
from tseries.hb_directional import directional_features
from tseries.hb_learned_coordinate_paths import (local_sites, descriptors,
                                                 network, constrain, SEED)

ORIGIN = 10
DATA = Path.home() / "workspace/gpt-model-check/data"
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
DOCD = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
ORIGINAL_SHA256 = "d35b5d92043fc62479a52c2468a8ae50e56d5442846af0f1d8df2ecfd3e88a05"
ORIGINAL_BUNDLE = Path("/home/hatch/workspace/k06b/gen22/generator.pt")

def load_bundle(path):
    path = Path(path)
    if path.resolve() == ORIGINAL_BUNDLE.resolve():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == ORIGINAL_SHA256, f"original checkpoint SHA mismatch: {actual}"
        print("original checkpoint SHA verified", flush=True)
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    model = network(len(bundle["center"]))
    model.load_state_dict(bundle["state"])
    model.eval()
    return model, bundle["center"].numpy(), bundle["scale"].numpy()

def run_one(reg, protein, replica, model, center, scale):
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
    norm = np.clip((x - center) / scale, -12., 12.).astype(np.float32)
    with torch.inference_mode():
        out = model(torch.from_numpy(norm))
    h = out[:, :360].reshape(-1, 6, 4, 5, 3).numpy()
    tc = torch.from_numpy(local)
    def distances(disp):
        p = constrain(tc, torch.as_tensor(disp[:, None], dtype=torch.float32)).numpy()
        return np.linalg.norm(p[:, :, :, 1], axis=-1)[:, 0]
    gen_dist = distances(coarse_local + h[:, 2])
    coarse_dist = distances(coarse_local)
    with np.load(CACHE / f"{protein}_R{replica}.npz") as c:
        keys = list(zip(c["donor_atom"].tolist(), c["acceptor_atom"].tolist()))
        assert len(set(keys)) == len(keys), "duplicate (donor, acceptor) in cache"
        mine = set(zip(pair[:, 0].tolist(), pair[:, 1].tolist()))
        assert mine == set(keys), "candidate sets differ from K08 cache"
        idx = {k: i for i, k in enumerate(keys)}
        order = np.array([idx[(int(d), int(a))] for d, a in pair[:, :2]], dtype=np.intp)
        real_mean = c["hist"][:, order].mean(axis=0).astype(np.float64)
    gen_mean = (pair[:, 2] + gen_dist.sum(axis=1)) / 5.0
    coarse_mean = (pair[:, 2] + coarse_dist.sum(axis=1)) / 5.0
    return dict(tid=tid, gen_mean=gen_mean, coarse_mean=coarse_mean,
                real_mean=real_mean, gen_fluct=float(gen_dist.std(axis=1).mean()))

def stats(err):
    return dict(n=int(len(err)), mean=float(err.mean()), std=float(err.std()),
                mae=float(np.abs(err).mean()))

def main():
    bundle_path, tag = sys.argv[1], sys.argv[2]
    model, center, scale = load_bundle(bundle_path)
    reg = Registry(Workspace(DATA))
    per_protein, all_err, all_coarse_err, all_gen, all_real, flucts = {}, [], [], [], [], []
    for protein in DOCD:
        errs, cerrs, gms, rms = [], [], [], []
        for replica in (1, 2, 3):
            r = run_one(reg, protein, replica, model, center, scale)
            errs.append(r["gen_mean"] - r["real_mean"])
            cerrs.append(r["coarse_mean"] - r["real_mean"])
            gms.append(r["gen_mean"]); rms.append(r["real_mean"])
            flucts.append(r["gen_fluct"])
            print(r["tid"], "n", len(r["gen_mean"]), "gen-err std %.4f" % (r["gen_mean"] - r["real_mean"]).std(), flush=True)
        err = np.concatenate(errs)
        per_protein[protein] = dict(**stats(err),
                                    corr=float(np.corrcoef(np.concatenate(gms), np.concatenate(rms))[0, 1]))
        all_err.append(err); all_coarse_err.append(np.concatenate(cerrs))
        all_gen.append(np.concatenate(gms)); all_real.append(np.concatenate(rms))
    err = np.concatenate(all_err)
    res = dict(tag=tag, bundle=str(bundle_path),
               pooled=dict(**stats(err), corr=float(np.corrcoef(np.concatenate(all_gen), np.concatenate(all_real))[0, 1])),
               pooled_coarse_only=stats(np.concatenate(all_coarse_err)),
               per_protein=per_protein,
               mean_gen_fluct_per_pair=float(np.mean(flucts)),
               reference_k08=dict(mean=-0.155, std=0.483),
               replica_floor_std=0.249)
    Path(f"/home/hatch/workspace/k74/k74_eval_{tag}.json").write_text(json.dumps(res, indent=1))
    print("EVAL", tag, json.dumps(res["pooled"]), "coarse-only", json.dumps(res["pooled_coarse_only"]), flush=True)

if __name__ == "__main__":
    main()
