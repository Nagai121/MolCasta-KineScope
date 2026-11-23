"""K14 mechanical probe: can proposal-22's generator be fed salt-bridge /
contact candidates (DocD definitions, fixed_1ns_history_3phen_6h.py)?
The generator's descriptors are defined only for (donor, acceptor) index
pairs of the HB frame, so a residue-pair candidate can be fed only via a
representative atom pair that is a donor-acceptor pair. For each candidate
the representative pair attains the minimum distance at the origin
(frame 10). Real series are kept for all candidates (minimum distance =
the label quantity, and the representative pair's own distance).
Box handled per-axis via the DocD/tseries MIC helpers throughout."""
import os, sys, json
os.environ.setdefault("ATLAS_EXPERIMENT_DIR", "/home/hatch/workspace/k14/ws")
sys.path.insert(0, "/home/hatch/workspace/gpt-model-check")
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
from pathlib import Path
import numpy as np
import torch
from atlas_minitopol_models import REG, atom_layout, distance
from tseries.data import topology, cell
from tseries.features import _mic, structural, feature_graph, enriched_features
from tseries.hb_directional import directional_features
from tseries.hb_elastic_paths import elastic_modes, mode_paths
from tseries.hb_learned_coordinate_paths import local_sites, descriptors, network, constrain, SEED

PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
FRAMES5 = (10, 11, 13, 15, 20)
OUT = Path("/home/hatch/workspace/k14/out")

def pair_dist(pos, box, a, b):
    return distance(pos[a] - pos[b], box)

def reduced(pos, box, a, b, k, n):
    vals = pair_dist(pos, box, a, b)
    out = np.full(n, np.inf)
    np.minimum.at(out, k, vals)
    return out

def build_candidates(phen, pos10, box10, maps, names):
    n = len(maps)
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    ca = np.array([m["CA"] for m in maps], dtype=int)
    if phen == "salt":
        positive, negative = [], []
        for k, (m, r) in enumerate(zip(maps, names)):
            if r == "LYS" and "NZ" in m:
                positive.append((k, np.array([m["NZ"]])))
            if r == "ARG":
                aa = np.array([m[a] for a in ("NE", "NH1", "NH2") if a in m], dtype=int)
                if len(aa): positive.append((k, aa))
            if r in ("ASP", "GLU"):
                aa = np.array([m[a] for a in (("OD1", "OD2") if r == "ASP" else ("OE1", "OE2")) if a in m], dtype=int)
                if len(aa): negative.append((k, aa))
        if not positive or not negative:
            return None
        pi, pj = np.indices((len(positive), len(negative)))
        ii, jj = pi.ravel(), pj.ravel()
        pp = [a for _, a in positive]; nn = [a for _, a in negative]
        aL, bL, kL = [], [], []
        for z, (i, j) in enumerate(zip(ii, jj)):
            aL.append(np.repeat(pp[i], len(nn[j]))); bL.append(np.tile(nn[j], len(pp[i])))
            kL.append(np.full(len(pp[i]) * len(nn[j]), z, dtype=np.int32))
        a, b, k = np.concatenate(aL), np.concatenate(bL), np.concatenate(kL)
        cur = reduced(pos10, box10, a, b, k, len(ii))
        sel = np.flatnonzero((cur > 4.0) & (cur <= 10.0) &
                             (sep[[positive[i][0] for i in ii], [negative[j][0] for j in jj]] >= 2))
        return a, b, k, sel
    ca_d = distance(pos10[ca[:, None], :] - pos10[ca[None, :], :], box10)
    ii, jj = np.where((ca_d <= 12.0) & (sep >= 2) & (np.arange(n)[:, None] < np.arange(n)[None, :]))
    heavy = [np.asarray([v for key, v in m.items()
                         if not key.startswith("H") and not key[0].isdigit()], dtype=int) for m in maps]
    aa, bb, kk = [], [], []
    for z, (i, j) in enumerate(zip(ii, jj)):
        ia, ib = heavy[int(i)], heavy[int(j)]
        aa.append(np.repeat(ia, len(ib))); bb.append(np.tile(ib, len(ia)))
        kk.append(np.full(len(ia) * len(ib), z, dtype=np.int32))
    a, b, k = np.concatenate(aa), np.concatenate(bb), np.concatenate(kk)
    cur = reduced(pos10, box10, a, b, k, len(ii))
    sel = np.flatnonzero(cur > 4.5)
    return a, b, k, sel

def generate(pos10, boxv, atoms, topo, ii, jj, replica, model, bundle):
    maps_t, rnames, context, types = topo
    d_at, a_at = context[5], context[9]
    a, b = d_at[ii], a_at[jj]
    delta = _mic(pos10[b] - pos10[a], boxv)
    dist0 = np.linalg.norm(delta, axis=1)
    pair = np.column_stack((a, b, dist0, delta, types[a], types[b])).astype(np.float32)
    base0 = structural(pos10, boxv, maps_t, rnames, context, ii, jj)
    graph = feature_graph(base0, pair[:, 0], pair[:, 1], np.zeros(len(pair)), len(pos10))
    feats = enriched_features(np.column_stack((base0, graph)))
    part = dict(pair=pair, base=feats)
    part["directional"] = directional_features(pos10, boxv, pair, topo[0])
    sites, basis, local = local_sites(pos10, boxv, pair, atoms)
    residues = topo[0]
    ca = np.array([r["CA"] for r in residues])
    owner = np.full(len(atoms["name"]), -1, dtype=np.intp)
    for i, residue in enumerate(residues):
        owner[list(residue.values())] = i
    values, modes, _, _, _ = elastic_modes(pos10[ca], boxv)
    rng = np.random.default_rng([SEED, 10, replica])
    signal = mode_paths(modes, values, rng)[:, (0, 2, 4, 7)]
    coarse = np.einsum("ncm,rtm->rtnc", modes, signal)
    coarse = coarse[:, :, np.maximum(owner[sites], 0)]
    coarse -= coarse[:, :, :, :1].copy()
    coarse_local = np.einsum("rtnsc,nck->nrtsk", coarse, basis).mean(axis=1).astype(np.float32)
    part["sites"] = local; part["coarse"] = coarse_local
    x = descriptors(part)
    finite = np.isfinite(x).all(axis=1)
    raw = (x - bundle["center"].numpy()) / bundle["scale"].numpy()
    clipfrac = float((np.abs(raw) > 12).mean())
    norm = np.clip(raw, -12., 12.).astype(np.float32)
    with torch.inference_mode():
        o = model(torch.from_numpy(norm))
    h = o[:, :360].reshape(-1, 6, 4, 5, 3).numpy()
    corrected = coarse_local + h[:, 2]
    lt = torch.from_numpy(local)
    gen_path = constrain(lt, torch.as_tensor(corrected[:, None], dtype=torch.float32)).numpy()
    gen_dist = np.linalg.norm(gen_path[:, :, :, 1], axis=-1)[:, 0]
    coarse_path = constrain(lt, torch.as_tensor(coarse_local[:, None], dtype=torch.float32)).numpy()
    coarse_dist = np.linalg.norm(coarse_path[:, :, :, 1], axis=-1)[:, 0]
    return dist0, gen_dist, coarse_dist, finite, clipfrac

def main():
    bundle = torch.load("/home/hatch/workspace/k06b/gen22/generator.pt",
                        map_location="cpu", weights_only=True)
    model = network(len(bundle["center"]))
    model.load_state_dict(bundle["state"]); model.eval()
    store = {p: dict(gen=[], coarse=[], rep=[], min=[], min_all=[]) for p in ("salt", "contact")}
    meta = {p: dict(per_traj={}, clipfracs=[], nonfinite=0, candidates=0, feedable=0)
            for p in ("salt", "contact")}
    for protein in PROTEINS:
        for replica in (1, 2, 3):
            tid = f"atlas-{protein}-analysis-R{replica}"
            blocks = list(REG.blocks(tid, start=0, stop=21, chunk_frames=21))
            pos = np.concatenate([b.positions for b in blocks], axis=0)
            box = np.concatenate([b.box for b in blocks], axis=0)
            atoms = REG.atoms(tid).to_pydict()
            maps, _, _ = atom_layout(tid)
            names = [atoms["residue_name"][m["CA"]] for m in maps]
            topo = topology(atoms)
            context = topo[2]
            d_index = {int(at): i for i, at in enumerate(context[5])}
            a_index = {int(at): j for j, at in enumerate(context[9])}
            pos10, box10 = np.asarray(pos[10]), box[10]
            for phen in ("salt", "contact"):
                got = build_candidates(phen, pos10, box10, maps, names)
                if got is None:
                    meta[phen]["per_traj"][tid] = (0, 0)
                    continue
                a, b, k, sel = got
                n_pairs = int(k.max()) + 1
                min_series = np.stack([reduced(pos[f], box[f], a, b, k, n_pairs)[sel] for f in FRAMES5])
                store[phen]["min_all"].append(min_series.T)
                d0_all = pair_dist(pos10, box10, a, b)
                rep_a, rep_b = [], []
                for z in sel:
                    idx = np.flatnonzero(k == z)
                    j = idx[np.argmin(d0_all[idx])]
                    rep_a.append(int(a[j])); rep_b.append(int(b[j]))
                # feedable = representative pair is a (donor, acceptor) pair in either orientation
                ii, jj, keep = [], [], []
                for z, (ra_, rb_) in enumerate(zip(rep_a, rep_b)):
                    if ra_ in d_index and rb_ in a_index:
                        ii.append(d_index[ra_]); jj.append(a_index[rb_]); keep.append(z)
                    elif rb_ in d_index and ra_ in a_index:
                        ii.append(d_index[rb_]); jj.append(a_index[ra_]); keep.append(z)
                keep = np.array(keep, dtype=int)
                meta[phen]["candidates"] += int(len(sel))
                meta[phen]["feedable"] += int(len(keep))
                meta[phen]["per_traj"][tid] = (int(len(sel)), int(len(keep)))
                if len(keep) == 0:
                    continue
                rep_series = np.stack([pair_dist(pos[f], box[f],
                                                 np.array(rep_a)[keep], np.array(rep_b)[keep])
                                       for f in FRAMES5])
                dist0, gen, coarse, finite, clipfrac = generate(
                    pos10, cell(box10), atoms, topo, np.array(ii), np.array(jj),
                    replica, model, bundle)
                assert np.allclose(dist0, rep_series[0], atol=1e-4), (phen, tid)
                meta[phen]["nonfinite"] += int((~finite).sum())
                meta[phen]["clipfracs"].append(clipfrac)
                fin = finite
                store[phen]["gen"].append(np.column_stack([dist0, gen])[fin])
                store[phen]["coarse"].append(np.column_stack([dist0, coarse])[fin])
                store[phen]["rep"].append(rep_series.T[fin])
                store[phen]["min"].append(min_series.T[keep][fin])
            print("DONE", tid, {p: meta[p]["per_traj"][tid] for p in meta}, flush=True)
    for phen in ("salt", "contact"):
        np.savez_compressed(OUT / f"k14_{phen}.npz",
                            **{kk: np.concatenate(v).astype(np.float32)
                               for kk, v in store[phen].items() if v})
        meta[phen]["clipfrac_mean"] = float(np.mean(meta[phen].pop("clipfracs"))) \
            if meta[phen]["clipfracs"] else None
    (OUT / "k14_meta.json").write_text(json.dumps(meta, indent=1))
    print("WROTE", {p: (meta[p]["candidates"], meta[p]["feedable"]) for p in meta})

if __name__ == "__main__":
    main()
