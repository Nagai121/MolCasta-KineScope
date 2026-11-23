"""K17 build: K13's contact candidates (DocD definitions) + proposals
11-14 (elastic modes / fluctuations, hb_elastic_paths.py) generated
minimum-distance tracks per residue pair.

Real side (static X, 11-point backward history, labels) is rebuilt with
the same definitions as K13 extract.py and verified array-exact against
the K13 cache. Generation follows K11's adapter protocol exactly:
fresh rng seeded [506166, replica, 10], consumed in the order
mode_paths -> local_paths(omega=0) -> local_paths(omega=.8).
Tracks: for each proposal, the per-atom-pair distance track of every
heavy atom pair is formed (11/13/14 via shared per-atom displacements,
12 via its per-pair mode selection), and the residue-pair track is the
minimum over the pair's atom pairs at each branch and step.
Boxes are handled per axis through tseries.data.cell + _mic."""
import os, sys, json, time
os.environ.setdefault("ATLAS_EXPERIMENT_DIR", "/home/hatch/workspace/k14/ws")
sys.path.insert(0, "/home/hatch/workspace/gpt-model-check")
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
from pathlib import Path
import numpy as np
from atlas_minitopol_models import REG, atom_layout, distance
from tseries.data import topology, cell
from tseries.features import _mic
from tseries.hb_elastic_paths import elastic_modes, mode_paths, local_paths

PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
FRAMES17 = list(range(11)) + [20, 60, 110, 310, 510, 910]
FRAMES5 = (10, 11, 13, 15, 20)
SEED = 506166
OUT = Path("/home/hatch/workspace/k17/out")
CACHE = Path("/home/hatch/workspace/kinescope/work/k13_cache")

def reduced_series(pos, box, a, b, k, n_pairs, frames):
    out = np.empty((len(frames), n_pairs), dtype=np.float64)
    for fi, f in enumerate(frames):
        vals = distance(pos[f, a, :] - pos[f, b, :], box[f])
        cur = np.full(n_pairs, np.inf)
        np.minimum.at(cur, k, vals)
        out[fi] = cur
    return out

def min_tracks_from_positions(pos10, boxv, disp, a, b, k, n_pairs):
    """disp: (atoms, 4, 8, 3) per-atom displacements -> (4, 8, n_pairs) min dist."""
    out = np.empty((4, 8, n_pairs), dtype=np.float64)
    for r in range(4):
        for t in range(8):
            newpos = pos10 + disp[:, r, t]
            vals = np.linalg.norm(_mic(newpos[b] - newpos[a], boxv), axis=1)
            cur = np.full(n_pairs, np.inf)
            np.minimum.at(cur, k, vals)
            out[r, t] = cur
    return out

def focused_tracks(pos10, boxv, modes, values, signal, owner, a, b, k, n_pairs):
    """Proposal 12: per-atom-pair mode selection by response, chunked."""
    out = np.full((4, 8, n_pairs), np.inf)
    base_all = _mic(pos10[b] - pos10[a], boxv)
    for s in range(0, len(a), 8192):
        aa, bb = a[s:s + 8192], b[s:s + 8192]
        diff = modes[owner[bb]] - modes[owner[aa]]              # (P, M, 3)
        base = base_all[s:s + 8192]
        dist = np.linalg.norm(base, axis=1)
        response = np.einsum("ncm,nc->nm", diff, base / np.maximum(dist, 1e-8)[:, None])
        order = np.argsort(np.abs(response) / np.sqrt(values), axis=1)
        selected = np.zeros_like(response, dtype=bool)
        np.put_along_axis(selected, order[:, -min(3, len(values)):], True, axis=1)
        focused = np.einsum("ncm,rtm->nrtc", diff * selected[:, None, :], signal)
        d = np.linalg.norm(base[:, None, None] + focused, axis=-1)  # (P, 4, 8)
        kk = k[s:s + 8192]
        for r in range(4):
            for t in range(8):
                np.minimum.at(out[r, t], kk, d[:, r, t])
    return out

def run_one(protein, replica):
    tid = f"atlas-{protein}-analysis-R{replica}"
    blocks = list(REG.blocks(tid, start=0, stop=911, chunk_frames=128))
    pos = np.concatenate([b.positions for b in blocks], axis=0).astype(np.float64)
    box = np.concatenate([b.box for b in blocks], axis=0).astype(np.float64)
    atoms = REG.atoms(tid).to_pydict()
    maps, _, _ = atom_layout(tid)
    names = np.array([atoms["residue_name"][m["CA"]] for m in maps], dtype=object)
    n = len(maps)
    ca_idx = np.array([m["CA"] for m in maps])
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    t = 10
    ca_d = distance(pos[t, ca_idx[:, None], :] - pos[t, ca_idx[None, :], :], box[t])
    ii, jj = np.where((ca_d <= 12.0) & (sep >= 2) & (np.arange(n)[:, None] < np.arange(n)[None, :]))
    heavy = [np.asarray([v for key, v in m.items()
                         if not key.startswith("H") and not key[0].isdigit()], dtype=int) for m in maps]
    aL, bL, kL = [], [], []
    for z, (i, j) in enumerate(zip(ii, jj)):
        ia, ib = heavy[int(i)], heavy[int(j)]
        aL.append(np.repeat(ia, len(ib))); bL.append(np.tile(ib, len(ia)))
        kL.append(np.full(len(ia) * len(ib), z, dtype=np.int32))
    a, b, k = np.concatenate(aL), np.concatenate(bL), np.concatenate(kL)
    series17 = reduced_series(pos, box, a, b, k, len(ii), FRAMES17)
    cur = series17[10]
    sel = np.flatnonzero(cur > 4.5)
    si, sj = ii[sel], jj[sel]
    hist = series17[:11][:, sel]
    X = np.stack([hist[-1], ca_d[si, sj], np.log1p(sep[si, sj]),
                  np.array([len(heavy[i]) for i in si]),
                  np.array([len(heavy[j]) for j in sj])], axis=1).astype(np.float32)
    y = (series17[11:][:, sel] <= 4.5).T.astype(np.uint8)
    fwd5 = reduced_series(pos, box, a, b, k, len(ii), FRAMES5)[:, sel]
    # verify against K13 cache
    with np.load(CACHE / f"{protein}_R{replica}_contact.npz") as z:
        ok = (np.array_equal(z["res_i"], si) and np.array_equal(z["res_j"], sj)
              and np.allclose(z["X"], X, atol=1e-5) and np.allclose(z["hist"], hist.T, atol=1e-9)
              and np.array_equal(z["y"], y))
    if not ok:
        raise ValueError(f"K13 cache mismatch for {tid}")
    # ---- generation (proposals 11-14), K11 protocol ----
    pos10 = pos[t]; boxv = cell(box[t])
    topo = topology(atoms)
    ca_t = np.array([r["CA"] for r in topo[0]])
    assert np.array_equal(ca_t, ca_idx), "residue order mismatch atlas vs tseries"
    owner = np.full(len(pos10), -1, dtype=np.intp)
    for i, m in enumerate(maps):
        owner[list(m.values())] = i
    owner = np.maximum(owner, 0)
    ca_pos = pos10[ca_idx]
    values, modes, left, right, blocks_e = elastic_modes(ca_pos, boxv)
    rng = np.random.default_rng([SEED, replica, t])
    signal = mode_paths(modes, values, rng)
    local0 = local_paths(ca_pos, pos10, owner, boxv, left, right, blocks_e, rng, omega=0.)
    fast = local_paths(ca_pos, pos10, owner, boxv, left, right, blocks_e, rng, omega=.8)
    disp_res = np.einsum("ncm,rtm->nrtc", modes, signal)          # (res, 4, 8, 3)
    disp11 = disp_res[owner]                                     # (atoms, 4, 8, 3)
    tracks = {}
    tracks[11] = min_tracks_from_positions(pos10, boxv, disp11, a, b, k, len(ii))[:, :, sel]
    tracks[13] = min_tracks_from_positions(pos10, boxv, local0, a, b, k, len(ii))[:, :, sel]
    tracks[14] = min_tracks_from_positions(pos10, boxv, disp11 + .55 * fast, a, b, k, len(ii))[:, :, sel]
    tracks[12] = focused_tracks(pos10, boxv, modes, values, signal, owner, a, b, k, len(ii))[:, :, sel]
    origin = cur[sel]
    out = dict(X=X, hist=hist.T.copy(), y=y, res_i=si, res_j=sj, fwd5=fwd5.T.copy())
    for num in (11, 12, 13, 14):
        full = np.concatenate([np.broadcast_to(origin, (4, 1, len(sel))), tracks[num]], axis=1)
        out[f"gen{num}"] = full.astype(np.float32)               # (4 branches, 9 points, N)
    np.savez_compressed(OUT / f"{protein}_R{replica}.npz", **out)
    return tid, len(sel), int(y.sum())

def main():
    for protein in PROTEINS:
        for replica in (1, 2, 3):
            if (OUT / f"{protein}_R{replica}.npz").is_file():
                print("SKIP", protein, replica, flush=True)
                continue
            t0 = time.time()
            tid, nsel, npos = run_one(protein, replica)
            print("DONE", tid, "candidates", nsel, "positives", npos,
                  "sec", round(time.time() - t0, 1), flush=True)

if __name__ == "__main__":
    main()
