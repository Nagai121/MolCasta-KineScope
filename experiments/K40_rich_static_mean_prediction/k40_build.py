"""K40 build: richer static features from the origin frame (frame 10)
for the K23/K08 DocD HB candidates (6 proteins x 3 replicas, 14,582).
Features = DocD 7 (first columns, strict superset) + structure-derived:
residue types (one-hot), atom names (one-hot), CB distance, local chain
geometry (CA i+-2 distances, angles), local density / competing partners,
packing, residue sizes, neighbourhood sizes. All from the origin frame only.
Distances use the per-frame box, axis-wise (cell + _mic)."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg")
import minitopol  # registers the minitopol_agx3 zarr codec
import numpy as np, pandas as pd, zarr
from pathlib import Path
from tseries.data import cell
from tseries.features import _mic

BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path("/home/hatch/workspace/k40/out"); OUT.mkdir(parents=True, exist_ok=True)
PROTS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
AA20 = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]

def load_traj(prot, rep):
    tid = f"atlas-{prot}-analysis-R{rep}"
    z = zarr.open_group(str(BASE / tid / "A.zarr"))
    pos10 = np.asarray(z["positions"][10], dtype=np.float64)
    box10 = np.asarray(z["box"][10], dtype=np.float64)
    at = pd.read_parquet(BASE / tid / "atoms.parquet")
    return pos10, box10, at

# pass 1: observed donor/acceptor atom names
names_d, names_a = set(), set()
for p in PROTS:
    for r in (1, 2, 3):
        c = np.load(CACHE / f"{p}_R{r}.npz")
        _, _, at = load_traj(p, r) if False else (None, None, None)
        # parquet only (no zarr read needed here)
        at = pd.read_parquet(BASE / f"atlas-{p}-analysis-R{r}" / "atoms.parquet")
        nm = at["name"].to_numpy()
        for i in c["donor_atom"]: names_d.add(str(nm[int(i)]))
        for i in c["acceptor_atom"]: names_a.add(str(nm[int(i)]))
NAMES_D = sorted(names_d); NAMES_A = sorted(names_a)
print("donor atom names:", NAMES_D)
print("acceptor atom names:", NAMES_A)

def build_one(prot, rep):
    c = np.load(CACHE / f"{p}_R{rep}".replace("{p}", prot)) if False else np.load(CACHE / f"{prot}_R{rep}.npz")
    X7 = c["X"].astype(np.float64)
    da, aa = c["donor_atom"].astype(int), c["acceptor_atom"].astype(int)
    dr, ar = c["donor_res"].astype(int), c["acceptor_res"].astype(int)
    real_mean = c["hist"].mean(axis=0).astype(np.float64)
    pos, box, at = load_traj(prot, rep)
    nm = at["name"].to_numpy().astype(str)
    rn = at["residue_name"].to_numpy().astype(str)
    ri = at["residue_index"].to_numpy().astype(int)
    # alignment checks: atom indices are parquet row indices
    assert (ri[da] == dr).all() and (ri[aa] == ar).all(), "residue index mismatch"
    assert all(n.startswith("N") or n.startswith("O") for n in nm[da]), "donor not N/O"
    assert all(n.startswith("O") for n in nm[aa]), "acceptor not O"
    n_res = int(ri.max()) + 1
    # residue maps: name -> atom index
    maps = [{} for _ in range(n_res)]
    for i, (r, n) in enumerate(zip(ri, nm)):
        maps[r][n] = i
    ca = np.array([m.get("CA", -1) for m in maps], dtype=int)
    cb = np.array([m.get("CB", -1) for m in maps], dtype=int)
    heavy = np.array([not n.startswith("H") and not n[0].isdigit() for n in nm])
    is_o = np.array([n.startswith("O") for n in nm])
    is_n = np.array([n.startswith("N") for n in nm])
    cbox = cell(box)
    def dist(a_idx, b_idx):
        return np.linalg.norm(_mic(pos[b_idx] - pos[a_idx], cbox), axis=1)
    def dist_points(P, Q):
        # P (M,3), Q (K,3) -> (M,K) MIC distances
        d = P[:, None, :] - Q[None, :, :]
        return np.linalg.norm(_mic(d.reshape(-1, 3), cbox).reshape(len(P), len(Q), 3), axis=2)
    N = len(da)
    feats, fnames = [], []
    def add(v, name):
        feats.append(np.asarray(v, dtype=np.float64)); fnames.append(name)
    # --- DocD 7 first (strict superset) ---
    for j, nm7 in enumerate(["origin_dist", "ca_dist", "log1p_sep", "donor_kind0",
                             "acceptor_kind0", "donor_kind2", "acceptor_kind1"]):
        add(X7[:, j], "docd_" + nm7)
    sep = np.abs(dr - ar).astype(np.float64)
    add(sep, "sep")
    # --- CB distance ---
    cb_d = np.full(N, np.nan)
    ok = (cb[dr] >= 0) & (cb[ar] >= 0)
    cb_d[ok] = dist(cb[dr[ok]], cb[ar[ok]])
    add(cb_d, "cb_dist")
    # --- local chain geometry: CA i+-2 distances ---
    def ca_shift(res, sh):
        out = np.full(N, np.nan); idx = res + sh
        idxc = np.clip(idx, 0, n_res - 1)
        ok = (idx >= 0) & (idx < n_res) & (ca[res] >= 0) & (ca[idxc] >= 0)
        out[ok] = dist(ca[res[ok]], ca[idxc[ok]])
        return out
    add(ca_shift(dr, 2), "donor_ca_i2"); add(ca_shift(dr, -2), "donor_ca_im2")
    add(ca_shift(ar, 2), "acceptor_ca_i2"); add(ca_shift(ar, -2), "acceptor_ca_im2")
    # --- angles (cos) ---
    def cos_angle(P, Q, R):  # angle PQR at Q, P/Q/R are (N,3) point arrays
        u = P - Q; v = R - Q
        return (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-12)
    v_da = pos[aa] - pos[da]
    v_ca = pos[ca[ar]] - pos[ca[dr]]
    add((v_da * v_ca).sum(1) / (np.linalg.norm(v_da, axis=1) * np.linalg.norm(v_ca, axis=1) + 1e-12),
        "cos_da_vs_caca")
    def chain_dir(res):
        # CA(i+1) - CA(i-1), NaN-safe fallback to zeros
        p = np.where((res + 1 < n_res) & (ca[np.clip(res + 1, 0, n_res - 1)] >= 0), ca[np.clip(res + 1, 0, n_res - 1)], -1)
        m = np.where((res - 1 >= 0) & (ca[np.clip(res - 1, 0, n_res - 1)] >= 0), ca[np.clip(res - 1, 0, n_res - 1)], -1)
        ok = (p >= 0) & (m >= 0)
        d = np.zeros((N, 3)); d[ok] = pos[p[ok]] - pos[m[ok]]
        return d, ok
    dd, okd = chain_dir(dr); ad, oka = chain_dir(ar)
    cosd = np.full(N, np.nan); cosa = np.full(N, np.nan)
    ok = okd & (np.linalg.norm(dd, axis=1) > 0)
    cosd[ok] = (v_da[ok] * dd[ok]).sum(1) / (np.linalg.norm(v_da[ok], axis=1) * np.linalg.norm(dd[ok], axis=1) + 1e-12)
    ok = oka & (np.linalg.norm(ad, axis=1) > 0)
    cosa[ok] = (v_da[ok] * ad[ok]).sum(1) / (np.linalg.norm(v_da[ok], axis=1) * np.linalg.norm(ad[ok], axis=1) + 1e-12)
    add(cosd, "cos_da_vs_donor_chain"); add(cosa, "cos_da_vs_acceptor_chain")
    # --- atom to own CA ---
    add(dist(da, ca[dr]), "donor_atom_to_ca"); add(dist(aa, ca[ar]), "acceptor_atom_to_ca")
    add(dist(ca[dr], aa), "donor_ca_to_acceptor"); add(dist(ca[ar], da), "acceptor_ca_to_donor")
    # --- residue sizes ---
    add(np.array([sum(1 for n in maps[r] if not n.startswith("H") and not n[0].isdigit()) for r in dr], dtype=float),
        "donor_heavy_count")
    add(np.array([sum(1 for n in maps[r] if not n.startswith("H") and not n[0].isdigit()) for r in ar], dtype=float),
        "acceptor_heavy_count")
    # --- local density / competing / packing (chunked MIC) ---
    hpos = pos[heavy]
    h_is_o = is_o[heavy]; h_is_n = is_n[heavy]
    def count_within(qpos, pts, cut):
        D = dist_points(qpos, pts)
        return (D <= cut).sum(1).astype(float), D
    cnt5_d, Dd = count_within(pos[da], hpos, 5.0)
    cnt8_d, _ = count_within(pos[da], hpos, 8.0)
    cnt5_a, Da = count_within(pos[aa], hpos, 5.0)
    cnt8_a, _ = count_within(pos[aa], hpos, 8.0)
    add(cnt5_d, "donor_heavy_5A"); add(cnt8_d, "donor_heavy_8A")
    add(cnt5_a, "acceptor_heavy_5A"); add(cnt8_a, "acceptor_heavy_8A")
    mid = (pos[da] + pos[aa]) / 2
    cntm, _ = count_within(mid, hpos, 5.0)
    add(cntm, "mid_heavy_5A")
    # competing: O within 4A of donor (excluding the acceptor itself), N within 4A of acceptor
    # (exclude the partner by masking its own entry; it need not be the nearest one)
    o_idx = np.flatnonzero(is_o); n_idx = np.flatnonzero(is_n)
    Do = dist_points(pos[da], pos[o_idx]); Dn = dist_points(pos[aa], pos[n_idx])
    Do[np.arange(N), np.searchsorted(o_idx, aa)] = np.inf
    Dn[np.arange(N), np.searchsorted(n_idx, da)] = np.inf
    add((Do <= 4.0).sum(1), "donor_competing_O_4A")
    add((Dn <= 4.0).sum(1), "acceptor_competing_N_4A")
    add(Do.min(1), "donor_nearest_other_O")
    add(Dn.min(1), "acceptor_nearest_other_N")
    # packing: min distance to any other heavy atom
    add(np.sort(Dd, axis=1)[:, 1], "donor_min_other_heavy")
    add(np.sort(Da, axis=1)[:, 1], "acceptor_min_other_heavy")
    # --- neighbourhood sizes (residue level, CA-based) ---
    ca_pos = pos[ca[ca >= 0]]; ca_res = np.flatnonzero(ca >= 0)
    Dca = dist_points(pos[ca[dr]], ca_pos)
    add((Dca <= 8.0).sum(1) - 1.0, "donor_res_neighbours_8A")
    Dca2 = dist_points(pos[ca[ar]], ca_pos)
    add((Dca2 <= 8.0).sum(1) - 1.0, "acceptor_res_neighbours_8A")
    # --- one-hots ---
    for j, a in enumerate(AA20):
        add((rn[dr] == a).astype(float), f"donor_res_{a}")
        add((rn[ar] == a).astype(float), f"acceptor_res_{a}")
    add(~np.isin(rn[dr], AA20), "donor_res_other")
    add(~np.isin(rn[ar], AA20), "acceptor_res_other")
    for n in NAMES_D:
        add((nm[da] == n).astype(float), f"donor_atom_{n}")
    for n in NAMES_A:
        add((nm[aa] == n).astype(float), f"acceptor_atom_{n}")
    add((nm[da] == "N").astype(float), "donor_backbone_N")
    add((nm[aa] == "O").astype(float), "acceptor_backbone_O")
    Xr = np.column_stack(feats)
    assert Xr.shape == (N, len(fnames)), (Xr.shape, len(fnames))
    np.savez(OUT / f"{prot}_R{rep}.npz", Xrich=Xr, feature_names=np.array(fnames),
             X7=X7, real_mean=real_mean, donor_atom=da, acceptor_atom=aa,
             donor_res=dr, acceptor_res=ar, protein=prot, replica=rep)
    print(f"{prot} R{rep}: N={N} F={Xr.shape[1]} nan={int(np.isnan(Xr).sum())}", flush=True)

if __name__ == "__main__":
    for p in PROTS:
        for r in (1, 2, 3):
            out = OUT / f"{p}_R{r}.npz"
            if out.is_file():
                print("SKIP", p, r); continue
            build_one(p, r)
    print("DONE")
