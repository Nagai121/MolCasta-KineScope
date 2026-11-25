"""K3-02 shared counting code: H-bond network counts per candidate at a
chosen history frame (frame 0 = initial structure, frame 10 = origin
frame). Topology/donor-acceptor definitions replicate the base pipeline
(tseries prepare). Pair atom indices (k06b_out, converted heavy-atom
order) map onto the total universe's protein atoms via heavy_rank.
"""
import numpy as np

HB = 3.5
DONOR_N = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"},
           "GLN": {"NE2"}, "HIS": {"ND1", "NE2"}, "TRP": {"NE1"}}
DONOR_O = {"SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}}
ACCEPTOR_O = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}, "ASN": {"OD1"},
              "GLN": {"OE1"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}}
COLUMNS = ["n_out_donor_res", "n_in_donor_res", "n_out_acceptor_res", "n_in_acceptor_res",
           "n_hb_total", "n_partner_backbone", "n_partner_sidechain", "n_partner_water",
           "n_competing", "dist_ab_frame0", "bonded_frame0", "n_hb_donor_res_total",
           "n_hb_acceptor_res_total"]
COLUMNS_F10 = ["n_out_donor_res", "n_in_donor_res", "n_out_acceptor_res", "n_in_acceptor_res",
               "n_hb_total", "n_partner_backbone", "n_partner_sidechain", "n_partner_water",
               "n_competing", "dist_ab_frame10", "bonded_frame10", "n_hb_donor_res_total",
               "n_hb_acceptor_res_total"]

def build_protein_topology(u):
    prot = u.select_atoms("protein")
    names = prot.names; resids = prot.resindices; resnames = prot.resnames
    n_atoms = len(names)
    res_atoms = {}
    for i, r in enumerate(resids):
        res_atoms.setdefault(int(r), []).append(i)
    keep = [r for r in sorted(res_atoms) if {"N", "CA", "C", "O"} <= {names[i] for i in res_atoms[r]}]
    res_ord = np.full(n_atoms, -1, dtype=np.int32)
    donors, acceptors = [], []
    for k, r in enumerate(keep):
        idxs = res_atoms[r]
        for i in idxs:
            res_ord[i] = k
        mp = {names[i]: i for i in idxs}
        rn = str(resnames[idxs[0]])
        if rn != "PRO":
            donors.append((mp["N"], k, 0))
        for an in sorted(DONOR_N.get(rn, ())):
            if an in mp: donors.append((mp[an], k, 1))
        for an in sorted(DONOR_O.get(rn, ())):
            if an in mp: donors.append((mp[an], k, 2))
        acceptors.append((mp["O"], k, 0))
        if "OXT" in mp: acceptors.append((mp["OXT"], k, 0))
        for an in sorted(ACCEPTOR_O.get(rn, ())):
            if an in mp: acceptors.append((mp[an], k, 1))
    d = np.array(donors, dtype=np.int64); a = np.array(acceptors, dtype=np.int64)
    heavy_rank = np.where(np.array([str(e).upper() != "H" for e in prot.elements]))[0]
    return dict(n_protein_atoms=n_atoms, res_ord=res_ord, heavy_rank=heavy_rank,
                d_at=d[:, 0], d_res=d[:, 1], d_kind=d[:, 2],
                a_at=a[:, 0], a_res=a[:, 1], a_kind=a[:, 2], elements=prot.elements)

def mic_dist(P, Q, box):
    out = np.empty((len(P), len(Q)), dtype=np.float64)
    for s in range(0, len(P), 256):
        d = Q[None, :, :] - P[s:s + 256, None, :]
        d -= box * np.round(d / box)
        out[s:s + 256] = np.linalg.norm(d, axis=2)
    return out

class FramesUnavailable(Exception):
    pass

def count_replica(u, topo, pair, hist0, count_frame=0):
    """Count the H-bond network quantities at history frame `count_frame`
    (0 = initial structure, 100 = total frame 100 = origin frame 10).
    The premise check always uses both frames 0 and 100 against the
    saved history distances, whichever frame is counted."""
    frames = {}
    try:
        for fi, ts in enumerate(u.trajectory):
            if fi in (0, 100):
                frames[fi] = (ts.positions.copy().astype(np.float64),
                              ts.dimensions[:3].astype(np.float64))
            if fi >= 100:
                break
    except Exception as e:
        raise FramesUnavailable(str(e)[:120])
    if 0 not in frames or 100 not in frames:
        raise FramesUnavailable(f"frames present: {sorted(frames)}")
    pos0, box0 = frames[0]; pos100, box100 = frames[100]
    a_idx = topo["heavy_rank"][pair[:, 0].astype(np.intp)]
    b_idx = topo["heavy_rank"][pair[:, 1].astype(np.intp)]
    def pdist(pos, box, x, y):
        d = pos[y] - pos[x]; d -= box * np.round(d / box)
        return np.linalg.norm(d, axis=1)
    prem = dict(
        frame0_maxdiff=float(np.max(np.abs(pdist(pos0, box0, a_idx, b_idx) - hist0))),
        frame100_maxdiff=float(np.max(np.abs(pdist(pos100, box100, a_idx, b_idx) - pair[:, 2]))),
        elem_a_ok=bool(np.all(topo["elements"][a_idx] == np.where(pair[:, 6] == 7, "N", "O"))),
        elem_b_ok=bool(np.all(topo["elements"][b_idx] == np.where(pair[:, 7] == 7, "N", "O"))))
    if count_frame not in frames:
        raise ValueError(f"count_frame {count_frame} not in {sorted(frames)}")
    pos, box = frames[count_frame]
    P = pos[:topo["n_protein_atoms"]]
    wat = u.select_atoms("resname SOL and name OW")
    W = pos[wat.indices] if len(wat) else np.empty((0, 3))
    dP = P[topo["d_at"]]; aP = P[topo["a_at"]]
    Dpp = mic_dist(dP, aP, box)
    bonds_pp = np.argwhere((Dpp <= HB) & (topo["d_res"][:, None] != topo["a_res"][None, :]))
    bl = []
    for i, j in bonds_pp:
        bl.append((int(topo["d_res"][i]), int(topo["a_res"][j]),
                   0 if topo["d_kind"][i] == 0 else 1, 0 if topo["a_kind"][j] == 0 else 1,
                   int(topo["d_at"][i]), int(topo["a_at"][j])))
    if len(W):
        Dpw = mic_dist(dP, W, box)
        for i, j in np.argwhere(Dpw <= HB):
            bl.append((int(topo["d_res"][i]), -1, 0 if topo["d_kind"][i] == 0 else 1, 2,
                       int(topo["d_at"][i]), -1))
        Dwp = mic_dist(W, aP, box)
        for i, j in np.argwhere(Dwp <= HB):
            bl.append((-1, int(topo["a_res"][j]), 2, 0 if topo["a_kind"][j] == 0 else 1,
                       -1, int(topo["a_at"][j])))
    bl = np.array(bl, dtype=np.int64) if bl else np.empty((0, 6), dtype=np.int64)
    bd_res, ba_res, bd_kc, ba_kc, bd_at, ba_at = (bl[:, k] for k in range(6))
    Rd = topo["res_ord"][a_idx]; Ra = topo["res_ord"][b_idx]
    n = len(pair)
    out = np.zeros((n, 13), dtype=np.float64)
    for ci, (rd, ra, a_atm, b_atm) in enumerate(zip(Rd, Ra, a_idx, b_idx)):
        is_d = bd_res == rd; is_a = ba_res == rd
        is_d2 = bd_res == ra; is_a2 = ba_res == ra
        n_out_d = int((is_d & (ba_res != ra)).sum()); n_in_d = int((is_a & (bd_res != ra)).sum())
        n_out_a = int((is_d2 & (ba_res != rd)).sum()); n_in_a = int((is_a2 & (bd_res != rd)).sum())
        inv = (is_d & (ba_res != ra)) | (is_a & (bd_res != ra)) | (is_d2 & (ba_res != rd)) | (is_a2 & (bd_res != rd))
        pk = np.where((is_d & (ba_res != ra)) | (is_d2 & (ba_res != rd)), ba_kc, bd_kc)
        n_bb = int((inv & (pk == 0)).sum()); n_sc = int((inv & (pk == 1)).sum()); n_w = int((inv & (pk == 2)).sum())
        comp = int(((bd_at == a_atm) & (ba_res != ra)).sum() + ((ba_at == b_atm) & (bd_res != rd)).sum())
        dv = (pos[b_atm] - pos[a_atm]); dv -= box * np.round(dv / box)
        dab = float(np.linalg.norm(dv))
        out[ci] = [n_out_d, n_in_d, n_out_a, n_in_a, n_out_d + n_in_d + n_out_a + n_in_a,
                   n_bb, n_sc, n_w, comp, dab, 1.0 if dab <= HB else 0.0,
                   n_out_d + n_in_d, n_out_a + n_in_a]
    prem["n_bonds_total"] = int(len(bl))
    return out, prem
