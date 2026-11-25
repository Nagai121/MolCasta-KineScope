"""CPU feature kernels adapted from Internal Research rich_build and T1_02."""
import numpy as np

HB_DIST = 3.5

DONOR_N = {"LYS": {"NZ"}, "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"},
           "GLN": {"NE2"}, "HIS": {"ND1", "NE2"}, "TRP": {"NE1"}}
DONOR_O = {"SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}}
ACCEPTOR_O = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}, "ASN": {"OD1"},
              "GLN": {"OE1"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}}
ANCHOR = {("LYS", "NZ"): "CE", ("ARG", "NE"): "CZ", ("ARG", "NH1"): "CZ", ("ARG", "NH2"): "CZ",
          ("ASN", "ND2"): "CG", ("GLN", "NE2"): "CD", ("HIS", "ND1"): "CG", ("HIS", "NE2"): "CE1",
          ("TRP", "NE1"): "CE2", ("SER", "OG"): "CB", ("THR", "OG1"): "CB", ("TYR", "OH"): "CZ",
          ("ASP", "OD1"): "CG", ("ASP", "OD2"): "CG", ("GLU", "OE1"): "CD", ("GLU", "OE2"): "CD",
          ("ASN", "OD1"): "CG", ("GLN", "OE1"): "CD"}
def _mic(d, box):
    if np.ndim(box) == 1:
        return d - box * np.round(d / box)
    inverse = np.linalg.inv(box)
    fractional = d @ inverse
    central = fractional - np.round(fractional)
    best = central @ box
    best_norm = np.einsum("...i,...i->...", best, best)
    for offset in np.ndindex(3, 3, 3):
        if offset == (1, 1, 1):
            continue
        shifted = (central + np.asarray(offset) - 1) @ box
        norm = np.einsum("...i,...i->...", shifted, shifted)
        better = norm < best_norm
        best[better] = shifted[better]
        best_norm[better] = norm[better]
    return best

def _pd(a, b, box):
    d = _mic(a[:, None, :] - b[None, :, :], box)
    return np.sqrt((d * d).sum(axis=2))

def _anchor(maps_k, rn, atom_name, fallback_names=("CA",)):
    want = ANCHOR.get((rn, atom_name))
    if want and want in maps_k:
        return maps_k[want]
    for f in fallback_names:
        if f in maps_k:
            return maps_k[f]
    return maps_k["CA"]


AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AAI = {a: i for i, a in enumerate(AA)}
HYDRO = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR"}
POS = {"LYS", "ARG", "HIS"}
NEG = {"ASP", "GLU"}
AROM = {"PHE", "TRP", "TYR", "HIS"}

RICH_NAMES = (
    [f"g_d{i}a{j}" for i in range(5) for j in range(5)]
    + ["g_minheavy", "g_cbcb", "g_centroid"]
    + ["c_d_near_other", "c_d_cnt6", "c_a_near_other", "c_a_cnt6",
       "c_d_near_res", "c_a_near_res"]
    + ["dih_phi_i", "dih_psi_i", "dih_phi_j", "dih_psi_j",
       "dih_phi_i_s", "dih_psi_i_s", "dih_phi_j_s", "dih_psi_j_s"]
    + ["env_hb_i", "env_hb_j", "env_hb_box", "env_hb_near_i"]
    + ["pack_d6", "pack_a6", "pack_d4", "pack_a4"]
    + [f"type_di_{a}" for a in AA] + [f"type_aj_{a}" for a in AA]
    + ["type_di_hydro", "type_di_pos", "type_di_neg", "type_di_arom",
       "type_aj_hydro", "type_aj_pos", "type_aj_neg", "type_aj_arom",
       "type_scsize_i", "type_scsize_j", "type_aj_pro", "type_di_gly",
       "type_aj_gly", "type_saltcap"])


def _dihedral(p0, p1, p2, p3, box):
    b0 = _mic(p1 - p0, box); b1 = _mic(p2 - p1, box); b2 = _mic(p3 - p2, box)
    b1n = b1 / np.maximum(1e-9, np.linalg.norm(b1, axis=1, keepdims=True))
    v = b0 - (b0 * b1n).sum(axis=1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(axis=1, keepdims=True) * b1n
    x = (v * w).sum(axis=1)
    y = (np.cross(b1n, v) * w).sum(axis=1)
    return np.arctan2(y, x)


def prepare(maps, rnames, by_res, keep, n_atoms):
    n = len(maps)
    is_pro = np.asarray(rnames) == "PRO"
    res_atoms = [np.array(by_res[keep[k]]) for k in range(n)]
    res_ord = np.full(n_atoms, -1, dtype=int)
    for k in range(n):
        res_ord[res_atoms[k]] = k
    cb = np.array([maps[k].get("CB", maps[k]["CA"]) for k in range(n)])
    sc_size = np.array([len(res_atoms[k]) for k in range(n)], dtype=np.float64)
    aa_idx = np.array([AAI.get(rnames[k], -1) for k in range(n)])

    donors, acceptors = [], []
    for k in range(n):
        rn = rnames[k]
        if not is_pro[k]:
            donors.append((maps[k]["N"], k, 0, maps[k]["CA"]))
        for an in sorted(DONOR_N.get(rn, ())):
            if an in maps[k]:
                donors.append((maps[k][an], k, 1, _anchor(maps[k], rn, an)))
        for an in sorted(DONOR_O.get(rn, ())):
            if an in maps[k]:
                donors.append((maps[k][an], k, 2, _anchor(maps[k], rn, an)))
        acceptors.append((maps[k]["O"], k, 0, maps[k]["C"]))
        if "OXT" in maps[k]:
            acceptors.append((maps[k]["OXT"], k, 0, maps[k]["C"]))
        for an in sorted(ACCEPTOR_O.get(rn, ())):
            if an in maps[k]:
                acceptors.append((maps[k][an], k, 1, _anchor(maps[k], rn, an)))
    d_at = np.array([d[0] for d in donors]); d_res = np.array([d[1] for d in donors])
    d_kind = np.array([d[2] for d in donors]); d_anc = np.array([d[3] for d in donors])
    a_at = np.array([a[0] for a in acceptors]); a_res = np.array([a[1] for a in acceptors])
    a_kind = np.array([a[2] for a in acceptors]); a_anc = np.array([a[3] for a in acceptors])
    diff_res = d_res[:, None] != a_res[None, :]

    return (res_atoms, res_ord, cb, sc_size, aa_idx, d_at, d_res, d_kind, d_anc, a_at, a_res, a_kind, a_anc, diff_res)


def structural(P, box, maps, rnames, context, ii, jj):
    res_atoms, res_ord, cb, sc_size, aa_idx, d_at, d_res, d_kind, d_anc, a_at, a_res, a_kind, a_anc, diff_res = context
    n = len(maps)
    ar = np.arange(n)
    seq = np.abs(np.subtract.outer(ar, ar))
    CAi = np.array([m["CA"] for m in maps])
    if not len(ii):
        return np.empty((0, 118), dtype=np.float32)
    dca = _pd(P[CAi], P[CAi], box)
    burial = ((dca <= 8.0) & (seq >= 1)).sum(axis=1)
    D = _pd(P[d_at], P[a_at], box)
    bonded = (D <= HB_DIST) & diff_res
    rd, ra = d_res[ii], a_res[jj]
    vA = _mic(P[a_anc[jj]] - P[a_at[jj]], box); vAD = _mic(P[d_at[ii]] - P[a_at[jj]], box)
    cosA = (vA * vAD).sum(axis=1) / np.maximum(1e-9, np.linalg.norm(vA, axis=1) * np.linalg.norm(vAD, axis=1))
    vD = _mic(P[d_anc[ii]] - P[d_at[ii]], box); vDA = -vAD
    cosD = (vD * vDA).sum(axis=1) / np.maximum(1e-9, np.linalg.norm(vD, axis=1) * np.linalg.norm(vDA, axis=1))
    near = (D <= 4.0) & diff_res
    cols = [D[ii, jj], dca[rd, ra], np.log(np.maximum(seq[rd, ra], 1).astype(np.float64)),
            (d_kind[ii] == 0).astype(np.float64), (d_kind[ii] == 2).astype(np.float64),
            (a_kind[jj] == 0).astype(np.float64), cosA, cosD,
            near.sum(axis=1)[ii] - near[ii, jj], near.sum(axis=0)[jj] - near[ii, jj],
            bonded.any(axis=1)[ii].astype(np.float64), bonded.any(axis=0)[jj].astype(np.float64),
            burial[rd].astype(np.float64), burial[ra].astype(np.float64)]
    base = np.column_stack(cols).astype(np.float32)

    # ---- rich blocks ----
    d_atoms = [d_at[ii], d_anc[ii], CAi[rd],
               np.array([maps[k]["N"] for k in rd]), np.array([maps[k]["C"] for k in rd])]
    a_atoms = [a_at[jj], a_anc[jj], CAi[ra],
               np.array([maps[k]["C"] for k in ra]), np.array([maps[k]["O"] for k in ra])]
    g = [np.linalg.norm(_mic(P[d_atoms[a]] - P[a_atoms[b]], box), axis=1)
         for a in range(5) for b in range(5)]
    Dall = _pd(P, P, box)
    mapped = np.flatnonzero(res_ord >= 0)
    mapped_residues = res_ord[mapped]
    pair_id = mapped_residues[:, None] * n + mapped_residues[None, :]
    minheavy = np.full(n * n, np.inf)
    np.minimum.at(minheavy, pair_id.ravel(), Dall[np.ix_(mapped, mapped)].ravel())
    minheavy = minheavy.reshape(n, n)
    cents = np.stack([P[res_atoms[k]].mean(axis=0) for k in range(n)])
    g += [minheavy[rd, ra],
          np.linalg.norm(_mic(P[cb[rd]] - P[cb[ra]], box), axis=1),
          np.linalg.norm(_mic(cents[rd] - cents[ra], box), axis=1)]
    # competitors from the full D matrix
    def fin(x):
        return np.where(np.isinf(x), 12.0, x)

    def alternative(matrix, rows, cols):
        if matrix.shape[1] < 2:
            return np.full(len(rows), 12.0)
        nearest = matrix.argmin(axis=1)
        best_two = np.partition(matrix, 1, axis=1)[:, :2]
        return fin(best_two[rows, (nearest[rows] == cols).astype(np.intp)])

    Dmod = np.where(diff_res, D, np.inf)
    c = [alternative(Dmod, ii, jj), (Dmod <= 6.0).sum(axis=1)[ii],
         alternative(Dmod.T, jj, ii), (Dmod <= 6.0).sum(axis=0)[jj]]
    Dres = np.full(n * n, np.inf)
    np.minimum.at(Dres, (d_res[:, None] * n + a_res[None, :]).ravel(), D.ravel())
    Dres = Dres.reshape(n, n)
    np.fill_diagonal(Dres, np.inf)
    c += [alternative(Dres, rd, ra), alternative(Dres.T, ra, rd)]
    # dihedrals per residue
    def dih_for(kind):
        out = np.zeros(n)
        for k in range(n):
            try:
                if kind == "phi" and k > 0:
                    out[k] = _dihedral(P[maps[k - 1]["C"]][None], P[maps[k]["N"]][None],
                                       P[maps[k]["CA"]][None], P[maps[k]["C"]][None], box)[0]
                elif kind == "psi" and k < n - 1:
                    out[k] = _dihedral(P[maps[k]["N"]][None], P[maps[k]["CA"]][None],
                                       P[maps[k]["C"]][None], P[maps[k + 1]["N"]][None], box)[0]
            except Exception:
                out[k] = 0.0
        return out
    phi, psi = dih_for("phi"), dih_for("psi")
    dih = [np.cos(phi[rd]), np.cos(psi[rd]), np.cos(phi[ra]), np.cos(psi[ra]),
           np.sin(phi[rd]), np.sin(psi[rd]), np.sin(phi[ra]), np.sin(psi[ra])]
    # local HB environment (residue-level bonded matrix)
    Rb = np.zeros((n, n), dtype=np.float64)
    bi, bj = np.where(bonded)
    np.maximum.at(Rb, (d_res[bi], a_res[bj]), 1.0)
    hb_per_res = Rb.sum(axis=1) + Rb.sum(axis=0)
    cs = np.zeros((n + 1, n + 1))
    cs[1:, 1:] = Rb.cumsum(axis=0).cumsum(axis=1)
    def boxsum(i, j, r=2):
        i0, i1 = max(0, i - r), min(n, i + r + 1)
        j0, j1 = max(0, j - r), min(n, j + r + 1)
        return (cs[i1, j1] - cs[i0, j1] - cs[i1, j0] + cs[i0, j0])
    env = [hb_per_res[rd], hb_per_res[ra],
           np.array([boxsum(a, b) for a, b in zip(rd, ra)]),
           np.array([boxsum(a, a) for a in rd])]
    pack = [(Dall <= 6.0).sum(axis=1)[d_at[ii]].astype(np.float64) - 1,
            (Dall <= 6.0).sum(axis=1)[a_at[jj]].astype(np.float64) - 1,
            (Dall <= 4.0).sum(axis=1)[d_at[ii]].astype(np.float64) - 1,
            (Dall <= 4.0).sum(axis=1)[a_at[jj]].astype(np.float64) - 1]
    oh_i = np.zeros((len(ii), 20)); oh_j = np.zeros((len(ii), 20))
    vi = aa_idx[rd] >= 0; vj = aa_idx[ra] >= 0
    oh_i[np.where(vi)[0], aa_idx[rd][vi]] = 1.0
    oh_j[np.where(vj)[0], aa_idx[ra][vj]] = 1.0
    def cls(k, s):
        return np.array([1.0 if rnames[x] in s else 0.0 for x in k])
    typ = [oh_i, oh_j,
           cls(rd, HYDRO)[:, None], cls(rd, POS)[:, None], cls(rd, NEG)[:, None],
           cls(rd, AROM)[:, None], cls(ra, HYDRO)[:, None], cls(ra, POS)[:, None],
           cls(ra, NEG)[:, None], cls(ra, AROM)[:, None],
           sc_size[rd][:, None], sc_size[ra][:, None],
           (np.array([rnames[x] for x in ra]) == "PRO").astype(np.float64)[:, None],
           (np.array([rnames[x] for x in rd]) == "GLY").astype(np.float64)[:, None],
           (np.array([rnames[x] for x in ra]) == "GLY").astype(np.float64)[:, None],
           ((cls(rd, POS) * cls(ra, NEG)) + (cls(rd, NEG) * cls(ra, POS)))[:, None]]
    blocks = ([np.asarray(x, dtype=np.float64)[:, None] if np.ndim(x) == 1 else x
               for x in g + c + dih + env + pack] + typ)
    return np.column_stack([base, *blocks]).astype(np.float32)


GRAPH_COLS=("d_count","a_count","count_total","count_abs_diff",
            "d_competitor_mass","a_competitor_mass","mass_total","mass_abs_diff",
            "d_best_distance","a_best_distance","best_distance_abs_diff",
            "d_competitor_mean_distance","a_competitor_mean_distance",
            "d_best_score","a_best_score","best_score_abs_diff",
            "relative_nearest_d","relative_nearest_a",
            "compete_ratio_d","compete_ratio_a","compete_ratio_geom",
            "diffuse_mass_1","diffuse_mass_2","diffuse_rank_1")


def feature_graph(x,donors,acceptors,origins,n_atoms):
    """Aggregate typed endpoint competition across all retained ROD options."""
    if not len(x):
        return np.empty((0, 24), dtype=np.float32)
    assert len(x)==len(donors)==len(acceptors)==len(origins)
    assert not len(x) or np.all(np.diff(origins)>=0)
    extra=np.empty((len(x),len(GRAPH_COLS)),dtype=np.float32)
    places=np.flatnonzero(np.r_[True,origins[1:]!=origins[:-1],True])
    for start,stop in zip(places[:-1],places[1:]):
        d=donors[start:stop].astype(np.int64)
        a=acceptors[start:stop].astype(np.int64)
        dist=x[start:stop,0].astype(np.float64)
        prox=np.exp(np.clip((3.5-dist)/.95,-20,0))
        dn=np.bincount(d,minlength=n_atoms).astype(np.float64)
        an=np.bincount(a,minlength=n_atoms).astype(np.float64)
        md=np.bincount(d,weights=prox,minlength=n_atoms)
        ma=np.bincount(a,weights=prox,minlength=n_atoms)
        ds=np.bincount(d,weights=dist,minlength=n_atoms)
        ass=np.bincount(a,weights=dist,minlength=n_atoms)
        maxd=np.zeros(n_atoms);maxa=np.zeros(n_atoms)
        np.maximum.at(maxd,d,prox)
        np.maximum.at(maxa,a,prox)
        mind=np.full(n_atoms,np.inf);mina=np.full(n_atoms,np.inf)
        np.minimum.at(mind,d,dist)
        np.minimum.at(mina,a,dist)
        dcount=np.maximum(dn[d]-1,0)
        acount=np.maximum(an[a]-1,0)
        dmass=np.maximum(md[d]-prox,0)
        amass=np.maximum(ma[a]-prox,0)
        bestd=mind[d];besta=mina[a]
        mean_d=np.where(dcount>0,(ds[d]-dist)/np.maximum(dcount,1),dist)
        mean_a=np.where(acount>0,(ass[a]-dist)/np.maximum(acount,1),dist)
        competition_d=prox/(prox+dmass+1e-4)
        competition_a=prox/(prox+amass+1e-4)
        # Two contextual relation updates: probability mass distributed across
        # each endpoint's alternative candidates, iterated over the graph.
        current=prox.copy()
        layers=[]
        for _ in range(2):
            dsum=np.bincount(d,weights=current,minlength=n_atoms)[d]
            asum=np.bincount(a,weights=current,minlength=n_atoms)[a]
            alternative=np.maximum(dsum+asum-2*current,0)
            current=prox/(1+alternative)
            layers.append(alternative)
        value=np.column_stack((
            np.log1p(dcount),np.log1p(acount),
            np.log1p(dcount+acount),np.log1p(np.abs(dcount-acount)),
            np.log1p(dmass),np.log1p(amass),
            np.log1p(dmass+amass),np.log1p(np.abs(dmass-amass)),
            bestd,besta,np.abs(bestd-besta),
            mean_d,mean_a,
            maxd[d],maxa[a],np.abs(maxd[d]-maxa[a]),
            dist-bestd,dist-besta,
            competition_d,competition_a,
            np.sqrt(competition_d*competition_a),
            np.log1p(layers[0]),np.log1p(layers[1]),current
        ))
        assert value.shape==(stop-start,len(GRAPH_COLS))
        extra[start:stop]=value.astype(np.float32)
    assert np.isfinite(extra).all()
    return extra



REL_COLS = tuple(range(24))
def enriched_features(X):
    """Origin-only relation interactions over all retained ROD options.

    Base columns 118:142 already aggregate a full graph of donor/acceptor
    alternatives (not gate-selected). This block adds higher-order
    competition and geometry interactions without any target information.
    """
    z=np.asarray(X,dtype=np.float32)
    assert z.shape[1]==142
    g=z[:,118:].astype(np.float64)
    eps=1e-5
    distance=np.maximum(z[:,0].astype(np.float64),eps)
    prox=np.exp(np.clip((3.5-distance)/.95,-20,0))
    d_count=np.expm1(g[:,0]);a_count=np.expm1(g[:,1])
    mass_d=np.expm1(g[:,4]);mass_a=np.expm1(g[:,5])
    best_d=g[:,8];best_a=g[:,9]
    mean_d=g[:,11];mean_a=g[:,12]
    max_d=g[:,13];max_a=g[:,14]
    advantage_d=distance-best_d;advantage_a=distance-best_a
    c_d=g[:,18];c_a=g[:,19]
    first_mass=np.expm1(g[:,21]);second_mass=np.expm1(g[:,22])
    ratio=(mass_d+eps)/(mass_a+eps)
    rival=1-np.sqrt(np.clip(c_d*c_a,0,1))
    burial_d=np.maximum(z[:,12].astype(np.float64),0)
    burial_a=np.maximum(z[:,13].astype(np.float64),0)
    angle_d=z[:,7].astype(np.float64)
    angle_a=z[:,6].astype(np.float64)
    fields=[
        advantage_d,advantage_a,
        np.sqrt((d_count+1)*(a_count+1)),
        np.log1p(d_count*a_count),
        .5*(mass_d/(d_count+eps)+mass_a/(a_count+eps)),
        np.abs(mass_d/(d_count+eps)-mass_a/(a_count+eps)),
        prox/(max_d+eps),prox/(max_a+eps),
        mass_d/(mass_d+prox+eps),mass_a/(mass_a+prox+eps),
        np.log1p(np.abs(first_mass-second_mass)),
        (second_mass+eps)/(first_mass+eps),
        np.log1p(d_count)*prox,np.log1p(a_count)*prox,
        rival*.5*(1+np.abs(advantage_d-advantage_a)),
        .5*(max_d+max_a),
        rival*prox,rival*distance,
        np.log1p(burial_d)*np.log1p(d_count+1),
        np.log1p(burial_a)*np.log1p(a_count+1),
        angle_a*angle_d/distance,
        (angle_a+angle_d)*prox,
        np.log1p(d_count+a_count)*(1-c_d*c_a),
        rival*(.5*(mean_d+mean_a)-distance),
    ]
    extra=np.column_stack(fields).astype(np.float32)
    assert extra.shape==(len(z),len(REL_COLS))
    assert np.isfinite(extra).all()
    return np.concatenate((z,extra),axis=1)
