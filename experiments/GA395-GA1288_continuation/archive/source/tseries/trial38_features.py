"""Historical Trial38 rich features on an origin frame (Internal Research rich_build.py)."""
import numpy as np

from .features import AA, AROM, HYDRO, NEG, POS, _dihedral

HB_DIST = 3.5
HB_REACH = 6.0


def _mic(d, box):
    return d - box * np.round(d / box)


def _pd(a, b, box):
    d = _mic(a[:, None, :] - b[None, :, :], box)
    return np.sqrt((d * d).sum(axis=2))


def historical_hb_features(P, box, topo):
    maps, rnames, context, _ = topo
    (res_atoms, res_ord, cb, sc_size, aa_idx, d_at, d_res, d_kind, d_anc,
     a_at, a_res, a_kind, a_anc, diff_res) = context
    P = np.asarray(P, dtype=np.float64)
    res_ord = np.maximum(res_ord, 0)
    n = len(maps)
    ar = np.arange(n)
    seq = np.abs(np.subtract.outer(ar, ar))
    CAi = np.array([m["CA"] for m in maps])
    dca = _pd(P[CAi], P[CAi], box)
    burial = ((dca <= 8.0) & (seq >= 1)).sum(axis=1)
    D = _pd(P[d_at], P[a_at], box)
    bonded = (D <= HB_DIST) & diff_res
    cand = (D <= HB_REACH) & diff_res
    ii, jj = np.where(cand)
    if not len(ii):
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 118), dtype=np.float32)
    rd, ra = d_res[ii], a_res[jj]
    vA = _mic(P[a_anc[jj]] - P[a_at[jj]], box); vAD = _mic(P[d_at[ii]] - P[a_at[jj]], box)
    cosA = (vA * vAD).sum(axis=1) / np.maximum(1e-9, np.linalg.norm(vA, axis=1) * np.linalg.norm(vAD, axis=1))
    vD = _mic(P[d_anc[ii]] - P[d_at[ii]], box); vDA = -vAD
    cosD = (vD * vDA).sum(axis=1) / np.maximum(1e-9, np.linalg.norm(vD, axis=1) * np.linalg.norm(vDA, axis=1))
    near = (D <= 4.0) & diff_res
    cols = [D[ii, jj], dca[rd, ra], np.log(np.maximum(seq[rd, ra], 1).astype(np.float64)),
            (d_kind[ii] == 0).astype(np.float64), (d_kind[ii] == 2).astype(np.float64),
            (a_kind[jj] == 0).astype(np.float64), cosA, cosD,
            (near.sum(axis=1)[ii] - 1).astype(np.float64), (near.sum(axis=0)[jj] - 1).astype(np.float64),
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
    pair_id = res_ord[:, None] * n + res_ord[None, :]
    minheavy = np.full(n * n, np.inf)
    np.minimum.at(minheavy, pair_id.ravel(), Dall.ravel())
    minheavy = minheavy.reshape(n, n)
    cents = np.stack([P[res_atoms[k]].mean(axis=0) for k in range(n)])
    g += [minheavy[rd, ra],
          np.linalg.norm(_mic(P[cb[rd]] - P[cb[ra]], box), axis=1),
          np.linalg.norm(_mic(cents[rd] - cents[ra], box), axis=1)]
    # competitors from the full D matrix
    Dmod = D.copy()
    Dmod[ii, jj] = np.inf
    def fin(x):
        return np.where(np.isinf(x), 12.0, x)
    c = [fin(Dmod.min(axis=1)[ii]), (D <= 6.0).sum(axis=1)[ii].astype(np.float64),
         fin(Dmod.min(axis=0)[jj]), (D <= 6.0).sum(axis=0)[jj].astype(np.float64)]
    Dres = np.full(n * n, np.inf)
    np.minimum.at(Dres, (d_res[:, None] * n + a_res[None, :]).ravel(), D.ravel())
    Dres = Dres.reshape(n, n)
    Dres[rd, ra] = np.inf
    c += [fin(Dres.min(axis=1)[rd]), fin(Dres.min(axis=0)[ra])]
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
    unformed = ~bonded[ii, jj]
    return (np.column_stack((d_at[ii], a_at[jj]))[unformed],
            np.column_stack([base, *blocks]).astype(np.float32)[unformed])
