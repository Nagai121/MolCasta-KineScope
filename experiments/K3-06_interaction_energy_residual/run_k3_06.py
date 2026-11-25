"""K3-06: interaction energies at the initial structure vs the residual.

Question (K3 series): what physical state of a pair does the trajectory
know that the static structure does not? The residual ("zure") is the
main appearance: real 0-1 ns mean distance minus the static-only
prediction, built exactly as in K39/K42 (static 5 features, regressors
fitted on the 8 calibration proteins only; HistGBR 160/0.065/15/20/0.15,
seed 20261121, plus OLS).

For every K06(b) hydrogen-bond formation candidate (8 calibration
proteins, 13,808 candidates; 100 unseen proteins x 3 replicas, 163,074
candidates), at the initial structure (frame 0 of the history window),
compute from the ATLAS initial structure:
  - nonbonded interaction energy between the donor residue and the
    acceptor residue, electrostatics and van der Waals separately;
  - interaction energy of each of the two residues with the rest of
    the protein, electrostatics and van der Waals separately
    (partner included; partner-excluded variants also stored).

Force field: CHARMM36m, the force field of the ATLAS runs themselves
(ATLAS About page: GROMACS with CHARMM36m, July 2020 version).
Charges are taken from each protein's own .tpr (the topology of the
actual ATLAS run). LJ sigma/epsilon per atom type come from OpenMM's
charmm36.xml LennardJonesForce table.
Combining: sigma_ij = (sigma_i+sigma_j)/2, epsilon_ij = sqrt(eps_i*eps_j)
(GROMACS comb-rule 1). Verified against the run topology itself
(gmx dump of the ATLAS .tpr, 1uha_A and 3txs_C): the rule-1 pair table
matches the .tpr's C6/C12 for every protein atom-type pair except one,
the CHARMM36m salt-bridge NBFIX for NC2-OC, applied here from the .tpr
values (C6 = 3.00044753e-03 kJ nm^6/mol, C12 = 3.47228797e-06
kJ nm^12/mol). The NBFIX is the in-data fingerprint that the run used
CHARMM36m, not CHARMM36.
Coulomb: 1389.35456 * q_i q_j / r[Angstrom] kJ/mol. 1-2/1-3 pairs
(bond-graph distance <= 2) excluded; 1-4 pairs at full strength
(CHARMM coulomb14scale = lj14scale = 1.0).
Direct pairwise sums, minimum image with the frame-0 box, no cutoff;
the PME reciprocal part is not pairwise decomposable and is excluded.
Surroundings = the rest of the protein (the base data contains protein
atoms only; no water or ions).

Pre-fixed readings (K3 series): |correlation| >= 0.3 counts as
"related" (same line as K2-03). If adding a quantity to the static
prediction brings the residual SD to <= 0.30 Angstrom, the quantity
"explains most of the residual" (replica floor 0.249 Angstrom, measured by
K23 on D conditions; no strong-base remeasurement on record).
Shuffle control: quantities permuted within protein x replica groups
(seed 20261125); the effect must disappear.
"""
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.mdamath import triclinic_vectors
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

KC = 1389.35456
# CHARMM36m salt-bridge NBFIX for the NC2-OC atom-type pair, taken from
# the ATLAS run topology itself (gmx dump of the .tpr: c6 = 3.00044753e-03
# kJ nm^6/mol, c12 = 3.47228797e-06 kJ nm^12/mol; identical values in
# 1uha_A and 3txs_C). Converted to Angstrom units.
NBFIX_C6_A = 3000.44753    # kJ Angstrom^6 / mol
NBFIX_C12_A = 3472287.97   # kJ Angstrom^12 / mol
XML = '/home/hatch/workspace/mt-venv/lib/python3.12/site-packages/openmm/app/data/charmm36.xml'
SAVED = Path('/home/hatch/workspace/k06b/run/k06b_out')
ORIG = json.loads(Path('/home/hatch/workspace/k06b/results_100.json').read_text())
CAL, TEST = ORIG['calibration_proteins'], ORIG['test_proteins']
SRCROOT = Path('/home/hatch/workspace/kinescope/k3work/atlas-workspace')
WORK = Path('/home/hatch/workspace/kinescope/k3work/energy_out')
OUT = Path(__file__).resolve().parent
GBM_KW = dict(max_iter=160, learning_rate=.065, max_leaf_nodes=15,
              min_samples_leaf=20, l2_regularization=.15,
              random_state=20261121)
QUANT_NAMES = ['pair_elec', 'pair_vdw',
               'donor_rest_elec', 'donor_rest_vdw',
               'acceptor_rest_elec', 'acceptor_rest_vdw',
               'donor_rest_elec_excl', 'donor_rest_vdw_excl',
               'acceptor_rest_elec_excl', 'acceptor_rest_vdw_excl']


# ---------------- energy machinery (same as k3work/energies.py) -------------

def lj_table():
    root = ET.parse(XML).getroot()
    return {at.get('type'): (float(at.get('sigma')) * 10.0,
                             float(at.get('epsilon')))
            for at in root.find('LennardJonesForce').iter('Atom')}


def load_protein(srcdir, protein):
    src = Path(srcdir) / f'{protein}-analysis'
    up = mda.Universe(str(src / f'{protein}.pdb'))
    names = np.array([a.name for a in up.atoms])
    elems = np.array([str(a.element) for a in up.atoms])
    residx = np.array([a.residue.resindex for a in up.atoms], dtype=np.int64)
    n = len(names)
    ut = mda.Universe(str(src / f'{protein}_R1.tpr'))
    tpos = np.asarray(ut.atoms.positions[:n], dtype=np.float64)
    ppos = np.asarray(up.atoms.positions, dtype=np.float64)
    tresidx = np.array([a.residue.resindex for a in ut.atoms[:n]], dtype=np.int64)
    telems = np.array([str(a.element) for a in ut.atoms[:n]])
    perm = np.full(n, -1, dtype=np.int64)
    for r in np.unique(residx):
        pm = np.where(residx == r)[0]
        tm = np.where(tresidx == r)[0]
        if len(pm) != len(tm):
            raise ValueError(f'{protein}: residue {r} atom count differs')
        d = np.linalg.norm(ppos[pm][:, None, :] - tpos[tm][None, :, :], axis=2)
        d[elems[pm][:, None] != telems[tm][None, :]] = 1e9
        j = d.argmin(axis=1)
        if len(set(j.tolist())) != len(pm) or d[np.arange(len(pm)), j].max() > 0.05:
            raise ValueError(f'{protein}: residue {r} position matching failed')
        perm[pm] = tm[j]
    types = np.array([str(x) for x in ut.atoms.types])[perm]
    charges = np.asarray(ut.atoms.charges, dtype=np.float64)[perm]
    inv = np.full(n, -1, dtype=np.int64)
    inv[perm] = np.arange(n)
    bonds = np.array([(inv[i], inv[j]) for i, j in ut.bonds.indices
                      if i < n and j < n], dtype=np.int64)
    heavy = elems != 'H'
    return dict(residx=residx, types=types, charges=charges, bonds=bonds,
                source_atom=np.where(heavy)[0].astype(np.int64),
                point_residue=residx[heavy], n=n)


def frame0(srcdir, protein, replica):
    src = Path(srcdir) / f'{protein}-analysis'
    u = mda.Universe(str(src / f'{protein}.pdb'),
                     str(src / f'{protein}_R{replica}.xtc'))
    ts = u.trajectory[0]
    return (np.asarray(ts.positions, dtype=np.float64),
            triclinic_vectors(ts.dimensions).astype(np.float64))


def excluded_pairs(n, bonds):
    A = sparse.csr_matrix((np.ones(len(bonds)), (bonds[:, 0], bonds[:, 1])),
                          shape=(n, n))
    A = A + A.T
    ex = ((A > 0) + ((A @ A) > 0)).tocoo()
    keep = ex.row < ex.col
    return np.column_stack([ex.row[keep], ex.col[keep]]).astype(np.int64)


def _pair_energies(d, q1, q2, s1, s2, e1, e2, nb=None):
    r = np.sqrt(np.einsum('...i,...i->...', d, d))
    inv_r = np.where(r > 0, 1.0 / np.where(r > 0, r, 1.0), 0.0)
    elec = np.where(r > 0, KC * (q1 * q2) * inv_r, 0.0)
    sr6 = (0.5 * (s1 + s2) * inv_r) ** 6
    vdw = np.where(r > 0, 4.0 * np.sqrt(e1 * e2) * sr6 * (sr6 - 1.0), 0.0)
    if nb is not None:
        inv_r6 = inv_r ** 6
        vdw = np.where(nb, NBFIX_C12_A * inv_r6 ** 2 - NBFIX_C6_A * inv_r6, vdw)
        vdw = np.where(r > 0, vdw, 0.0)
    return elec, vdw


def _nbfix_mask(t1, t2):
    return ((t1 == 1) & (t2 == 2)) | ((t1 == 2) & (t2 == 1))


def residue_matrices(pos, box, q, sig, eps, residx, excl, types=None, chunk=256):
    n = len(pos)
    nres = int(residx.max()) + 1
    inv_box = np.linalg.inv(box)
    frac = pos @ inv_box
    Re = np.zeros((nres, nres))
    Rv = np.zeros((nres, nres))
    code = None
    if types is not None:
        code = (types == 'NC2').astype(np.int8) + 2 * (types == 'OC').astype(np.int8)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        df = frac[s:e, None, :] - frac[None, :, :]
        df -= np.round(df)
        d = df @ box
        nb = _nbfix_mask(code[s:e, None], code[None, :]) if code is not None else None
        elec, vdw = _pair_energies(d, q[s:e, None], q[None, :],
                                   sig[s:e, None], sig[None, :],
                                   eps[s:e, None], eps[None, :], nb)
        idx = (residx[s:e, None] * nres + residx[None, :]).ravel()
        Re += np.bincount(idx, weights=elec.ravel(),
                          minlength=nres * nres).reshape(nres, nres)
        Rv += np.bincount(idx, weights=vdw.ravel(),
                          minlength=nres * nres).reshape(nres, nres)
    if len(excl):
        i, j = excl[:, 0], excl[:, 1]
        df = frac[i] - frac[j]
        df -= np.round(df)
        nb = _nbfix_mask(code[i], code[j]) if code is not None else None
        elec, vdw = _pair_energies(df @ box, q[i], q[j], sig[i], sig[j],
                                   eps[i], eps[j], nb)
        np.subtract.at(Re, (residx[i], residx[j]), elec)
        np.subtract.at(Re, (residx[j], residx[i]), elec)
        np.subtract.at(Rv, (residx[i], residx[j]), vdw)
        np.subtract.at(Rv, (residx[j], residx[i]), vdw)
    return Re, Rv


def mic_distances(pos, box, a, b):
    frac = pos @ np.linalg.inv(box)
    df = frac[a] - frac[b]
    df -= np.round(df)
    d = df @ box
    return np.sqrt(np.einsum('ij,ij->i', d, d))


# ---------------- stage 1: energies per protein -----------------------------

def compute_protein(protein, tab):
    srcdir = SRCROOT / protein / 'sources' / 'ATLAS'
    P = load_protein(srcdir, protein)
    missing = sorted(set(P['types']) - set(tab))
    if missing:
        raise ValueError(f'{protein}: LJ types missing: {missing}')
    sig = np.array([tab[t][0] for t in P['types']])
    eps = np.array([tab[t][1] for t in P['types']])
    excl = excluded_pairs(P['n'], P['bonds'])
    z = np.load(SAVED / f'{protein}.npz')
    out = {'charge_sum': float(P['charges'].sum()), 'n_atoms': int(P['n']),
           'n_residues': int(P['residx'].max()) + 1}
    for r in (1, 2, 3):
        pos, box = frame0(srcdir, protein, r)
        pair = z[f'r{r}_pair']
        hist = z[f'r{r}_history'].astype(np.float64)
        da = P['source_atom'][pair[:, 0].astype(int)]
        db = P['source_atom'][pair[:, 1].astype(int)]
        d0 = mic_distances(pos, box, da, db)
        maxdiff = float(np.abs(d0 - hist[:, 0]).max())
        if maxdiff > 0.05:
            raise ValueError(f'{protein} R{r}: frame0 distance mismatch {maxdiff}')
        Re, Rv = residue_matrices(pos, box, P['charges'], sig, eps,
                                  P['residx'], excl, P['types'])
        dres = P['point_residue'][pair[:, 0].astype(int)]
        ares = P['point_residue'][pair[:, 1].astype(int)]
        pe, pv = Re[dres, ares], Rv[dres, ares]
        dre = Re.sum(1)[dres] - Re[dres, dres]
        drv = Rv.sum(1)[dres] - Rv[dres, dres]
        are = Re.sum(1)[ares] - Re[ares, ares]
        arv = Rv.sum(1)[ares] - Rv[ares, ares]
        Q = np.column_stack([pe, pv, dre, drv, are, arv,
                             dre - pe, drv - pv, are - pe, arv - pv])
        out[f'r{r}_quant'] = Q.astype(np.float64)
        out[f'r{r}_dres'] = dres
        out[f'r{r}_ares'] = ares
        out[f'r{r}_maxdiff'] = maxdiff
    np.savez_compressed(WORK / f'{protein}.npz', **out)
    return out


def stage1():
    WORK.mkdir(parents=True, exist_ok=True)
    tab = lj_table()
    done = []
    for p in CAL + TEST:
        f = WORK / f'{p}.npz'
        if f.exists():
            done.append(p)
            continue
        tpr = SRCROOT / p / 'sources' / 'ATLAS' / f'{p}-analysis' / f'{p}_R1.tpr'
        if not tpr.exists():
            print(f'PENDING (not downloaded) {p}', flush=True)
            continue
        t0 = time.time()
        compute_protein(p, tab)
        done.append(p)
        print(f'[{len(done)}/108] {p} {time.time()-t0:.1f}s', flush=True)
    print('stage1 complete:', len(done))


# ---------------- stage 2: analysis ------------------------------------------

def load_rows(proteins):
    rows = []
    for p in proteins:
        with np.load(SAVED / f'{p}.npz') as z, \
             np.load(WORK / f'{p}.npz') as w:
            for r in (1, 2, 3):
                rows.append(dict(protein=p, replica=r,
                                 static=z[f'r{r}_static'].astype(np.float64),
                                 mean=z[f'r{r}_history'][:, 1].astype(np.float64),
                                 quant=w[f'r{r}_quant'],
                                 dres=w[f'r{r}_dres'], ares=w[f'r{r}_ares']))
    return rows


def sanitize(o):
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def stage2():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    cal = load_rows(CAL)
    test = load_rows(TEST)
    Str = np.concatenate([r['static'] for r in cal])
    Ste = np.concatenate([r['static'] for r in test])
    m_cal = np.concatenate([r['mean'] for r in cal])
    m_te = np.concatenate([r['mean'] for r in test])
    Q_cal = np.concatenate([r['quant'] for r in cal])
    Q_te = np.concatenate([r['quant'] for r in test])
    groups = np.concatenate([np.full(len(r['mean']), i)
                             for i, r in enumerate(test)])
    assert len(m_cal) == 13808 and len(m_te) == 163074, (len(m_cal), len(m_te))

    res = {'n_cal': int(len(m_cal)), 'n_test': int(len(m_te))}
    # residual, K39's construction
    resid = {}
    for method in ('gbm', 'ols'):
        mk = (HistGradientBoostingRegressor(**GBM_KW) if method == 'gbm'
              else LinearRegression())
        pred = mk.fit(Str, m_cal).predict(Ste)
        resid[method] = m_te - pred
        res[f'residual_{method}'] = {
            'r2_on_test': float(r2_score(m_te, pred)),
            'residual_sd': float(resid[method].std()),
            'residual_mean': float(resid[method].mean())}
        print(method, res[f'residual_{method}'], flush=True)
    R = resid['gbm']

    # quantity distributions + correlations
    derived = {'pair_total': Q_te[:, 0] + Q_te[:, 1],
               'rest_elec_sum': Q_te[:, 2] + Q_te[:, 4],
               'rest_vdw_sum': Q_te[:, 3] + Q_te[:, 5],
               'rest_total_sum': Q_te[:, 2] + Q_te[:, 3] + Q_te[:, 4] + Q_te[:, 5]}
    allq = {name: Q_te[:, i] for i, name in enumerate(QUANT_NAMES)}
    allq.update(derived)
    res['quantities'] = {}
    for name, v in allq.items():
        e = {'mean': float(v.mean()), 'sd': float(v.std()),
             'min': float(v.min()), 'max': float(v.max()),
             'pearson_gbm': float(pearsonr(v, R)[0]),
             'spearman_gbm': float(spearmanr(v, R)[0]),
             'pearson_ols': float(pearsonr(v, resid['ols'])[0])}
        res['quantities'][name] = e
        print(name, {k: round(x, 4) for k, x in e.items()}, flush=True)

    # per-protein correlations (GBM residual), proteins with >= 30 candidates
    prot_of = np.concatenate([np.full(len(r['mean']), r['protein'])
                              for r in test])
    pp = {}
    for name in QUANT_NAMES[:6]:
        v = allq[name]
        vals = []
        for p in TEST:
            m = prot_of == p
            if m.sum() >= 30 and v[m].std() > 0:
                vals.append(pearsonr(v[m], R[m])[0])
        vals = np.array(vals)
        pp[name] = {'n_proteins': int(len(vals)),
                    'median': float(np.median(vals)),
                    'q25': float(np.percentile(vals, 25)),
                    'q75': float(np.percentile(vals, 75)),
                    'frac_abs_ge_0.3': float((np.abs(vals) >= 0.3).mean())}
    res['per_protein_pearson'] = pp

    # augmented fits: static + quantity sets, fitted on cal
    sets = {f'single_{n}': [i] for i, n in enumerate(QUANT_NAMES[:6])}
    sets['pair2'] = [0, 1]
    sets['rest4'] = [2, 3, 4, 5]
    sets['all6'] = [0, 1, 2, 3, 4, 5]
    sets['all10'] = list(range(10))
    base_sd = res['residual_gbm']['residual_sd']
    fits = {}
    models = {}
    for sname, cols in sets.items():
        Xtr = np.column_stack([Str, Q_cal[:, cols]])
        Xte = np.column_stack([Ste, Q_te[:, cols]])
        mdl = HistGradientBoostingRegressor(**GBM_KW).fit(Xtr, m_cal)
        sd = float((m_te - mdl.predict(Xte)).std())
        fits[sname] = {'cols': [QUANT_NAMES[c] for c in cols],
                       'residual_sd_gbm': sd}
        models[sname] = (mdl, cols)
        print('fit', sname, round(sd, 4), flush=True)
    for sname in ('pair2', 'rest4', 'all6'):
        cols = sets[sname]
        mdl = LinearRegression().fit(np.column_stack([Str, Q_cal[:, cols]]), m_cal)
        sd = float((m_te - mdl.predict(np.column_stack([Ste, Q_te[:, cols]]))).std())
        fits[sname]['residual_sd_ols'] = sd
    res['augmented_fits'] = fits
    res['baseline_residual_sd_gbm'] = base_sd

    # shuffle control: permute quantities within protein x replica (test only)
    rng = np.random.default_rng(20261125)
    Q_sh = Q_te.copy()
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        Q_sh[m] = Q_sh[m][rng.permutation(len(m))]
    sh = {}
    for name in QUANT_NAMES[:6]:
        v = Q_sh[:, QUANT_NAMES.index(name)]
        sh[f'corr_{name}'] = float(pearsonr(v, R)[0])
    for sname in ('pair2', 'rest4', 'all6'):
        mdl, cols = models[sname]
        Xte = np.column_stack([Ste, Q_sh[:, cols]])
        sh[f'sd_{sname}'] = float((m_te - mdl.predict(Xte)).std())
    res['shuffle_control'] = sh
    print('shuffle', {k: round(v, 4) for k, v in sh.items()}, flush=True)

    # premise: frame-0 distance identity over all proteins
    md = []
    for p in CAL + TEST:
        with np.load(WORK / f'{p}.npz') as w:
            md += [float(w[f'r{r}_maxdiff']) for r in (1, 2, 3)]
    res['premise_frame0_distance_maxdiff'] = float(np.max(md))
    same = int(sum(int((r['dres'] == r['ares']).sum()) for r in test))
    res['same_residue_candidates_test'] = same

    (OUT / 'results.json').write_text(json.dumps(sanitize(res), indent=2))
    import csv
    with open(OUT / 'results.csv', 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['quantity', 'mean', 'sd', 'min', 'max',
                       'pearson_gbm', 'spearman_gbm', 'pearson_ols'])
        for name, e in res['quantities'].items():
            wcsv.writerow([name, e['mean'], e['sd'], e['min'], e['max'],
                           e['pearson_gbm'], e['spearman_gbm'], e['pearson_ols']])
    np.savez_compressed(
        OUT / 'candidates.npz',
        protein=prot_of,
        replica=np.concatenate([np.full(len(r['mean']), r['replica'])
                                for r in test]),
        residual_gbm=R, residual_ols=resid['ols'], real_mean=m_te,
        quant=Q_te, quant_names=np.array(QUANT_NAMES),
        donor_res=np.concatenate([r['dres'] for r in test]),
        acceptor_res=np.concatenate([r['ares'] for r in test]))
    print('stage2 saved to', OUT)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('stage1', 'all'):
        stage1()
    if which in ('stage2', 'all'):
        stage2()
