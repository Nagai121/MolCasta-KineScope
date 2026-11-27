"""L3 analysis, mirroring E03's analyze() exactly (same definitions, same
hyper-parameters, same GroupKFold(5), same seed 20261126), but reading the
generated distances from run_l3_gbn2.py outputs instead of E03's own MD.

E03's definitions:
  S = real 0-1 ns series (11 frames, 100 ps spacing); real_mean = S.mean(1).
  real_h6 = [S[:,0], S.mean(1), S.std(1), S.min(1), S[:,-1]-S[:,0], (S<=t).mean(1)].
  G (gen 7 stats) = [mean, std, min, max, end-start, frac<=t, frac<=t+1] over the
  post-origin snapshots of the condition; end-start = mean over seeds of the
  final snapshot minus the frame-10 distance.
  Residual correlation = Pearson/Spearman between (gen_mean - static prediction
  of gen_mean) and (real_mean - static prediction of real_mean).
  Shuffle control = G rows permuted WITHIN each protein.
  Downstream = pooled AUPRC for static20 / real_h6 / gen_mean / gen_all7 /
  shuffle_all7; recovery = gen gain / real_h6 gain.

Usage:
  analyze_l3.py --condition 4x100 --out l3_00_reproduction.json
  analyze_l3.py --all-conditions --out l3_01_hb_form.json
Conditions: 1x50, 1x100, 1x200, 1x400, 2x100, 4x100, 8x100.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, r2_score
from sklearn.model_selection import GroupKFold

WORK = Path('/home/hatch/workspace/kinescope/l3work')
EXTRACTED = WORK / 'extracted'
GPU_OUT = WORK / 'gpu_out'
PROTEINS = ['1a62_A', '1dd3_B', '1ikt_A', '1m8s_A', '1sf9_A', '1uha_A', '1xmk_A',
            '2e5y_B', '2gpi_A', '2i9x_A', '2nwv_A', '2ysk_A', '3c9p_A', '3fpn_A',
            '3kz7_A', '3pes_A', '3t92_A', '3v3l_A', '4cv4_A', '4myy_A', '4zds_B',
            '5ii7_A', '6bk4_A', '6in7_A', '7ead_A']
HORIZONS = [1, 5, 10, 30, 50, 90]  # ns; y columns are frame offsets 10/50/100/300/500/900
SEED = 20261126
REG_KW = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
              min_samples_leaf=25, l2_regularization=0.2, random_state=SEED)
CLF_KW = dict(max_iter=160, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=25, l2_regularization=0.15, random_state=SEED)
THRESH = {'hb_form': 3.5, 'hb_diss': 3.5, 'ct_form': 4.5, 'ct_diss': 4.5,
          'sb_form': 4.0, 'sb_diss': 4.0}
CONDITIONS = ['1x50', '1x100', '1x200', '1x400', '2x100', '4x100', '8x100']


def condition_samples(batch, long, condition):
    """(samples (M, N) post-origin distances, last (R, N) final snapshots)."""
    if condition == '1x50':
        return batch[0, 1:6], batch[0, 5:6]
    if condition == '1x100':
        return batch[0, 1:11], batch[0, 10:11]
    if condition == '1x200':
        return long[0, 1:21], long[0, 20:21]
    if condition == '1x400':
        return long[0, 1:41], long[0, 40:41]
    if condition == '2x100':
        return batch[0:2, 1:11].reshape(20, -1), batch[0:2, 10]
    if condition == '4x100':
        return batch[0:4, 1:11].reshape(40, -1), batch[0:4, 10]
    if condition == '8x100':
        return batch[0:8, 1:11].reshape(80, -1), batch[0:8, 10]
    raise ValueError(condition)


def stats7(samples, last, d0_ref, thresh):
    """E03's G: [mean, std, min, max, end-start, frac<=t, frac<=t+1]."""
    return np.column_stack([
        samples.mean(axis=0), samples.std(axis=0), samples.min(axis=0),
        samples.max(axis=0), last.mean(axis=0) - d0_ref,
        (samples <= thresh).mean(axis=0), (samples <= thresh + 1.0).mean(axis=0),
    ]).astype(np.float32)


def real_h6(S, thresh):
    return np.column_stack([
        S[:, 0], S.mean(axis=1), S.std(axis=1), S.min(axis=1),
        S[:, -1] - S[:, 0], (S <= thresh).mean(axis=1),
    ]).astype(np.float32)


def load_mode_data(mode, condition):
    """Stack all 25 proteins for one phenomenon and one length condition."""
    Xs, Ys, Ss, Gs, Ps = [], [], [], [], []
    for gi, prot in enumerate(PROTEINS):
        z = np.load(EXTRACTED / f'{prot}.npz')
        S = z[f'{mode}_series'].astype(np.float64)   # (N, 11) frames 0..10
        y = z[f'{mode}_y']
        keep = np.isfinite(S).all(axis=1) & np.isfinite(y).all(axis=1)
        b = np.load(GPU_OUT / f'{prot}_batch.npz')
        batch = b[f'{mode}_dists'].astype(np.float64)
        long = None
        lf = GPU_OUT / f'{prot}_long.npz'
        if lf.exists():
            long = np.load(lf)[f'{mode}_dists'].astype(np.float64)
        if condition in ('1x200', '1x400') and long is None:
            raise FileNotFoundError(f'{prot}_long.npz missing for {condition}')
        samples, last = condition_samples(batch, long, condition)
        gen = stats7(samples, last, S[:, 10], THRESH[mode])
        X = z[f'{mode}_X']
        if mode.startswith('hb'):
            X = X[:, :20]
        Xs.append(X[keep]); Ys.append(y[keep]); Ss.append(S[keep])
        Gs.append(gen[keep]); Ps.append(np.full(int(keep.sum()), prot, dtype=object))
    return (np.vstack(Xs).astype(np.float64), np.vstack(Ys),
            np.vstack(Ss), np.vstack(Gs).astype(np.float64), np.concatenate(Ps))


def oof_reg(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    oof = np.empty(len(y), dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(**REG_KW)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


def oof_clf(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    oof = np.empty(len(y), dtype=np.float64)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingClassifier(**CLF_KW)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def analyze(mode, condition, with_downstream=True):
    X, Y, S, G, P = load_mode_data(mode, condition)
    real_mean = S.mean(axis=1)
    gen_mean = G[:, 0]
    real_h = real_h6(S, THRESH[mode])

    pred_real = oof_reg(X, real_mean, P)
    pred_gen = oof_reg(X, gen_mean, P)
    real_resid = real_mean - pred_real
    gen_resid = gen_mean - pred_gen
    r_p, _ = pearsonr(gen_resid, real_resid)
    r_s, _ = spearmanr(gen_resid, real_resid)

    pred_gmean = oof_reg(np.column_stack([X, gen_mean[:, None]]), real_mean, P)
    pred_gall = oof_reg(np.column_stack([X, G]), real_mean, P)
    rng = np.random.default_rng(SEED)
    G_sh = G.copy()
    for pr in np.unique(P):
        idx = np.where(P == pr)[0]
        G_sh[idx] = G_sh[idx][rng.permutation(len(idx))]
    pred_gshuf = oof_reg(np.column_stack([X, G_sh]), real_mean, P)

    out = {'mode': mode, 'condition': condition, 'n': int(len(real_mean)),
           'n_proteins': int(len(np.unique(P))),
           'static20_resid_sd': float(real_resid.std()),
           'r2_static_to_real_mean': float(r2_score(real_mean, pred_real)),
           'r2_static_to_gen_mean': float(r2_score(gen_mean, pred_gen)),
           'residual_pearson_r': float(r_p),
           'residual_spearman_r': float(r_s),
           'static20_plus_gen_mean_resid_sd': float((real_mean - pred_gmean).std()),
           'static20_plus_gen_all7_resid_sd': float((real_mean - pred_gall).std()),
           'static20_plus_gen_shuffle_resid_sd': float((real_mean - pred_gshuf).std()),
           'r2_real_mean_with_gen_all7': float(r2_score(real_mean, pred_gall))}
    print(f"[{mode} {condition}] N={out['n']} static20 resid SD={out['static20_resid_sd']:.4f} "
          f"R2={out['r2_static_to_real_mean']:.4f} | resid r Pearson={r_p:.4f} Spearman={r_s:.4f} | "
          f"+gen_mean SD={out['static20_plus_gen_mean_resid_sd']:.4f} "
          f"+gen_all7 SD={out['static20_plus_gen_all7_resid_sd']:.4f} "
          f"shuffle SD={out['static20_plus_gen_shuffle_resid_sd']:.4f}", flush=True)
    if with_downstream:
        downstream = {}
        for hi, h_ns in enumerate(HORIZONS):
            y = Y[:, hi].astype(int)
            p_s = oof_clf(X, y, P)
            p_real = oof_clf(np.column_stack([X, real_h]), y, P)
            p_gm = oof_clf(np.column_stack([X, gen_mean[:, None]]), y, P)
            p_ga = oof_clf(np.column_stack([X, G]), y, P)
            p_sh = oof_clf(np.column_stack([X, G_sh]), y, P)
            ap_s = float(average_precision_score(y, p_s))
            ap_r = float(average_precision_score(y, p_real))
            ap_gm = float(average_precision_score(y, p_gm))
            ap_ga = float(average_precision_score(y, p_ga))
            ap_sh = float(average_precision_score(y, p_sh))
            g_real = ap_r - ap_s
            g_gm = ap_gm - ap_s
            g_ga = ap_ga - ap_s
            downstream[str(h_ns)] = {
                'ap_static20': ap_s, 'ap_real_1ns_6stats': ap_r,
                'ap_gen_mean': ap_gm, 'ap_gen_all7': ap_ga, 'ap_gen_shuffle': ap_sh,
                'gain_real': float(g_real), 'gain_gen_mean': float(g_gm),
                'gain_gen_all7': float(g_ga),
                'recovery_gen_mean': float(g_gm / g_real) if g_real > 1e-6 else None,
                'recovery_gen_all7': float(g_ga / g_real) if g_real > 1e-6 else None,
                'corr_real_resid_vs_label': float(pearsonr(real_resid, y)[0]),
                'corr_gen_resid_vs_label': float(pearsonr(gen_resid, y)[0]),
            }
            print(f"  H={h_ns:2d}ns | static={ap_s:.4f} real1ns={ap_r:.4f} (+{g_real:.4f}) | "
                  f"gen_mean={ap_gm:.4f} (+{g_gm:.4f}, rec={g_gm/g_real:.3f}) | "
                  f"gen_all7={ap_ga:.4f} (+{g_ga:.4f}, rec={g_ga/g_real:.3f}) | shuffle={ap_sh:.4f}",
                  flush=True)
        out['downstream'] = downstream
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='hb_form')
    ap.add_argument('--condition', default=None)
    ap.add_argument('--all-conditions', action='store_true')
    ap.add_argument('--no-downstream', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    conds = CONDITIONS if args.all_conditions else [args.condition or '4x100']
    results = {}
    for c in conds:
        results[c] = analyze(args.mode, c, with_downstream=not args.no_downstream)
    Path(args.out).write_text(json.dumps(results, indent=1), encoding='utf-8')
    print('wrote', args.out, flush=True)


if __name__ == '__main__':
    main()
