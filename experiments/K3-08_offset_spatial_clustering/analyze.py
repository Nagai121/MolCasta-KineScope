"""K3-08 analysis: is the K06(b) HB-formation residual (real 0-1 ns mean
distance minus the static-only GBM prediction, K39/K42 construction)
independent per pair, or do nearby pairs have similar residuals?

Within each test protein x replica group (100 x 3 = 300 groups,
163,074 candidates), candidate pairs are formed three ways:
  1. share_residue: the two candidates share at least one residue
     (donor or acceptor side).
  2. near_8A_no_share: midpoint distance at the origin frame (frame 10)
     <= 8 A (minimum image), sharing no residue.
  3. far_20A: midpoint distance >= 20 A (control).
For each category the Pearson correlation between the two residuals of
a pair is accumulated exactly (streaming sums over all pairs, no
sampling), for (a) raw residuals and (b) residuals demeaned within the
protein x replica group. Permutation control: 3 within-group shuffles
of the residuals, same pair sets.
Premise gate: the test-side GBM residual std must reproduce 0.6002 A.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

WORK = Path("/home/hatch/workspace/k3_08")
GEOM = WORK / "geom"
K06B_OUT = Path("/home/hatch/workspace/k06b/run/k06b_out")
SEED = 20261121
SHUFFLE_SEED = 20261126
N_SHUFFLES = 3


def mic(v, box):
    box = np.asarray(box, dtype=np.float64)
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-5):
        lengths = np.diag(box)
        return v - lengths * np.rint(v / lengths)
    inv = np.linalg.inv(box)
    fr = v @ inv
    fr -= np.rint(fr)
    return fr @ box


def load_side(proteins):
    S, M, groups = [], [], []
    for p in proteins:
        z = np.load(K06B_OUT / f"{p}.npz")
        for r in (1, 2, 3):
            S.append(z[f"r{r}_static"].astype(np.float64))
            M.append(z[f"r{r}_history"][:, 1].astype(np.float64))
            groups += [f"{p}|{r}"] * len(S[-1])
    return np.vstack(S), np.concatenate(M), np.array(groups)


class Acc:
    """Streaming Pearson sums for one category."""
    def __init__(self, n_variants):
        self.n = 0
        self.sx = np.zeros(n_variants)
        self.sy = np.zeros(n_variants)
        self.sxy = np.zeros(n_variants)
        self.sx2 = np.zeros(n_variants)
        self.sy2 = np.zeros(n_variants)

    def add(self, ii, jj, variants):
        if len(ii) == 0:
            return
        self.n += len(ii)
        for v, vals in enumerate(variants):
            x = vals[ii]
            y = vals[jj]
            self.sx[v] += x.sum()
            self.sy[v] += y.sum()
            self.sxy[v] += (x * y).sum()
            self.sx2[v] += (x * x).sum()
            self.sy2[v] += (y * y).sum()

    def pearson(self, v):
        n = self.n
        num = n * self.sxy[v] - self.sx[v] * self.sy[v]
        den = np.sqrt((n * self.sx2[v] - self.sx[v] ** 2) *
                      (n * self.sy2[v] - self.sy[v] ** 2))
        return float(num / den) if den > 0 else float("nan")


def main():
    orig = json.loads(Path("/home/hatch/workspace/k06b/results_100.json").read_text())
    CAL, TEST = orig["calibration_proteins"], orig["test_proteins"]
    S_cal, M_cal, _ = load_side(CAL)
    S_test, M_test, groups = load_side(TEST)
    gbm = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.065,
                                        max_leaf_nodes=15, min_samples_leaf=20,
                                        l2_regularization=0.15, random_state=SEED)
    gbm.fit(S_cal, M_cal)
    resid = M_test - gbm.predict(S_test)
    resid_std = float(resid.std())
    print(f"n_cal {len(M_cal)} n_test {len(M_test)} resid std {resid_std:.4f}", flush=True)
    assert abs(resid_std - 0.6002) < 0.001, f"premise failed: resid std {resid_std}"

    # variants: 0 raw, 1 demeaned, then per shuffle s: raw, demeaned
    n_var = 2 + 2 * N_SHUFFLES
    cats = {"share_residue": Acc(n_var), "near_8A_no_share": Acc(n_var), "far_20A": Acc(n_var)}
    share_two = 0          # cat-1 pairs sharing two residues
    far_sharing = 0        # cat-3 pairs that share a residue
    chunk_sharing_pairs = 0  # cross-check of cat-1 construction
    worst_verify = 0.0
    n_groups = 0
    spans = {}
    pos = 0
    for g in dict.fromkeys(groups.tolist()):
        spans[g] = (pos, pos + int((groups == g).sum()))
        pos = spans[g][1]

    for gi, (p, r) in enumerate([(g.split("|")[0], int(g.split("|")[1])) for g in
                                 dict.fromkeys(groups.tolist())]):
        lo, hi = spans[f"{p}|{r}"]
        res_g = resid[lo:hi]
        n = len(res_g)
        z = np.load(GEOM / f"{p}.npz")
        mid = z[f"r{r}_mid"].astype(np.float64)
        box = z[f"r{r}_box"].astype(np.float64)
        dres = z[f"r{r}_dres"].astype(np.int64)
        ares = z[f"r{r}_ares"].astype(np.int64)
        worst_verify = max(worst_verify, float(z[f"r{r}_verify_d10_maxdiff"][0]))
        assert len(mid) == n, (p, r, len(mid), n)
        n_groups += 1

        rd = res_g - res_g.mean()
        rng = np.random.default_rng(SHUFFLE_SEED + gi)
        variants = [res_g, rd]
        for _ in range(N_SHUFFLES):
            perm = rng.permutation(n)
            variants.append(res_g[perm])
            variants.append(rd[perm])

        # category 1: pairs sharing a residue (dict construction)
        by_res = {}
        for i in range(n):
            by_res.setdefault(int(dres[i]), []).append(i)
            by_res.setdefault(int(ares[i]), []).append(i)
        pair_set = set()
        for lst in by_res.values():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    i, j = lst[a], lst[b]
                    if i != j:
                        pair_set.add((min(i, j), max(i, j)))
        if pair_set:
            ii = np.fromiter((a for a, _ in pair_set), dtype=np.int64, count=len(pair_set))
            jj = np.fromiter((b for _, b in pair_set), dtype=np.int64, count=len(pair_set))
            cats["share_residue"].add(ii, jj, variants)
            two = (((dres[ii] == dres[jj]) & (ares[ii] == ares[jj])) |
                   ((dres[ii] == ares[jj]) & (ares[ii] == dres[jj])))
            share_two += int(two.sum())

        # categories 2 and 3: chunked upper triangle over midpoint distances
        for s in range(0, n, 512):
            e = min(s + 512, n)
            ii_c, jj_c = np.meshgrid(np.arange(s, e), np.arange(n), indexing="ij")
            mask_up = jj_c > ii_c
            if not mask_up.any():
                continue
            disp = mic(mid[jj_c] - mid[ii_c], box)
            dist = np.linalg.norm(disp, axis=-1)
            sharing = ((dres[ii_c] == dres[jj_c]) | (dres[ii_c] == ares[jj_c]) |
                       (ares[ii_c] == dres[jj_c]) | (ares[ii_c] == ares[jj_c]))
            chunk_sharing_pairs += int((sharing & mask_up).sum())
            for cat_name, sel in (("near_8A_no_share", mask_up & (dist <= 8.0) & ~sharing),
                                  ("far_20A", mask_up & (dist >= 20.0))):
                if sel.any():
                    ii_s = ii_c[sel]
                    jj_s = jj_c[sel]
                    cats[cat_name].add(ii_s, jj_s, variants)
                    if cat_name == "far_20A":
                        far_sharing += int(sharing[sel].sum())
        if gi % 50 == 0:
            print(f"group {gi} {p} R{r} done", flush=True)

    assert chunk_sharing_pairs == cats["share_residue"].n, (
        f"cat1 cross-check failed: chunk {chunk_sharing_pairs} vs dict {cats['share_residue'].n}")

    def cat_out(acc):
        sh_raw = [acc.pearson(2 + 2 * s) for s in range(N_SHUFFLES)]
        sh_dem = [acc.pearson(3 + 2 * s) for s in range(N_SHUFFLES)]
        return {
            "n_pairs": int(acc.n),
            "pearson_raw": acc.pearson(0),
            "pearson_demeaned": acc.pearson(1),
            "shuffled_raw": sh_raw,
            "shuffled_raw_max_abs": float(np.max(np.abs(sh_raw))),
            "shuffled_demeaned": sh_dem,
            "shuffled_demeaned_max_abs": float(np.max(np.abs(sh_dem))),
        }

    out = {
        "n_cal": int(len(M_cal)),
        "n_test": int(len(M_test)),
        "n_groups": n_groups,
        "premise": {
            "resid_std_gbm_test_A": resid_std,
            "expected_A": 0.6002,
            "geom_verify_worst_maxdiff_A": worst_verify,
            "geom_all_ok": bool(worst_verify < 0.1),
        },
        "pair_counts": {
            "share_residue": int(cats["share_residue"].n),
            "share_residue_sharing_two": int(share_two),
            "near_8A_no_share": int(cats["near_8A_no_share"].n),
            "far_20A": int(cats["far_20A"].n),
            "far_20A_sharing_residue": int(far_sharing),
        },
        "correlations": {name: cat_out(acc) for name, acc in cats.items()},
        "sampling": "none: exact streaming sums over all pairs in all 300 groups",
        "n_shuffles": N_SHUFFLES,
    }
    (WORK / "analysis_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.savez_compressed(WORK / "analysis_arrays.npz", residual_gbm=resid, groups=groups)
    print(json.dumps(out, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
