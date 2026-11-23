"""GC01/GC02 diagnostics on frozen T1I2007 candidates."""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from tseries.hb_validation10 import panel10
from .explore import record
from .frontends import data, design, infer, DEST
from .frontdiagnostic import short_target


def matched(parts, base, signals, same_frame=False):
    """Uniform positive sampling per stratum, with 16 sampled negative matches."""
    y = np.concatenate([p["y"] for p in parts])
    d = np.concatenate([p["pair"][:, 2] for p in parts])
    names = list(dict.fromkeys(p["protein"] for p in parts))
    group = np.concatenate([np.full(len(p["y"]), names.index(p["protein"])) for p in parts])
    distance_bin = np.floor((d-3.5)/.1).astype(int).clip(0, 25)
    odds_bin = np.floor((design(d[:, None], base)[:, 0]+15)/.2).astype(int).clip(0, 150)
    key = group*26*151+distance_bin*151+odds_bin
    if same_frame:
        frame_ids = np.concatenate([(p["origin"].astype(np.int64) * 4 + p["replica"]) for p in parts])
        key = key.astype(np.int64) * (int(frame_ids.max())+1) + frame_ids
    order = np.argsort(key, kind="stable")
    bins = np.split(order, np.flatnonzero(np.diff(key[order]))+1)
    rng = np.random.default_rng(402)
    totals = np.zeros(len(names), dtype=int)
    wins = {name: np.zeros(len(names)) for name in signals}
    for b in bins:
        pos, neg = b[y[b]], b[~y[b]]
        if not len(pos) or not len(neg):
            continue
        opposite = rng.choice(neg, size=(len(pos), 16))
        g = group[b[0]]
        totals[g] += opposite.size
        for name, signal in signals.items():
            gap = signal[pos, None]-signal[opposite]
            wins[name][g] += (gap > 0).sum()+.5*(gap == 0).sum()
    boot = rng.integers(len(names), size=(2000, len(names)))
    results = {}
    denom = totals[boot].sum(axis=1)
    for name, w in wins.items():
        diff = w-wins["frozen_t1"]
        results[name] = dict(concordance=float(w.sum()/totals.sum()),
                             delta_vs_t1=float(diff.sum()/totals.sum()),
                             delta_ci95=np.quantile(diff[boot].sum(axis=1)/denom, [.025,.975]).tolist(),
                             by_protein={p: float(w[i]/totals[i]) for i,p in enumerate(names) if totals[i]})
    return dict(proteins=names, positive_rows=int(y.sum()), matched_positives=int(totals.sum()/16),
                matched_pairs=int(totals.sum()), distance_bin_angstrom=.1, logit_bin=.2,
                results=results)


def main():
    import torch
    torch.set_num_threads(3)
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/gc"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parts, x, base = data(panel10())
    scores = {"frozen_t1":base}
    for name, features in x.items():
        scores[name] = infer(design(features, base), DEST/f"{name}.pt")
    result = matched(parts, base, scores)
    record(args.output/"GC01.json", result)
    print("GC01", json.dumps(result), flush=True)
    with ThreadPoolExecutor(3) as pool:
        actual = np.concatenate(list(pool.map(short_target, parts)))
    # Lookahead diagnostic only. The prospective 100 ps frame is deliberately
    # withheld from static models and cannot be used for deployment scoring.
    result = matched(parts, base, {"frozen_t1":base, "observed_100ps":-actual})
    record(args.output/"GC02.json", result)
    print("GC02", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
