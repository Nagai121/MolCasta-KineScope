"""Compare review workload for the two frozen 1 ns hydrogen-bond models."""
import json
import math
from pathlib import Path

import numpy as np

from .compare_horizons import PROTEIN, STRIDE, hb_predictions, trial38
from .data import REPLICAS, cell, registry, topology, trajectory
from .features import _mic
from .models import load_cpu
from .search_events import count_to_recall


def summarize(labels, previous, current):
    keep = np.ones(len(labels), dtype=bool)
    old_count = count_to_recall(labels, previous, keep)
    new_count = count_to_recall(labels, current, keep)
    return dict(rows=len(labels), positives=int(labels.sum()),
                target_positives=math.ceil(.8 * labels.sum()),
                trial38_review=old_count, candidate_review=new_count,
                reduction=old_count / new_count if old_count and new_count else None)


def main():
    runs = Path("runs")
    old = trial38(runs / "legacy/trial38.pkl")
    meta, predict = load_cpu("T1I14", runs / "confirmation")
    reg = registry("atlas-workspace", PROTEIN)
    replicas = {}
    combined = []
    for replica in REPLICAS:
        tid = trajectory(PROTEIN, replica)
        topo = topology(reg.atoms(tid).to_pydict())
        blocks = list(reg.blocks(tid))
        positions = np.concatenate([b.positions for b in blocks])
        boxes = np.concatenate([b.box for b in blocks])
        times = np.concatenate([b.time for b in blocks])
        length = float(np.asarray(boxes[0]).reshape(-1)[0])
        rows = []
        for t in range(49, len(times) - 10, STRIDE):
            if not np.isclose(times[t+10] - times[t], 1000, rtol=0, atol=.01):
                raise ValueError("Unexpected 1 ns timestamp")
            ids, current, previous, omitted = hb_predictions(
                positions[t], cell(boxes[t]), length, topo, meta, predict, old)
            if omitted:
                raise ValueError(f"Unmatched trial38 candidates in {tid} at {t}")
            delta = positions[t+10, ids[:, 1]] - positions[t+10, ids[:, 0]]
            labels = np.linalg.norm(_mic(delta, cell(boxes[t+10])), axis=1) <= 3.5
            rows.append((labels, previous, current))
        flat = tuple(np.concatenate(col) for col in zip(*rows))
        combined.append(flat)
        replicas[str(replica)] = summarize(*flat)
        print(tid, replicas[str(replica)], flush=True)
    total = summarize(*(np.concatenate(col) for col in zip(*combined)))
    prior = json.loads((runs / "horizon_comparison.json").read_text())["results"]["hb"]["1"]["current"]
    assert (total["rows"], total["positives"]) == (prior["rows"], prior["positives"])
    report = dict(protein=PROTEIN, horizon_ns=1, origins_stride=STRIDE, target_recall=.8,
                  candidate="T1I14",
                  pooled=total, replicas=replicas)
    (runs / "hb_review_comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
