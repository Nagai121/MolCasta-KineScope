"""Record panel metadata and compare sequences across panel boundaries."""
import argparse
import csv
import itertools
import json
from pathlib import Path

from .data import PANELS


def similarity(a, b):
    """Global affine alignment, scores match +2, mismatch -1, open -10, extend -1."""
    impossible = (-10**6, 0, 0)
    previous = [(0, 0, 0)] + [(-9 - j, 0, 0) for j in range(1, len(b) + 1)]
    vertical = [impossible] * (len(b) + 1)
    for i, left in enumerate(a, 1):
        row = [(-9 - i, 0, 0)]
        horizontal = impossible
        for j, right in enumerate(b, 1):
            score, matches, pairs = previous[j - 1]
            same = left == right
            diagonal = (score + (2 if same else -1), matches + same, pairs + 1)
            op = previous[j]
            prior = vertical[j]
            vertical[j] = max((op[0] - 10, *op[1:]), (prior[0] - 1, *prior[1:]))
            op = row[-1]
            horizontal = max((op[0] - 10, *op[1:]), (horizontal[0] - 1, *horizontal[1:]))
            row.append(max(diagonal, vertical[j], horizontal))
        previous = row
    score, matches, pairs = previous[-1]
    return dict(score=score, identity_over_shorter=matches / min(len(a), len(b)),
                paired_identity=matches / pairs if pairs else 0,
                shorter_coverage=pairs / min(len(a), len(b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.catalog.open(encoding="utf-8") as source:
        rows = {r["PDB"]: r for r in csv.DictReader(source, delimiter="\t")}
    ids = set().union(*PANELS.values())
    if len({rows[k]["UniProt"] for k in ids}) != len(ids):
        raise ValueError("Expected unique UniProt accessions across the panels")
    keys = ("length", "UniProt", "protein_name", "alpha%", "beta%", "avg_RMSF",
            "CATH_class", "CATH_architecture", "non_redundant_protein")
    output = {"metadata": {p: {k: rows[p][k] for k in keys} for p in sorted(ids)},
              "cross_split": []}
    for (an, aa), (bn, bb) in itertools.combinations(PANELS.items(), 2):
        for a, b in itertools.product(aa, bb):
            comparison = similarity(rows[a]["sequence"], rows[b]["sequence"])
            output["cross_split"].append(dict(a=a, b=b, split_a=an, split_b=bn, **comparison))
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
