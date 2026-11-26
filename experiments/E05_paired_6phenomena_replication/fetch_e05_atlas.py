"""Fetch all 49 strictly unseen MDGen atlas_test.csv proteins with length <= 250 residues into shared_data/atlas."""
import concurrent.futures
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, r"d:\KineScope\experiments\E01_true_short_md_guidance")
import fetch_atlas

MDGEN_SPLIT = Path(r"d:\KineScope\experiments\E04_public_generator_static_redundancy\mdgen_repo\splits\atlas_test.csv")


def select_e05_proteins(max_len: int = 250):
    rows = list(csv.DictReader(MDGEN_SPLIT.open(encoding="utf-8")))
    cand = sorted([(r["name"], len(r["seqres"])) for r in rows if len(r["seqres"]) <= max_len], key=lambda x: (x[1], x[0]))
    return cand


def main():
    chosen = select_e05_proteins(250)
    print(f"Selected {len(chosen)} strictly unseen MDGen atlas_test proteins (len {chosen[0][1]}..{chosen[-1][1]})", flush=True)
    t0 = time.time()
    ok = cached = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(fetch_atlas.fetch_one, p) for p, _ in chosen]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            prot, status, dt = fut.result()
            if status == "cached":
                cached += 1
            elif status == "ok":
                ok += 1
            else:
                fail += 1
            print(f"[{i}/{len(chosen)}] {prot}: {status} ({dt:.1f}s)", flush=True)
    print(f"E05 ATLAS fetch done in {time.time() - t0:.1f}s | ok={ok} cached={cached} fail={fail}", flush=True)
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
