"""Fetch E04 assets:
1. Official MDGen ATLAS checkpoint (`atlas.ckpt`) from HuggingFace `bjing-mit/mdgen`.
2. 25 strictly unseen MDGen test proteins (length <= 135 res from `mdgen_repo/splits/atlas_test.csv`)
   into `shared_data/atlas/` (reusing already-downloaded proteins).
"""
import concurrent.futures
import csv
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.path.insert(0, r"d:\KineScope\experiments\E01_true_short_md_guidance")
import fetch_atlas  # reuse fetch_one & OUT_ROOT

MDGEN_SPLIT = Path(r"d:\KineScope\experiments\E04_public_generator_static_redundancy\mdgen_repo\splits\atlas_test.csv")
CKPT_DIR = Path(r"d:\KineScope\experiments\E04_public_generator_static_redundancy\checkpoints")


def select_e04_proteins(n_target: int = 25):
    rows = list(csv.DictReader(MDGEN_SPLIT.open(encoding="utf-8")))
    # Strictly proteins in MDGen's official atlas_test.csv with length <= 140 residues (fits comfortably in 8 GB VRAM)
    cand = sorted([(r["name"], len(r["seqres"])) for r in rows if 45 <= len(r["seqres"]) <= 140], key=lambda x: x[1])
    # Prioritize ones already downloaded in shared_data/atlas, then shortest remaining up to n_target
    already = [c for c in cand if (fetch_atlas.OUT_ROOT / c[0] / "done.json").exists()]
    rest = [c for c in cand if not (fetch_atlas.OUT_ROOT / c[0] / "done.json").exists()]
    chosen = (already + rest)[:n_target]
    return chosen


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    # 1. Download official MDGen atlas.ckpt
    print("Downloading official MDGen atlas.ckpt from HuggingFace bjing-mit/mdgen...", flush=True)
    ckpt_path = hf_hub_download(
        repo_id="bjing-mit/mdgen",
        filename="atlas.ckpt",
        local_dir=str(CKPT_DIR),
    )
    print(f"Checkpoint ready: {ckpt_path} ({Path(ckpt_path).stat().st_size / 1e6:.1f} MB)", flush=True)

    # 2. Download any missing ATLAS trajectories for the 25 MDGen test proteins
    chosen = select_e04_proteins(25)
    print(f"Selected {len(chosen)} strictly unseen MDGen atlas_test proteins (len {chosen[0][1]}..{chosen[-1][1]}):", flush=True)
    print([c[0] for c in chosen], flush=True)

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
    print(f"E04 ATLAS fetch done in {time.time() - t0:.1f}s | ok={ok} cached={cached} fail={fail}", flush=True)
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
