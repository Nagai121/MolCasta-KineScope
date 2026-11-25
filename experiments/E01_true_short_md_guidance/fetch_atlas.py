"""Download ATLAS analysis trajectories (.pdb + R1/R2/R3 .xtc) for the 100 unseen test proteins (+ 1mix_A for MD cost benchmark).

Source API: https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/{prot}
Extracts only {prot}.pdb and {prot}_prod_R{1,2,3}_fit.xtc (or matching .xtc files),
then deletes the zip archive to keep disk usage compact (~2-3 GB total).
"""
import concurrent.futures
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/{prot}"
PROT_LIST_FILE = Path(r"d:\KineScope\MolCasta-KineScope\experiments\K75_readout_transfer_across_horizons_dissociation\proteins108.txt")
OUT_ROOT = Path(r"d:\KineScope\experiments\shared_data\atlas")


def get_proteins():
    lines = [ln.strip() for ln in PROT_LIST_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # Lines 0..7 are the 8 calibration proteins; lines 8..107 are the 100 unseen test proteins.
    test100 = lines[8:108]
    assert len(test100) == 100, len(test100)
    # Also include 1mix_A (G04 reference protein) for direct GPU/CPU MD cost comparison.
    return test100, ["1mix_A"]


def is_done(pdir: Path) -> bool:
    if not (pdir / "done.json").exists():
        return False
    info = json.loads((pdir / "done.json").read_text(encoding="utf-8"))
    return all((pdir / fn).exists() for fn in info.get("files", [])) and len(info.get("xtc_files", [])) == 3


def fetch_one(prot: str, retries: int = 3):
    pdir = OUT_ROOT / prot
    if is_done(pdir):
        return (prot, "cached", 0.0)
    pdir.mkdir(parents=True, exist_ok=True)
    url = BASE_URL.format(prot=prot)
    t0 = time.time()
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "KineScope-E01/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read()
            saved_files = []
            xtc_files = []
            pdb_file = None
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for member in zf.namelist():
                    base = Path(member).name
                    if not base:
                        continue
                    low = base.lower()
                    # Keep the reference structure (.pdb) and the 3 production trajectories (.xtc)
                    if low.endswith(".pdb") and ("_r" not in low or low == f"{prot.lower()}.pdb"):
                        (pdir / base).write_bytes(zf.read(member))
                        saved_files.append(base)
                        if low == f"{prot.lower()}.pdb" or pdb_file is None:
                            pdb_file = base
                    elif low.endswith(".xtc"):
                        (pdir / base).write_bytes(zf.read(member))
                        saved_files.append(base)
                        xtc_files.append(base)
            # Fallback: if no pdb matched strict filter, take any .pdb in zip
            if pdb_file is None:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for member in zf.namelist():
                        base = Path(member).name
                        if base.lower().endswith(".pdb"):
                            (pdir / base).write_bytes(zf.read(member))
                            saved_files.append(base)
                            pdb_file = base
                            break
            xtc_files = sorted(xtc_files)
            assert pdb_file is not None, f"{prot}: no PDB found in zip"
            assert len(xtc_files) >= 3, f"{prot}: expected 3 xtc, got {xtc_files}"
            meta = {
                "protein": prot,
                "pdb": pdb_file,
                "xtc_files": xtc_files,
                "files": sorted(set(saved_files)),
                "zip_bytes": len(raw),
            }
            (pdir / "done.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return (prot, "ok", time.time() - t0)
        except Exception as e:
            last_err = str(e)
            time.sleep(2.0 * attempt)
    return (prot, f"FAIL: {last_err}", time.time() - t0)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    test100, extra = get_proteins()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(test100) + len(extra)
    targets = (test100 + extra)[:limit]
    t_start = time.time()
    ok = cached = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(fetch_one, p) for p in targets]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            prot, status, dt = fut.result()
            if status == "cached":
                cached += 1
            elif status == "ok":
                ok += 1
            else:
                fail += 1
            if status != "cached" or i % 20 == 0 or i == len(targets):
                print(f"[{i}/{len(targets)}] {prot}: {status} ({dt:.1f}s) | ok={ok} cached={cached} fail={fail}", flush=True)
    print(f"Done in {time.time() - t_start:.1f}s | ok={ok} cached={cached} fail={fail}", flush=True)
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
