"""Generate a self-contained Colab script (v2, range fetch) for a batch
of proteins: embeds k3lib + fetch_range sources and the k06b pair data;
counts H-bonds at history frame 10 (total frame 100, origin frame);
prints one RESULT JSON line per protein (base64 counts + premises).
Usage: make_colab_batch_f10.py list_file out_script.py"""
import base64, io, json, sys
from pathlib import Path
import numpy as np

K06B = Path("/home/hatch/workspace/k06b/run/k06b_out")
WORK = Path("/home/hatch/workspace/k3_02")
proteins = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
buf = io.BytesIO()
payload = {}
for p in proteins:
    z = np.load(K06B / f"{p}.npz")
    for r in (1, 2, 3):
        payload[f"{p}|r{r}_pair"] = z[f"r{r}_pair"]
        payload[f"{p}|r{r}_hist0"] = z[f"r{r}_history"][:, 0]
np.savez_compressed(buf, **payload)
b64 = base64.b64encode(buf.getvalue()).decode()
k3lib = (WORK / "k3lib.py").read_text()
fetch = (WORK / "fetch_range.py").read_text()

HEADER = '''
import base64, io, json, sys, time
from pathlib import Path
print("batch start", flush=True)
try:
    import MDAnalysis  # noqa
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "MDAnalysis"], check=True)
    print("MDAnalysis installed", flush=True)
import numpy as np
import MDAnalysis as mda
'''
MAIN = '''
PROTEINS = __PROTEINS__
DATA = np.load(io.BytesIO(base64.b64decode("__B64__")))
WORKD = Path("/tmp/k3batch2"); WORKD.mkdir(exist_ok=True)
for protein in PROTEINS:
    t0 = time.time()
    try:
        results, premises = {}, {}
        last_err = None
        for target_mb in (48, 160, None):
            dest, got = fetch_protein(protein, WORKD, xtc_raw_target=(target_mb * 1024 * 1024) if target_mb else 1 << 62)
            try:
                topo = None
                for r in (1, 2, 3):
                    u = mda.Universe(str(dest / f"{protein}_prod_R{r}.tpr"), str(dest / f"{protein}_prod_R{r}.xtc"))
                    if topo is None: topo = build_protein_topology(u)
                    counts, prem = count_replica(u, topo, DATA[f"{protein}|r{r}_pair"], DATA[f"{protein}|r{r}_hist0"].astype(np.float64), count_frame=100)
                    assert prem["frame0_maxdiff"] < 0.1 and prem["frame100_maxdiff"] < 0.1 and prem["elem_a_ok"] and prem["elem_b_ok"], f"premise failed R{r}: {prem}"
                    results[f"r{r}_counts"] = counts; premises[f"r{r}"] = prem
                last_err = None
                break
            except FramesUnavailable as e:
                last_err = e
                continue
        if last_err is not None:
            raise RuntimeError(f"frames unavailable at all targets: {last_err}")
        buf = io.BytesIO(); np.savez_compressed(buf, **results)
        print("RESULT " + json.dumps({"protein": protein, "premises": premises,
                                      "counts_b64": base64.b64encode(buf.getvalue()).decode(),
                                      "elapsed": round(time.time() - t0, 1)}), flush=True)
    except Exception as e:
        print("FAILED " + json.dumps({"protein": protein, "error": f"{type(e).__name__}: {str(e)[:300]}"}), flush=True)
    import shutil; shutil.rmtree(WORKD / protein, ignore_errors=True)
print("batch done", flush=True)
'''
script = HEADER + "\n" + k3lib + "\n" + fetch + "\n" + MAIN.replace("__PROTEINS__", json.dumps(proteins)).replace("__B64__", b64)
Path(sys.argv[2]).write_text(script)
print("wrote", sys.argv[2], "proteins", len(proteins), "script KB", len(script) // 1024)
