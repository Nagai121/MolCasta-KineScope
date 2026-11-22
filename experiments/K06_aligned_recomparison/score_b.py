"""K06(b) per-protein scoring: frozen T1I2006 + DocA sidecar features +
proposal-22 generated features, on the original DocA protein lists.
Checkpointed: skips proteins whose output npz exists; deletes the
converted trajectory after scoring to bound disk."""
import sys, json, shutil, subprocess, time
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg_doca")
from pathlib import Path
import numpy as np
from tseries.data import registry, trajectory, topology, cell, acquire
from tseries.t1i2006_initial_history_100 import prepare_model, predict_snapshot, HORIZONS, TARGETS
from tseries.features import _mic

ROOT = Path("atlas-workspace")
OUTD = Path("k06b_out"); OUTD.mkdir(exist_ok=True)
MODEL = None

def gen6(series5):
    return np.column_stack([series5[0], series5.mean(0), series5.std(0), series5.min(0),
                            series5[-1] - series5[0], (series5 <= 3.5).mean(0)]).astype(np.float32)

def one_replica(protein, replica):
    reg = registry(ROOT, protein)
    tid = trajectory(protein, replica)
    blocks = list(reg.blocks(tid, start=0, stop=911, chunk_frames=128))
    pos = np.concatenate([b.positions for b in blocks], axis=0)
    boxes = np.concatenate([b.box for b in blocks], axis=0)
    topo = topology(reg.atoms(tid).to_pydict())
    pair, base, frozen, counts = predict_snapshot(pos[10], cell(boxes[10]), topo, *MODEL)
    a, b = pair[:, :2].astype(np.intp).T
    series = np.stack([np.linalg.norm(_mic(pos[k, b] - pos[k, a], cell(boxes[k])), axis=1)
                       for k in range(11)], axis=0)
    history = np.column_stack((series[0], series.mean(axis=0), series.std(axis=0), series.min(axis=0),
                               series.max(axis=0), series[-1] - series[0], series[-1] - series[-2],
                               (series <= 3.5).mean(axis=0), (series <= 4.5).mean(axis=0),
                               (np.diff(series, axis=0) > 0).mean(axis=0))).astype(np.float32)
    static = np.column_stack((frozen, pair[:, 2], base[:, 1], base[:, 3], base[:, 5])).astype(np.float32)
    label = np.column_stack([np.linalg.norm(_mic(pos[TARGETS[h], b] - pos[TARGETS[h], a],
                                                 cell(boxes[TARGETS[h]])), axis=1) <= 3.5
                             for h in HORIZONS]).astype(np.uint8)
    gen_npz = Path(f"/tmp/gen_{protein}_{replica}.npz")
    subprocess.run(["/home/hatch/workspace/mt-venv/bin/python", "/home/hatch/workspace/k06b/gen_sub.py",
                    str(ROOT), protein, str(replica), str(gen_npz)],
                   check=True, capture_output=True, text=True)
    with np.load(gen_npz) as z:
        gpair, gdist, g8 = z["pair"], z["gen_dist"], z["gen8"]
    key = {(int(p[0]), int(p[1])): i for i, p in enumerate(gpair)}
    idx = np.array([key[(int(u), int(v))] for u, v in zip(a, b)])
    assert len(idx) == len(pair), (protein, replica, len(idx), len(pair))
    gen5 = np.column_stack([pair[:, 2], gdist[idx]]).T
    return dict(pair=pair, static=static, history=history, frozen=frozen, label=label,
                gen8=g8[idx].astype(np.float32), gen6=gen6(gen5))

def main():
    global MODEL
    proteins = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    MODEL = prepare_model()
    for p in proteins:
        out = OUTD / f"{p}.npz"
        if out.is_file():
            print("SKIP", p, flush=True); continue
        t0 = time.time()
        try:
            acquire(ROOT, p)
            reps = [one_replica(p, r) for r in (1, 2, 3)]
            np.savez_compressed(out, **{f"r{r}_{k}": v for r, rep in enumerate(reps, 1)
                                        for k, v in rep.items()})
            print("SCORED", p, "candidates", sum(len(r["label"]) for r in reps),
                  "sec", round(time.time() - t0, 1), flush=True)
        except Exception as e:
            print("FAILED", p, type(e).__name__, str(e)[:200], flush=True)
        finally:
            shutil.rmtree(ROOT / p, ignore_errors=True)

if __name__ == "__main__":
    main()
