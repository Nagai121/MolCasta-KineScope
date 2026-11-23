"""K63 preparation: extract K06(b)-style features for 1mix_A and 2qt4_A.

1mix_A and 2qt4_A are the weak base's test proteins; they are NOT among
the K06(b) 108 proteins, so no K06(b) features exist for them. This
script follows K06(b) score_b.py's one_replica exactly (same frames,
same frozen T1I2006 model, same history10 formula, same static columns)
but skips the proposal-22 generated features, which K63 does not need.

Run with cwd=~/workspace/k06b/run (prepare_model uses relative paths
"runs/..." from there, as score_b.py did). Downloads go into a K63
workspace (~/workspace/kinescope/work/k63_atlas) and the converted
trajectory is deleted after scoring, as score_b.py did. Output:
~/workspace/kinescope/work/k63_cache/<protein>.npz with keys
r{1,2,3}_pair/_static/_history/_frozen/_label, same layout as k06b_out.
Checkpointed: skips proteins whose output npz exists.
"""
import sys, shutil, time
sys.path.insert(0, "/home/hatch/workspace/k06b/pkg_doca")
from pathlib import Path
import numpy as np
from tseries.data import registry, trajectory, topology, cell, acquire
from tseries.t1i2006_initial_history_100 import prepare_model, predict_snapshot, HORIZONS, TARGETS
from tseries.features import _mic

ROOT = Path("/home/hatch/workspace/kinescope/work/k63_atlas")
OUTD = Path("/home/hatch/workspace/kinescope/work/k63_cache")
OUTD.mkdir(parents=True, exist_ok=True)

def one_replica(protein, replica):
    reg = registry(ROOT, protein)
    tid = trajectory(protein, replica)
    blocks = list(reg.blocks(tid, start=0, stop=911, chunk_frames=128))
    pos = np.concatenate([b.positions for b in blocks], axis=0)
    boxes = np.concatenate([b.box for b in blocks], axis=0)
    times = np.concatenate([b.time for b in blocks], axis=0)
    assert len(pos) == 911, (tid, len(pos))
    assert np.allclose(times[:11] - times[0], np.arange(11) * 100, rtol=0, atol=0.01), "0..1ns grid mismatch"
    assert np.allclose(times[[TARGETS[h] for h in HORIZONS]] - times[10],
                       np.array(HORIZONS) * 1000, rtol=0, atol=.01), "target grid mismatch"
    topo = topology(reg.atoms(tid).to_pydict())
    pair, base, frozen, counts = predict_snapshot(pos[10], cell(boxes[10]), topo, *MODEL)
    a, b = pair[:, :2].astype(np.intp).T
    series = np.stack([np.linalg.norm(_mic(pos[k, b] - pos[k, a], cell(boxes[k])), axis=1)
                       for k in range(11)], axis=0)
    np.testing.assert_allclose(series[-1], pair[:, 2], atol=2e-4, rtol=0)
    history = np.column_stack((series[0], series.mean(axis=0), series.std(axis=0), series.min(axis=0),
                               series.max(axis=0), series[-1] - series[0], series[-1] - series[-2],
                               (series <= 3.5).mean(axis=0), (series <= 4.5).mean(axis=0),
                               (np.diff(series, axis=0) > 0).mean(axis=0))).astype(np.float32)
    static = np.column_stack((frozen, pair[:, 2], base[:, 1], base[:, 3], base[:, 5])).astype(np.float32)
    label = np.column_stack([np.linalg.norm(_mic(pos[TARGETS[h], b] - pos[TARGETS[h], a],
                                                 cell(boxes[TARGETS[h]])), axis=1) <= 3.5
                             for h in HORIZONS]).astype(np.uint8)
    return dict(pair=pair, static=static, history=history, frozen=frozen, label=label)

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
            print("FAILED", p, type(e).__name__, str(e)[:300], flush=True)
        finally:
            shutil.rmtree(ROOT / p, ignore_errors=True)

if __name__ == "__main__":
    main()
