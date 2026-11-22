"""Walk T1I1451/T1I1453 calibration manifests recursively and list every
checkpoint file the frozen T1I2006 prediction path needs."""
import json
from pathlib import Path
OUT = Path("/tmp/doca_run/runs/hb_single_structure")
needed = {}   # relpath -> sha256 or None
manifests = []

def walk(folder: Path, record: dict, depth=0):
    fit = record.get("fit", {})
    ckpt = fit.get("checkpoint")
    cfg = record.get("config", [None])[0]
    if ckpt:
        rel = folder / ckpt
        needed[str(rel.relative_to(OUT))] = fit.get("sha256")
        if ckpt.endswith(".json"):
            manifests.append(rel)
    if cfg == "hb-calibration":
        # the checkpoint json itself is the manifest
        pass
    return

def walk_manifest(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print("UNREADABLE", path, e); return
    for comp in data.get("components", []):
        folder = OUT / comp["folder"]
        rec = comp["record"]
        walk(folder, rec)
        ck = rec.get("fit", {}).get("checkpoint", "")
        if ck.endswith(".json"):
            walk_manifest(folder / ck)

for gate in ("T1I1451", "T1I1453"):
    walk_manifest(OUT / "ensemble_gate" / f"{gate}.json")
print(f"{len(needed)} checkpoint files needed:")
total = 0
for rel, sha in sorted(needed.items()):
    print(rel, (sha or "")[:12])
json_needed = [r for r in needed if r.endswith(".json")]
print("json manifests among them:", len(json_needed))
Path("/tmp/needed_files.json").write_text(json.dumps(needed, indent=1))
