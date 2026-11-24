"""Rebuilt for GAm01 (Dosukoi side): JSON record writer used across the GA package.

The original explore module was never committed. record() is reconstructed
from its call sites: record(path, payload) writes the payload as JSON.
The committed GA result files use 2-space indentation, which is matched here.
score() exists in the original (imported by ga_native240 / ga_openmm_pilot)
but is never called on the GA351 blind scoring path; it raises if called.
"""
import json
from pathlib import Path


def record(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def score(*args, **kwargs):
    raise NotImplementedError(
        "explore.score is not part of the GAm01 blind scoring path")
