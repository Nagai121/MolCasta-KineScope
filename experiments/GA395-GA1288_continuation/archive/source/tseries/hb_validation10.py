"""Lock and prepare a ten-protein panel before selecting with its labels."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from .data import HISTORICAL, TEST, TRAIN, VALIDATION, acquire
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, ROOT
from .hb_volume import choose
from .search_events import EXTRA_TRAINING, dump

PANEL = OUT / "validation10.json"
SOURCES = ("diverse_training_panel", "volume_training_panel", "volume_96_training_panel",
           "volume_128_training_panel", "ensemble_calibration_panel", "ensemble_gate_panel",
           "ap10_calibration_panel", "ap10_volume_144_training_panel", "ap10_volume_176_training_panel")


def reserved_proteins():
    existing = TRAIN + EXTRA_TRAINING + ADDITIONAL_TRAINING
    for source in SOURCES:
        path = OUT / f"{source}.json"
        if path.is_file():
            existing += tuple(json.loads(path.read_text(encoding="utf-8"))["proteins"])
    return existing


def panel10():
    existing = reserved_proteins()
    added = tuple(choose(existing, OUT / "validation10_added.json", per_class=(2, 1, 2, 1)))
    panel = VALIDATION + added
    if len(set(panel)) != 10 or set(panel) & set(existing + TEST + HISTORICAL):
        raise ValueError("Validation panel is not independent of other panels")
    if PANEL.is_file():
        saved = json.loads(PANEL.read_text(encoding="utf-8"))
        if tuple(saved["proteins"]) != panel:
            raise ValueError("Locked validation panel changed")
    else:
        dump(PANEL, dict(proteins=panel, added=added, main_metric="pooled AUPRC",
                         target=.480, logloss_reference="T1I2006", final_test=TEST))
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-calibration", action="store_true")
    args = parser.parse_args()
    validation = panel10()
    if args.prepare_calibration:
        panel = tuple(choose(reserved_proteins() + validation[4:],
                             OUT / "ap10_calibration_panel.json", per_class=(2, 1, 3, 2)))
        if set(panel) & set(validation + TEST + HISTORICAL):
            raise ValueError("Calibration overlaps evaluation")
        print("LOCKED_CALIBRATION", panel, flush=True)
    else:
        panel = validation
        print("LOCKED_VALIDATION", panel, flush=True)
    with ThreadPoolExecutor(3) as executor:
        for protein, _ in zip(panel, executor.map(lambda p: acquire(ROOT, p), panel)):
            print("READY", protein, flush=True)


if __name__ == "__main__":
    main()
