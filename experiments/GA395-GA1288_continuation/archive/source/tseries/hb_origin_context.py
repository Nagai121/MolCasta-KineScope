"""Candidate and frame-level context derived from one static prior."""
import json

import numpy as np
from scipy.stats import rankdata

from .data import TRAIN, VALIDATION
from .hb_single_structure import ADDITIONAL_TRAINING, OUT, explore, load
from .search_events import EXTRA_TRAINING


def context(part):
    ends=part["ends"]
    blocks=[]
    for start,end in zip(np.r_[0,ends[:-1]],ends):
        pair=part["pair"][start:end]
        if not len(pair):
            continue
        score=part["old"][start:end]
        if np.any(score<0):
            raise ValueError("Frame context requires a static prior for all candidates")
        a,b=pair[:,:2].astype(np.intp).T
        donors=np.bincount(a,weights=score)[a]
        acceptors=np.bincount(b,weights=score)[b]
        mass=float(score.sum(dtype=np.float64))
        average=mass/len(score)
        close=float(np.mean(pair[:,2]<=4.0))
        blocks.append(np.column_stack((
            rankdata(score,method="average")/len(score),
            score/(average+.01), score/(donors+.01),score/(acceptors+.01),
            np.full(len(score),average),np.full(len(score),close),
            np.log1p(np.maximum(donors-score,0)),np.log1p(np.maximum(acceptors-score,0)),
        )))
    return np.concatenate(blocks).astype(np.float32) if blocks else np.empty((0,8),dtype=np.float32)


def main():
    panel=json.loads((OUT/"diverse_training_panel.json").read_text(encoding="utf-8"))["proteins"]
    train=load(TRAIN,25,geometry=True)+load(EXTRA_TRAINING+ADDITIONAL_TRAINING,50,geometry=True)
    train+=load(panel,100,geometry=True)
    valid=load(VALIDATION,100,geometry=True)
    configs=(("xgb",118,"hb-origin-context",False,6,1.0),
             ("xgb",166,"hb-origin-context",True,5,1.0),
             ("mlp",118,"hb-origin-context",True,96,1.0))
    explore("origin_context",configs,1311,train,valid)


if __name__=="__main__":
    main()
