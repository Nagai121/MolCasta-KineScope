"""Train static front-end corrections on conditionally matched candidate pairs."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tseries.hb_single_structure import OUT
from tseries.hb_validation10 import panel10
from .explore import record, score
from .frontends import data, design, infer, net
from .gc_assist import matched


def pairs(parts, matrix):
    rng = np.random.default_rng(211)
    cursor = 0
    positives, negatives = [], []
    for p in parts:
        n = len(p["y"])
        x = matrix[cursor:cursor+n]
        d = p["pair"][:, 2]
        distance = np.floor((d-3.5)/.15).astype(int).clip(0, 17)
        score_bin = np.floor((x[:, 0]+15)/.3).astype(int).clip(0, 100)
        key = (p["origin"]*18+distance)*101+score_bin
        idx = np.argsort(key, kind="stable")
        for g in np.split(idx, np.flatnonzero(np.diff(key[idx]))+1):
            pos, neg = g[p["y"][g]], g[~p["y"][g]]
            if len(pos) and len(neg):
                positives.append(np.repeat(pos+cursor, 4))
                negatives.append(rng.choice(neg+cursor, 4*len(pos)))
        cursor += n
    return np.concatenate(positives), np.concatenate(negatives)


def fit(x, y, positive, negative, weight, checkpoint):
    torch.manual_seed(218)
    center = torch.from_numpy(x.mean(0))
    scale = torch.from_numpy(np.maximum(x.std(0), .001))
    z = ((torch.from_numpy(x)-center)/scale).clamp(-10,10).cuda()
    offset = torch.from_numpy(x[:, 0].copy()).cuda()
    target = torch.from_numpy(y.astype(np.float32)).cuda()
    positive, negative = [torch.from_numpy(a).cuda() for a in (positive, negative)]
    model = net(z.shape[1]).cuda()
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    opt = torch.optim.AdamW(model.parameters(), lr=.002)
    for _ in range(5):
        for i in torch.randperm(len(z), device=z.device).split(4096):
            logits = offset[i]+.35*model(z[i]).flatten().tanh()
            loss = F.binary_cross_entropy_with_logits(logits, target[i])
            if weight:
                k = torch.randint(len(positive), (2048,), device=z.device)
                a, b = positive[k], negative[k]
                correction = .35*model(torch.cat((z[a],z[b]))).flatten().tanh()
                cp, cn = correction.chunk(2)
                loss = loss + weight*F.softplus(offset[b]-offset[a]+cn-cp).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    assert model[-1].weight.grad.is_cuda
    torch.save(dict(state=model.cpu().state_dict(),center=center,scale=scale),checkpoint)


def main():
    torch.set_num_threads(3)
    dest = Path("runs/g2i")
    dest.mkdir(parents=True, exist_ok=True)
    train_names = tuple(json.loads((OUT/"volume_training_panel.json").read_bytes())["proteins"])[:24]
    val_names = panel10()
    tr, a, bt = data(train_names)
    va, b, bv = data(val_names)
    y = np.concatenate([p["y"] for p in tr])
    positive, negative = pairs(tr, design(a["distance"],bt))
    scores = {"frozen_t1":bv}
    results = dict(train=list(train_names),validation=list(val_names),
                   match_count=len(positive),baseline=score(va,bv),trials={})
    for family in ("distance","existing_static","local_response","network_response","occupancy","combined"):
        for weight in (0.,.15,.5):
            name = f"{family}-{weight}"
            path = dest/f"{name}.pt"
            fit(design(a[family],bt),y,positive,negative,weight,path)
            p = infer(design(b[family],bv),path)
            scores[name] = p
            results["trials"][name] = dict(metrics=score(va,p),checkpoint=path.name)
            print(name,results["trials"][name]["metrics"]["auprc"],flush=True)
    results["conditional"] = matched(va,bv,scores,same_frame=True)
    record(dest/"conditional_rank.json",results)


if __name__=="__main__":
    main()
