"""GA588+: pretrain a static encoder on extra proteins, then add generated dynamics."""
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from tseries.data import TEST
from .ga_force import build
from .ga_native_offset import HOLD
from .ga_sequential50 import pool
from .ga_static_offset import VALID, estimates

R = Path("runs")
OUT = R / "ga588-transfer"


def contextual_response(values, origins, proteins):
    from .ga_replica_teacher import trajectory_rows

    result = np.empty((len(values), 22), dtype=np.float32)
    for name in np.unique(proteins):
        pairs, times = trajectory_rows(name, 1)
        for origin in np.unique(origins[proteins == name]):
            mask = (proteins == name) & (origins == origin)
            ids = pairs[times == origin, :2].astype(np.intp)
            assert len(ids) == mask.sum()
            source = values[mask]
            steps = (source.shape[1] - 4) // 3
            signal = np.c_[source[:, :steps].mean(1), source[:, steps:2*steps].mean(1), source[:, -4:]]
            center = np.median(signal, axis=0)
            delta = signal - center
            mad = np.median(np.abs(delta), axis=0).clip(.04)
            endpoints = []
            for field in (0, 1):
                counts = np.bincount(ids.ravel()).clip(1)
                mass = np.bincount(ids.ravel(), np.repeat(signal[:, field], 2), minlength=len(counts))
                endpoints.extend((mass[ids[:, 0]]/counts[ids[:, 0]], mass[ids[:, 1]]/counts[ids[:, 1]]))
            result[mask] = np.c_[delta, delta/mad, np.broadcast_to(center, signal.shape), *endpoints]
    return np.c_[values, result]


def main(expanded=False, joint=False, contextual=False):
    out = R / "ga642-joint-transfer" if joint else (R / "ga616-transfer-expanded" if expanded else OUT)
    if contextual:
        out = R / "ga659-context-transfer"
    torch.set_num_threads(3)
    out.mkdir(exist_ok=True)
    data, validation = pool()
    x, d, g, y, origin, train = data
    v, e, h, t, vo, vg, base, parent = validation
    if contextual:
        d = contextual_response(d, origin, train)
        e = contextual_response(e, vo, vg)
    panels = set(VALID + TEST + HOLD)
    panels.update(json.loads((R/"ga321/reserve.json").read_bytes())["proteins"])
    panels.update(row["id"] for row in json.loads((R/"ga351/reserved.json").read_bytes())["proteins"])
    assert not set(train) & panels
    candidates = json.loads((R/"ga330/development.json").read_bytes())["train"]
    extra = [n for n in candidates if n not in panels and n not in set(train)
             and (R/"ga_force"/f"{n}-R1-4.npz").exists()]
    if not extra:
        raise RuntimeError("No additional cached training proteins")
    replicas = (1, 2, 3) if expanded else (1,)
    examples = [build((n, replica, 4)) for n in extra for replica in replicas
                if (R/"ga_force"/f"{n}-R{replica}-4.npz").exists()]
    static = np.concatenate([row[0] for row in examples])
    targets = np.concatenate([row[1] for row in examples])
    mu, scale = static.mean(0), static.std(0).clip(.001)
    dm, ds = d.mean(0), d.std(0).clip(.001)
    def tensor(values, center, spread, device):
        return torch.tensor(((values-center)/spread).clip(-10,10),dtype=torch.float32,device=device)
    s = tensor(static,mu,scale,"cuda")
    sy = torch.tensor(targets-static[:,0],dtype=torch.float32,device="cuda")
    a = tensor(x,mu,scale,"cuda")
    b = tensor(v,mu,scale,"cpu")
    da = tensor(d,dm,ds,"cuda")
    db = tensor(e,dm,ds,"cpu")
    yy = torch.tensor(y-x[:,0],dtype=torch.float32,device="cuda")
    wt = torch.tensor(np.where(origin==49,1.,4.),dtype=torch.float32,device="cuda")
    torch.manual_seed(588)
    encoder = nn.Sequential(nn.Linear(x.shape[1],128),nn.LayerNorm(128),nn.SiLU(),
                            nn.Linear(128,48),nn.LayerNorm(48),nn.SiLU()).cuda()
    static_head = nn.Linear(48,1).cuda()
    opt = torch.optim.AdamW([*encoder.parameters(),*static_head.parameters()],lr=.002)
    for epoch in range(12):
        for ids in torch.randperm(len(s),device="cuda").split(2048):
            loss = nn.functional.mse_loss(static_head(encoder(s[ids])).flatten(),sy[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch in (3,7,11):
            print("pretraining",epoch+1,len(extra),len(s),flush=True)
    enc = deepcopy(encoder.state_dict())
    static_state = deepcopy(static_head.state_dict())
    masks = {"all":vo>=0,"first":vo==49,"later":vo!=49}
    specs = [(initial, freeze, head_width, blend) for initial,freeze in
             (("pretrained",True),("pretrained",False),("random",False))
             for head_width in (32,96) for blend in (.35,.65)]
    if expanded:
        specs = [("pretrained", False, width, .35) for width in (48,72,96,144,192)]
    if joint:
        specs = [("pretrained", False, width, .35) for width in (48,72,96,144)]
    if contextual:
        specs = [("pretrained", False, width, .35) for width in (48,72,96,144,192)]
    stats = []
    for index,(initial,freeze,width,w) in enumerate(specs):
        torch.manual_seed(589)
        if initial == "pretrained":
            encoder.load_state_dict(enc)
        else:
            for layer in encoder:
                if hasattr(layer,"reset_parameters"):
                    layer.reset_parameters()
        encoder.requires_grad_(not freeze)
        head = nn.Sequential(nn.Linear(48+d.shape[1],width),nn.LayerNorm(width),
                             nn.SiLU(),nn.Linear(width,1)).cuda()
        static_head.load_state_dict(static_state)
        opt = torch.optim.AdamW([*filter(lambda p:p.requires_grad,encoder.parameters()),
                                 *head.parameters(), *static_head.parameters()],lr=.0015)
        start=perf_counter()
        for _ in range(16):
            for ids in torch.randperm(len(a),device="cuda").split(1536):
                z = head(torch.cat([encoder(a[ids]),da[ids]],1)).flatten()
                loss = ((z-yy[ids]).square()*wt[ids]).mean()
                if joint:
                    extra_ids = torch.randint(len(s), (len(ids),),device="cuda")
                    extra_error = nn.functional.mse_loss(static_head(encoder(s[extra_ids])).flatten(),sy[extra_ids])
                    loss = loss + (.25 if width <= 72 else .5) * extra_error
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        torch.cuda.synchronize()
        cost=perf_counter()-start
        ce,ch = deepcopy(encoder).cpu().eval(),head.cpu().eval()
        with torch.inference_mode():
            direct = ch(torch.cat([ce(b),db],1)).numpy().ravel()+v[:,0]
        p=(1-w)*parent+w*direct
        results={k:estimates(p[m],t[m],base[m]) for k,m in masks.items()}
        tag=f"GA{(659 if contextual else (642 if joint else (616 if expanded else 588)))+index}"
        torch.save(dict(encoder=ce.state_dict(),head=ch.state_dict(),mu=mu,scale=scale,
                        dynamic_mean=dm,dynamic_scale=ds,width=width,blend=w),out/f"{tag}.pt")
        np.savez_compressed(out/f"{tag}.npz",prediction=p,readout=direct)
        row=dict(id=tag,initial=initial,freeze=freeze,head_width=width,blend=w,metrics=results,fit_seconds=cost)
        stats.append(row)
        with (out/"trials.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n")
        print(tag,initial,freeze,width,w,results["all"]["residual_correlation"],flush=True)
        if results["all"]["residual_correlation"] >= .700:
            break
    best=max(stats,key=lambda row:row["metrics"]["all"]["residual_correlation"])
    (out/"summary.json").write_text(json.dumps(dict(best=best,pretrain=extra,pretrain_pairs=len(targets)),indent=2),encoding="utf-8")


if __name__=="__main__":
    import sys
    main(expanded="--expanded" in sys.argv, joint="--joint" in sys.argv, contextual="--context" in sys.argv)



