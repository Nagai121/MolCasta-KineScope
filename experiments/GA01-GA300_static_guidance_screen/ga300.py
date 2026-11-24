"""GA static-only history substitution: 300 generator/readout trials."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tseries.data import HISTORICAL, TEST
from tseries.hb_single_structure import OUT
from tseries.hb_validation10 import panel10
from .explore import record, score
from .frontends import data, design, fit as reference_fit, infer as reference_infer
from .gc_history import past, stats
from .g2i_path35 import build

DEST = Path("runs/ga300")
VIEWS = ("distance","existing_static","local_response","network_response","occupancy",
         "competitors","combined","straight","detour","anisotropy","availability","hybrid")
TARGETS = ("point","moments","hurdle","distribution","joint")
ARCHITECTURES = ("additive","lowrank","gated","experts","residual")
EDGES = np.array((2.8,3.2,3.5,3.8,4.25,5.,6.))


def add_routes(parts, old):
    with ThreadPoolExecutor(4) as pool:
        path = np.concatenate(list(pool.map(build,parts)))
    d=old["distance"][:,:1]
    return dict(old, straight=np.column_stack((d,path[:,:4])),
                detour=np.column_stack((d,path[:,4:15])),
                anisotropy=np.column_stack((d,path[:,15:21])),
                availability=np.column_stack((d,path[:,21:25])),
                hybrid=np.column_stack((old["combined"],path)))


def labels(parts):
    with ThreadPoolExecutor(4) as pool:
        h=np.concatenate(list(pool.map(past,parts)))[:,-5:]
    d=np.concatenate([p["pair"][:,2] for p in parts])
    frac=(h<=3.5).mean(1)
    y=np.column_stack((d-h.mean(1),h.std(1),frac,
                       d-h.min(1),frac>0,(h<3.2).mean(1),
                       (h<3.8).mean(1),(h<4.25).mean(1)))
    classes=np.digitize(h,EDGES)
    hist=np.stack([(classes==i).mean(1) for i in range(len(EDGES)+1)],axis=1)
    return y.astype(np.float32),hist.astype(np.float32)


class Generator(nn.Module):
    def __init__(self,n,structure):
        super().__init__()
        self.structure=structure
        self.encode=nn.Linear(n,32)
        self.inner=nn.Linear(32,32)
        self.combine=nn.Linear(n,32)
        self.reduced=nn.Linear(n,8)
        self.expand=nn.Linear(8,32)
        self.linear=nn.Linear(n,16)
        self.head=nn.Linear(32,16)
        self.expert=nn.Linear(32,48)
        self.gate=nn.Linear(n,3)

    def forward(self,x):
        h=F.silu(self.encode(x))
        if self.structure=="lowrank":
            h=h+F.silu(self.expand(self.reduced(x).square()))
        elif self.structure=="gated":
            h=F.silu(self.inner(h))*self.combine(x).sigmoid()
        elif self.structure=="experts":
            e=self.expert(h).reshape(-1,3,16)
            return (self.gate(x).softmax(1)[:,:,None]*e).sum(1)
        elif self.structure=="residual":
            h=h+F.silu(self.inner(h))
        out=self.head(h)
        return out+self.linear(x) if self.structure=="additive" else out


def train_generator(x,target,hist,architecture,mode,seed):
    torch.manual_seed(seed)
    model=Generator(x.shape[1],architecture).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    for _ in range(4):
        for i in torch.randperm(len(x),device=x.device).split(4096):
            z=model(x[i])
            y=target[i]
            if mode=="point":
                loss=F.binary_cross_entropy_with_logits(z[:,2],y[:,2])
            elif mode=="moments":
                loss=F.smooth_l1_loss(z[:,[0,1,3]],y[:,[0,1,3]])+F.binary_cross_entropy_with_logits(z[:,2],y[:,2])
            elif mode=="hurdle":
                entered=y[:,4]
                within=F.binary_cross_entropy_with_logits(z[:,2],y[:,2],reduction="none")
                loss=F.binary_cross_entropy_with_logits(z[:,4],entered)+(within*entered).mean()
            elif mode=="distribution":
                loss=-(z[:,8:].log_softmax(1)*hist[i]).sum(1).mean()
            else:
                loss=-(z[:,8:].log_softmax(1)*hist[i]).sum(1).mean()
                loss=loss+F.smooth_l1_loss(z[:,[0,1,3]],y[:,[0,1,3]])
                loss=loss+F.binary_cross_entropy_with_logits(z[:,[2,4,5,6,7]],y[:,[2,4,5,6,7]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    assert model.encode.weight.grad.is_cuda
    return model.cpu().eval()


def transform(model,x,mode):
    p=[]
    with torch.inference_mode():
        for b in x.split(32768):
            z=model(b)
            if mode=="point":
                v=z[:,2:3].sigmoid()
            elif mode=="moments":
                v=torch.column_stack((z[:,0],z[:,1].clamp_min(0),z[:,2].sigmoid(),z[:,3]))
            elif mode=="hurdle":
                chance=z[:,4].sigmoid()
                v=torch.column_stack((chance,chance*z[:,2].sigmoid()))
            elif mode=="distribution":
                v=z[:,8:].softmax(1)
            else:
                v=torch.column_stack((z[:,[0,1,3]],z[:,[2,4,5,6,7]].sigmoid(),z[:,8:].softmax(1)))
            p.append(v)
    return torch.cat(p).numpy()


def train_readout(x,base,y,seed):
    torch.manual_seed(seed)
    x=torch.from_numpy(design(x,base))
    center,scale=x.mean(0),x.std(0).clamp_min(.001)
    z=((x-center)/scale).clamp(-8,8).cuda()
    prior=x[:,0].contiguous().cuda()
    y=torch.from_numpy(y.astype(np.float32)).cuda()
    model=nn.Sequential(nn.Linear(z.shape[1],20),nn.SiLU(),nn.Linear(20,1)).cuda()
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    opt=torch.optim.AdamW(model.parameters(),lr=.002)
    for _ in range(3):
        for i in torch.randperm(len(z),device=z.device).split(4096):
            correction=.35*model(z[i]).flatten().tanh()
            loss=F.binary_cross_entropy_with_logits(prior[i]+correction,y[i])+.01*correction.square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    assert model[-1].weight.grad.is_cuda
    return model.cpu().eval(),center,scale


def readout(model,center,scale,x,base):
    a=torch.from_numpy(design(x,base))
    with torch.inference_mode():
        z=((a-center)/scale).clamp(-8,8)
        c=torch.cat([model(b).flatten().tanh() for b in z.split(32768)])
        return (a[:,0]+.35*c).sigmoid().numpy()


def observed_anchor(parts,cpu_labels,base,valid,valid_labels,valid_base):
    ckpt=DEST/"observed-reference.pt"
    y=np.concatenate([p["y"] for p in parts])
    columns=(0,1,2,3)
    reference_fit(design(cpu_labels[:,columns],base),y,ckpt)
    pred=reference_infer(design(valid_labels[:,columns],valid_base),ckpt)
    return score(valid,pred)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--limit",type=int,default=300)
    args=parser.parse_args()
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    DEST.mkdir(parents=True,exist_ok=True)
    names=tuple(json.loads((OUT/"volume_training_panel.json").read_bytes())["proteins"])[:24]
    gs,rs=names[::2],names[1::2]
    valid=panel10()
    assert not set(gs)&set(rs) and not set(names)&set(valid+TEST+HISTORICAL)
    tr,a,bt=data(names)
    va,b,bv=data(valid)
    a,b=add_routes(tr,a),add_routes(va,b)
    teacher,hist=labels(tr)
    actual,_=labels(va)
    all_y=np.concatenate([p["y"] for p in tr])
    group=np.concatenate([np.full(len(p["y"]),p["protein"] in gs) for p in tr])
    gi,ri=np.flatnonzero(group),np.flatnonzero(~group)
    rng=np.random.default_rng(300)
    gi,ri=[np.sort(rng.choice(i,min(len(i),65536),replace=False)) for i in (gi,ri)]
    t=torch.from_numpy(teacher[gi]).cuda()
    h=torch.from_numpy(hist[gi]).cuda()
    summary=DEST/"results.json"
    source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if summary.exists():
        report=json.loads(summary.read_bytes())
        if report["source_sha256"]!=source:
            raise RuntimeError("GA source changed since recorded trials")
    else:
        report=dict(generator=gs,readout=rs,validation=valid,source_sha256=source,
                    generator_rows=len(gi),readout_rows=len(ri),
                    cuda=torch.cuda.get_device_name(),baseline=score(va,bv),
                    observed=observed_anchor(tr,teacher,bt,va,actual,bv),trials={})
        record(summary,report)
    print(f"GA prepared {len(gi)} generator + {len(ri)} readout candidates",flush=True)
    for vi,name in enumerate(VIEWS):
        train=design(a[name],bt)
        valid_x=design(b[name],bv)
        center=torch.from_numpy(train[gi].mean(0))
        scale=torch.from_numpy(np.maximum(train[gi].std(0),.001))
        def normalize(x):
            return ((torch.from_numpy(x)-center)/scale).clamp(-8,8)
        x=normalize(train[gi]).cuda()
        read_x=normalize(train[ri])
        v=normalize(valid_x)
        for mi,mode in enumerate(TARGETS):
            for si,structure in enumerate(ARCHITECTURES):
                i=vi*25+mi*5+si+1
                ident=f"GA{i:02d}"
                if i>args.limit or ident in report["trials"]:
                    continue
                started=time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                generator=train_generator(x,t,h,structure,mode,10000+i)
                train_repr=transform(generator,read_x,mode)
                correction,rc,rsd=train_readout(train_repr,bt[ri],all_y[ri],20000+i)
                elapsed=time.perf_counter()-started
                started=time.perf_counter()
                validation_repr=transform(generator,v,mode)
                p=readout(correction,rc,rsd,validation_repr,bv)
                path=DEST/f"{ident}.pt"
                torch.save(dict(generator=generator.state_dict(),correction=correction.state_dict(),
                                name=name,mode=mode,structure=structure,center=center,scale=scale,
                                rc=rc,rsd=rsd),path)
                result=score(va,p)
                row=dict(view=name,target=mode,architecture=structure,metrics=result,
                         fit_seconds=elapsed,cpu_seconds=time.perf_counter()-started,
                         gpu_peak=torch.cuda.max_memory_allocated(),checkpoint=path.name,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                report["trials"][ident]=row
                record(summary,report)
                print(ident,name,mode,structure,result["auprc"],flush=True)
    if len(report["trials"])==300:
        best=max(report["trials"],key=lambda n:report["trials"][n]["metrics"]["auprc"])
        saved=torch.load(DEST/f"{best}.pt",map_location="cpu",weights_only=True)
        test,xx,p0=data(TEST)
        q=torch.from_numpy(design(add_routes(test,xx)[saved["name"]],p0))
        generator=Generator(q.shape[1],saved["structure"]).eval()
        generator.load_state_dict(saved["generator"])
        out=transform(generator,((q-saved["center"])/saved["scale"]).clamp(-8,8),saved["mode"])
        correction=nn.Sequential(nn.Linear(out.shape[1]+1,20),nn.SiLU(),nn.Linear(20,1)).eval()
        correction.load_state_dict(saved["correction"])
        p=readout(correction,saved["rc"],saved["rsd"],out,p0)
        record(DEST/"test.json",dict(chosen=best,proteins=TEST,
                                    baseline=score(test,p0),corrected=score(test,p)))


if __name__=="__main__":
    main()
