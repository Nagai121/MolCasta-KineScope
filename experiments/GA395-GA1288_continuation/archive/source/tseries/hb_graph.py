"""Origin-only learned message passing over hydrogen-bond options."""
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

from .data import TRAIN, VALIDATION
from .hb_single_structure import OUT, key, load, validation_report
from .search_events import EXTRA_TRAINING, dump, view


class OptionsNetwork(nn.Module):
    def __init__(self, size, hidden, rounds, weighted=False, predicts_motion=False):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(size,hidden),nn.LayerNorm(hidden),nn.SiLU())
        self.layers=nn.ModuleList(nn.Sequential(nn.Linear(hidden*3+2,hidden),nn.SiLU(),
                                              nn.Linear(hidden,hidden)) for _ in range(rounds))
        self.gates=nn.ModuleList(nn.Linear(hidden,2) for _ in range(rounds)) if weighted else None
        self.head=nn.Linear(hidden,2 if predicts_motion else 1)

    def forward(self, x, endpoints, size):
        h=self.encoder(x)
        for index,layer in enumerate(self.layers):
            messages=[]
            counts=[]
            weights=None if self.gates is None else .05+torch.sigmoid(self.gates[index](h))
            for axis,ids in enumerate(endpoints.T):
                count=h.new_zeros((size,1)).index_add_(0,ids,h.new_ones((len(h),1)))[ids]-1
                if weights is None:
                    total=h.new_zeros((size,h.shape[1])).index_add_(0,ids,h)[ids]-h
                    message=total/count.clamp_min(1)
                else:
                    weight=weights[:,axis:axis+1]
                    value=h*weight
                    total=h.new_zeros((size,h.shape[1])).index_add_(0,ids,value)[ids]-value
                    mass=h.new_zeros((size,1)).index_add_(0,ids,weight)[ids]-weight
                    message=total/mass.clamp_min(.05)
                messages.append(message)
                counts.append(torch.log1p(count.clamp_min(0)))
            h=h+.3*layer(torch.cat([h,*messages,*counts],dim=1))
        return self.head(h).squeeze(1)


def graphs(parts,config,training=False,with_motion=False):
    for part in parts:
        x=view(part,config)
        ends=part["pair"][:,:2].astype(np.int64)
        for start,end in zip(np.r_[0,part["ends"][:-1]],part["ends"]):
            if end>start:
                _, inverse=np.unique(ends[start:end],return_inverse=True)
                pair=inverse.reshape(-1,2)
                labels=part["y"][start:end] if training else None
                if training and with_motion:
                    delta=np.clip(part["future_distance"][start:end]-part["pair"][start:end,2],-3.,3.)
                    labels=np.column_stack((labels,delta)).astype(np.float32)
                yield x[start:end],labels,pair,int(pair.max())+1


def batches(rows, order, center, scale, device, graph_limit):
    for start in range(0,len(order),graph_limit):
        group=[rows[int(i)] for i in order[start:start+graph_limit]]
        x,y,edges,sizes=zip(*group)
        offsets=np.r_[0,np.cumsum(sizes[:-1])]
        ends=np.concatenate([pair+offset for pair,offset in zip(edges,offsets)])
        target=None if y[0] is None else torch.as_tensor(np.concatenate(y),device=device,dtype=torch.float32)
        yield (torch.as_tensor((np.concatenate(x)-center)/scale,device=device),target,
               torch.as_tensor(ends,device=device,dtype=torch.long),sum(sizes))


def read_and_predict(parts,folder,record):
    torch.set_num_threads(3)
    state=torch.load(folder/record["fit"]["checkpoint"],map_location="cpu",weights_only=True)
    center,scale=state["center"].numpy(),state["scale"].numpy()
    motion=state.get("predicts_motion",False)
    model=OptionsNetwork(len(center),state["hidden"],state["rounds"],state.get("weighted",False),motion).eval()
    model.load_state_dict(state["model"])
    y=[]
    with torch.inference_mode():
        for part in parts:
            rows=list(graphs([part],record["config"]))
            output=[]
            for x,_,ends,size in batches(rows,np.arange(len(rows)),center,scale,"cpu",6):
                scores=model(x,ends,size)
                output.append(torch.sigmoid(scores[:,0] if motion else scores).numpy())
            y.append(np.concatenate(output))
    return y


def fit_model(name,folder,train,config,rounds,epochs=16,weighted=False,motion_weight=0.):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    torch.manual_seed(27182)
    motion=motion_weight>0.
    rows=list(graphs(train,config,training=True,with_motion=motion))
    all_x=np.concatenate([r[0] for r in rows])
    center=all_x.mean(0,dtype=np.float64).astype(np.float32)
    scale=np.maximum(1e-6,all_x.std(0,dtype=np.float64)).astype(np.float32)
    del all_x
    hidden=config[4]
    model=OptionsNetwork(len(center),hidden,rounds,weighted,motion).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.02)
    torch.cuda.reset_peak_memory_stats()
    clock=time.perf_counter()
    rng=np.random.default_rng(27182)
    for _ in range(epochs):
        for x,y,ends,size in batches(rows,rng.permutation(len(rows)),center,scale,"cuda",5):
            optimizer.zero_grad(set_to_none=True)
            scores=model(x,ends,size)
            if motion:
                loss=nn.functional.binary_cross_entropy_with_logits(scores[:,0],y[:,0])
                loss+=motion_weight*nn.functional.smooth_l1_loss(scores[:,1],y[:,1],beta=.5)
            else:
                loss=nn.functional.binary_cross_entropy_with_logits(scores,y)
            loss.backward()
            optimizer.step()
    torch.cuda.synchronize()
    checkpoint=folder/f"{name}.pt"
    torch.save(dict(model=model.cpu().state_dict(),center=torch.from_numpy(center),
                    scale=torch.from_numpy(scale),hidden=hidden,rounds=rounds,weighted=weighted,
                    predicts_motion=motion),checkpoint)
    return dict(checkpoint=checkpoint.name,sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                train_rows=sum(len(x[1]) for x in rows),
                train_positives=sum(int((x[1][:,0] if motion else x[1]).sum()) for x in rows),
                updater="AdamW",cuda="cuda:0",fit_seconds=time.perf_counter()-clock,
                cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(),message_rounds=rounds,
                epochs=epochs,weighted_neighbors=weighted,motion_weight=motion_weight)


def main():
    import argparse
    from .hb_directional import add_directional
    from .hb_single_structure import ADDITIONAL_TRAINING

    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument("--volume",action="store_true")
    modes.add_argument("--expanded10",action="store_true")
    parser.add_argument("--weighted",action="store_true")
    parser.add_argument("--motion",action="store_true")
    args=parser.parse_args()
    if (args.weighted or args.motion) and not args.expanded10:
        parser.error("--weighted and --motion require --expanded10")
    folder=OUT/("ap10_graph_motion" if args.motion else
                "ap10_graph_weighted" if args.weighted else "ap10_graph_expanded" if args.expanded10
                else "graph_volume" if args.volume else "graph")
    folder.mkdir(exist_ok=True)
    basic=load(TRAIN,25,targets=args.motion,geometry=True)
    middle=load(EXTRA_TRAINING,50,targets=args.motion,geometry=True)
    train=basic+middle
    validation_proteins=VALIDATION
    if args.expanded10:
        from .hb_validation10 import panel10
        validation_proteins=panel10()
    validation=load(validation_proteins,100,geometry=True)
    choices=((118,"environment",False,64,1),(118,"local",True,64,2),(142,"competition",True,96,3))
    if args.volume or args.expanded10:
        panels=(OUT/"diverse_training_panel.json",OUT/"volume_training_panel.json")
        if args.expanded10:
            panels+=(OUT/"volume_96_training_panel.json",OUT/"volume_128_training_panel.json")
        if args.motion:
            panels+=(OUT/"ap10_volume_144_training_panel.json",)
        extra=tuple(p for panel in panels for p in json.loads(panel.read_text(encoding="utf-8"))["proteins"])
        if set(TRAIN+EXTRA_TRAINING+ADDITIONAL_TRAINING+extra)&set(validation_proteins):
            raise ValueError("Training and validation proteins overlap")
        middle+=load(ADDITIONAL_TRAINING,50,targets=args.motion,geometry=True)
        late=load(extra,100,targets=args.motion,geometry=True)
        for parts,stride in ((basic,25),(middle,50),(late,100),(validation,100)):
            for part in parts:
                add_directional(part,stride)
        train=basic+middle+late
        choices=((118,"directional",False,64,1),(118,"directional",True,96,2))
    if args.expanded10:
        choices=((118,"directional",False,96,1),(166,"directional",True,96,2))
        print("AP_GRAPH_DATA",len({p["protein"] for p in train}),len(validation_proteins),flush=True)
    if args.weighted or args.motion:
        from .hb_cloud import add_cloud
        from .hb_explicit_context import add_explicit_context
        from .hb_explicit_hydrogen import add_explicit
        from .hb_occupied import add_occupied

        for parts,stride in ((basic,25),(middle,50),(late,100),(validation,100)):
            for part in parts:
                add_cloud(part,stride)
                add_occupied(part,stride)
                add_explicit(part,stride)
                add_explicit_context(part)
        choices=((118,"directional",False,96,1),(118,"stacked-explicit_context",False,96,1))
    if args.motion:
        choices=((118,"stacked-explicit_context",False,96,1),)*2
    ledger={}
    first=1861 if args.motion else 1741 if args.weighted else 1711 if args.expanded10 else 1221 if args.volume else 631
    for i,(width,kind,inherit,hidden,rounds) in enumerate(choices,first):
        name=f"T1I{i}"
        config=("hb-gnn",width,f"hb-origin-{kind}",inherit,hidden,1.0)
        row=dict(config=config,fit=fit_model(name,folder,train,config,rounds,
                                            epochs=12 if args.expanded10 else 16,weighted=args.weighted or args.motion,
                                            motion_weight=.12 if args.motion and i==first+1 else 0.))
        row["validation"]=validation_report(validation,read_and_predict(validation,folder,row))
        ledger[name]=row
        dump(folder/"validation.json",ledger)
        print(name,row["validation"]["candidate"]["auprc"],key(row),flush=True)
    best=max(ledger,key=lambda n:key(ledger[n]))
    dump(folder/"selection.json",dict(name=best,**ledger[best]))


if __name__=="__main__":
    main()
