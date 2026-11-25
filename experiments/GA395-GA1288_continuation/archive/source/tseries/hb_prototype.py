"""CUDA-learned local structural prototypes with CPU probability inference."""
import hashlib
import time

import numpy as np
import torch

from .data import TRAIN,VALIDATION
from .hb_single_structure import OUT, key, load, validation_report
from .search_events import EXTRA_TRAINING, dump


def columns(part):
    base=part["base"]
    local=part["local"]
    return np.column_stack((base[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,39,40,60,61,118,119,122,123]],
                            local[:,:12])).astype(np.float32)


def std_out(x,center,scale):
    return np.clip((x-center)/scale,-5,5)


def kernels(part,state):
    x=std_out(columns(part),state["mean"],state["scale"])
    anchors=state["centers"]
    prior=np.clip(part["old"].astype(np.float64),1e-6,1-1e-6)
    answer=[]
    for start in range(0,len(x),4096):
        a=x[start:start+4096]
        d=(a*a).sum(1,keepdims=True)-2*a@anchors.T+(anchors*anchors).sum(1)
        ids=np.argpartition(d,5,axis=1)[:,:5]
        selected=np.take_along_axis(d,ids,axis=1)
        w=np.exp(-(selected-selected.min(1,keepdims=True))/state["temperature"])
        answer.append((w*state["prob"][ids]).sum(1)/w.sum(1))
    value=np.clip(np.concatenate(answer),1e-6,1-1e-6)
    return np.column_stack((np.log(value/(1-value)),np.log(prior/(1-prior))))


def predict_prototypes(parts,folder,record):
    path=folder/record["fit"]["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record["fit"]["sha256"]:
        raise ValueError("Prototype checkpoint changed")
    with np.load(path) as z:
        state={k:z[k] for k in z.files}
    return [1/(1+np.exp(-np.clip(kernels(p,state)@state["calibration"][:2]+state["calibration"][2],-40,40)))
            for p in parts]


def train(tag,folder,fit_parts,calibration,size,temperature,emphasis):
    start=time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(27182)
    x=np.concatenate([columns(p) for p in fit_parts])
    y=np.concatenate([p["y"] for p in fit_parts])
    center=x.mean(0,dtype=np.float64).astype(np.float32)
    span=np.maximum(x.std(0,dtype=np.float64),1e-4).astype(np.float32)
    span[:2]/=emphasis
    xx=torch.as_tensor(std_out(x,center,span),device="cuda")
    yy=torch.as_tensor(y,device="cuda",dtype=torch.float32)
    near=torch.as_tensor(x[:,0]<=4.5,device="cuda",dtype=torch.float32)
    weights=1+near*2
    centers=xx[torch.randperm(len(x),device="cuda")[:size]].clone()
    for _ in range(13):
        elements=torch.zeros((size,xx.shape[1]),device="cuda")
        counts=torch.zeros(size,device="cuda")
        for indices in torch.randperm(len(x),device="cuda")[:65000].split(4096):
            batch=xx[indices]
            nearest=torch.cdist(batch,centers).argmin(1)
            elements.index_add_(0,nearest,batch*weights[indices,None])
            counts.index_add_(0,nearest,weights[indices])
        centers=torch.where(counts[:,None]>0,elements/counts[:,None].clamp_min(1),centers)
    positive=torch.zeros(size,device="cuda")
    count=torch.zeros(size,device="cuda")
    for ids in torch.arange(len(x),device="cuda").split(4096):
        distances=torch.cdist(xx[ids],centers)
        near=distances.argmin(1)
        positive.index_add_(0,near,yy[ids])
        count.index_add_(0,near,torch.ones_like(yy[ids]))
    rates=(positive+4*yy.mean())/(count+4)
    state=dict(mean=center,scale=span,centers=centers.cpu().numpy(),prob=rates.cpu().numpy(),
               temperature=temperature)
    calib_x=torch.as_tensor(np.concatenate([kernels(p,state) for p in calibration]),device="cuda",dtype=torch.float32)
    calib_y=torch.as_tensor(np.concatenate([p["y"] for p in calibration]),device="cuda",dtype=torch.float32)
    raw=torch.nn.Parameter(torch.zeros(2,device="cuda"))
    bias=torch.nn.Parameter(torch.zeros((),device="cuda"))
    opt=torch.optim.Adam([raw,bias],lr=.025)
    for _ in range(180):
        opt.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(calib_x@raw.softmax(0)+bias,calib_y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    state["calibration"]=np.r_[raw.softmax(0).detach().cpu().numpy(),float(bias.detach().cpu())]
    path=folder/f"{tag}.npz"
    np.savez(path,**state)
    return dict(checkpoint=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                cuda="cuda:0",updater="CUDA k-means and Adam",fit_seconds=time.perf_counter()-start,
                train_rows=len(x),train_positives=int(y.sum()),centers=size,temperature=temperature)


def main():
    folder=OUT/"prototype"
    folder.mkdir(exist_ok=True)
    training=load(TRAIN,25,geometry=True)+load(tuple(p for p in EXTRA_TRAINING if p!="2w0g_A"),50,geometry=True)
    calibration=load(("2w0g_A",),50,geometry=True)
    valid=load(VALIDATION,100,geometry=True)
    log={}
    for i,(size,temp,emphasis) in enumerate(((64,1.,1.5),(128,.8,2.),(256,1.3,2.5),(384,.7,2.)),881):
        name=f"T1I{i}"
        record=dict(config=("hb-prototype",118,"hb-origin-local",False,0,1.),
                    fit=train(name,folder,training,calibration,size,temp,emphasis))
        record["validation"]=validation_report(valid,predict_prototypes(valid,folder,record))
        log[name]=record
        dump(folder/"validation.json",log)
        print(name, record["validation"]["candidate"]["auprc"],key(record),flush=True)
    name=max(log,key=lambda n:key(log[n]))
    dump(folder/"selection.json",dict(name=name,**log[name]))


if __name__=="__main__":
    main()
