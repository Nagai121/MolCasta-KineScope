"""Frozen R8 and T1_NN CPU inference kernels from Internal Research.

Source branch research/t1-02-three-refinement-loops-20261120.
Adaptations are limited to loading MiniTopol inputs and omitting legacy training.
"""
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .data import topology

TASKS = ("hb", "salt", "contact")
THRESHOLDS = (3.5, 4.0, 4.5)
TOPK = {"salt": 512, "contact": 9000}
USE_CONTEXT = True

def distance(v, box):
    """Correct fractional minimum image for a 3x3 triclinic box (row vectors)."""
    box = np.asarray(box, dtype=np.float64)
    if not np.isfinite(box).all() or abs(np.linalg.det(box)) < 1e-9:
        return np.linalg.norm(v, axis=-1)
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-5):
        lengths = np.diag(box)
        v = v - lengths * np.rint(v / lengths)
    else:
        inv = np.linalg.inv(box)
        fractional = v @ inv
        fractional -= np.rint(fractional)
        v = fractional @ box
    return np.linalg.norm(v, axis=-1)


@dataclass
class Options:
    x: np.ndarray
    y: np.ndarray
    i: np.ndarray
    j: np.ndarray
    n_atoms: int
    name: str
    t: int
    lag: int
    task: str


def mic_dist(pos, box, i, j):
    if not len(i):
        return np.empty(0,dtype=np.float32)
    return distance(pos[i]-pos[j],box).astype(np.float32)


def features(s, ids, task_id, lag, pos=None):
    i=s.i[ids];j=s.j[ids]
    p=s.pos if pos is None else pos
    dist=mic_dist(p,s.box,i,j)
    ca_d=mic_dist(p,s.box,s.atom_ca[i],s.atom_ca[j])
    sep=np.abs(s.atom_res[i]-s.atom_res[j]).astype(np.float32)
    elems_i=s.elements[i];elems_j=s.elements[j]
    nt=len(ids)
    fields=[
      dist,1/(dist+1),np.log1p(sep),np.minimum(sep,100)/100,
      ca_d,
      s.donor[i].astype(np.float32),s.donor[j].astype(np.float32),
      s.acceptor[i].astype(np.float32),s.acceptor[j].astype(np.float32),
      s.poscharge[i].astype(np.float32),s.poscharge[j].astype(np.float32),
      s.negcharge[i].astype(np.float32),s.negcharge[j].astype(np.float32),
    ]
    for k in range(5):
        fields.append((elems_i==k).astype(np.float32))
    for k in range(5):
        fields.append((elems_j==k).astype(np.float32))
    if USE_CONTEXT:
        dens_i=s.density8[i]
        dens_j=s.density8[j]
        d12_i=s.density12[i]
        d12_j=s.density12[j]
        fields.extend([dens_i,dens_j,d12_i,d12_j,
                       np.abs(dens_i-dens_j),
                       np.minimum(dens_i,dens_j),
                       np.abs(d12_i-d12_j),
                       np.minimum(d12_i,d12_j)])
    fields.append(np.full(nt,np.log1p(lag/10),np.float32))
    for k in range(3):
        fields.append(np.full(nt,float(k==task_id),np.float32))
    return np.column_stack(fields).astype(np.float32)


class Gate(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,48),nn.SiLU(),nn.Linear(48,32),
                               nn.SiLU(),nn.Linear(32,1))
    def forward(self,x):
        return self.net(x).squeeze(-1)


class OptionsNetwork(nn.Module):
    """ROD-type message-passing variants; no canonical ROD math is assumed.

    baseline:  previous R4 aggregate of all alternatives per edge
    residual:  same relation, but accumulate logit refinements
    endpoint:  treat competing alternatives around each endpoint separately,
               then use endpoint-symmetric sum and absolute difference
    """
    def __init__(self,d,steps,hidden=48,mode="baseline"):
        super().__init__()
        if mode not in ("baseline","residual","endpoint"):
            raise ValueError("unknown relation mode")
        self.mode=mode
        self.steps=steps
        self.encoder=nn.Sequential(nn.Linear(d,hidden),nn.SiLU(),
                                   nn.Linear(hidden,hidden),nn.SiLU())
        self.initial=nn.Linear(hidden,1)
        context_dim=hidden*3+4 if mode=="endpoint" else hidden*2+2
        self.updates=nn.ModuleList([
            nn.Sequential(nn.Linear(context_dim,hidden),nn.SiLU(),
                          nn.Linear(hidden,hidden)) for _ in range(steps)
        ])
        self.heads=nn.ModuleList([nn.Linear(hidden,1) for _ in range(steps)])

    def forward(self,x,ends,n_atoms):
        h=self.encoder(x)
        logits=self.initial(h).squeeze(1)
        for up,head in zip(self.updates,self.heads):
            p=torch.sigmoid(logits).unsqueeze(1)
            mass=h.new_zeros((n_atoms,1))
            accumulated=h.new_zeros((n_atoms,h.shape[1]))
            count=h.new_zeros((n_atoms,1))
            for side in (0,1):
                idx=ends[:,side]
                mass.index_add_(0,idx,p)
                accumulated.index_add_(0,idx,h*p)
                count.index_add_(0,idx,torch.ones_like(p))
            if self.mode=="endpoint":
                # Each endpoint's alternatives exclude this very edge.
                mass_l=(mass[ends[:,0]]-p).clamp_min(0)
                mass_r=(mass[ends[:,1]]-p).clamp_min(0)
                left=(accumulated[ends[:,0]]-h*p)/mass_l.clamp_min(1e-4)
                right=(accumulated[ends[:,1]]-h*p)/mass_r.clamp_min(1e-4)
                cnt_l=(count[ends[:,0]]-1).clamp_min(0)
                cnt_r=(count[ends[:,1]]-1).clamp_min(0)
                # Invariant under atom i/j exchange, unlike ordered endpoints.
                ctx=torch.cat([
                    h,0.5*(left+right),torch.abs(left-right),
                    torch.log1p(mass_l+mass_r),torch.log1p(torch.abs(mass_l-mass_r)),
                    torch.log1p(cnt_l+cnt_r),torch.log1p(torch.abs(cnt_l-cnt_r)),
                ],dim=1)
            else:
                shared=accumulated[ends[:,0]]+accumulated[ends[:,1]]-2*h*p
                other_mass=(mass[ends[:,0]]+mass[ends[:,1]]-2*p).clamp_min(0)
                other_count=(count[ends[:,0]]+count[ends[:,1]]-2).clamp_min(0)
                ctx=torch.cat([
                    h,shared/other_mass.clamp_min(1e-4),
                    torch.log1p(other_mass),torch.log1p(other_count)],dim=1)
            h=h+0.35*torch.tanh(up(ctx))
            delta=head(h).squeeze(1)
            logits=logits+0.5*delta if self.mode in ("residual","endpoint") else delta
        return logits


def periodic_vector(v,box):
    box=np.asarray(box,dtype=np.float64)
    if not np.isfinite(box).all() or abs(np.linalg.det(box))<1e-9:
        return v
    if np.allclose(box,np.diag(np.diag(box)),atol=1e-5):
        diagonal=np.diag(box)
        return v-diagonal*np.rint(v/diagonal)
    inv=np.linalg.inv(box)
    f=v@inv
    return (f-np.rint(f))@box


def norm(x):
    return np.linalg.norm(x,axis=1).clip(1e-7)


def cosine(x,y):
    return np.clip(np.sum(x*y,axis=1)/(norm(x)*norm(y)),-1,1)


def screen_snapshot(s,task_id,lag,gate,gm,gs,cap,proximity_fraction=.2,
                    chunk=20000):
    """Complete candidate scoring, current frame only. No future positions read.

    Exact same physical mask as original R8; no 6/10/12Å distance cutoff.
    Candidate limit is a measured top-K option budget, not a prior search limit.
    """
    task=TASKS[task_id]
    ids=s.physics_ids[task]
    threshold=THRESHOLDS[task_id]
    all_n=len(s.i)
    retain_i=np.empty(0,np.int64)
    retain_s=np.empty(0,np.float32)
    near_i=np.empty(0,np.int64)
    near_d=np.empty(0,np.float32)
    phys_count=0
    gate.eval()
    with torch.no_grad():
        for begin in range(0,len(ids),chunk):
            rows=ids[begin:begin+chunk]
            f=features(s,rows,task_id,lag)
            keep=f[:,0]>threshold
            if not keep.any():continue
            rows=rows[keep];f=f[keep]
            phys_count+=len(rows)
            scores=torch.sigmoid(gate(torch.from_numpy(
                np.clip((f-gm)/gs,-12,12)))).numpy().astype(np.float32)
            distance=f[:,0]
            if proximity_fraction:
                count=min(cap,len(rows))
                nearest=np.argpartition(distance,count-1)[:count]
                nd=np.concatenate((near_d,distance[nearest]))
                ni=np.concatenate((near_i,rows[nearest]))
                take=np.argsort(nd,kind="stable")[:cap]
                near_d,near_i=nd[take],ni[take]
            count=min(cap,len(rows))
            highest=np.argpartition(scores,-count)[-count:]
            all_scores=np.concatenate((retain_s,scores[highest]))
            all_idx=np.concatenate((retain_i,rows[highest]))
            if len(all_scores)>cap:
                pick=np.argpartition(all_scores,-cap)[-cap:]
                all_scores,all_idx=all_scores[pick],all_idx[pick]
            retain_s,retain_i=all_scores,all_idx
    if not len(retain_i):
        return None,{"full_all_atom_pairs":int(all_n),
                     "physical_after_chemistry":int(len(ids)),
                     "initially_unformed_physical":int(phys_count),
                     "nn_retained_options":0}
    reserve=min(int(round(cap*proximity_fraction)),len(near_i))
    reserved=near_i[:reserve]
    mask=~np.isin(retain_i,reserved)
    remain=retain_i[mask]
    rscore=retain_s[mask]
    n_remaining=max(0,cap-len(reserved))
    if len(remain)>n_remaining:
        take=np.argpartition(rscore,-n_remaining)[-n_remaining:]
        remain=remain[take]
    selected=np.sort(np.concatenate((reserved,remain)))
    f=features(s,selected,task_id,lag)
    with torch.no_grad():
        probabilities=torch.sigmoid(gate(torch.from_numpy(
            np.clip((f-gm)/gs,-12,12)))).numpy().astype(np.float32)
    original_features=np.column_stack((f,probabilities)).astype(np.float32)
    graph=Options(
        original_features,np.zeros(len(selected),np.float32),
        s.i[selected],s.j[selected],s.n,s.name,s.t,lag,task)
    counts={"full_all_atom_pairs":int(all_n),
            "physical_after_chemistry":int(len(ids)),
            "initially_unformed_physical":int(phys_count),
            "nn_retained_options":int(len(selected))}
    return graph,counts


def enrich(s,g):
    """Current-origin structural descriptors plus competition on the selected graph.

    The input is rotation/translation invariant. These 18 additional fields are
    not available in the old R8 independent options head.
    """
    i=g.i.astype(np.int64);j=g.j.astype(np.int64)
    p=s.pos.astype(np.float64,copy=False)
    sep=periodic_vector(p[j]-p[i],s.box)
    vi=periodic_vector(p[i]-p[s.atom_ca[i]],s.box)
    vj=periodic_vector(p[j]-p[s.atom_ca[j]],s.box)
    ca_sep=periodic_vector(p[s.atom_ca[j]]-p[s.atom_ca[i]],s.box)
    cos_i=cosine(vi,sep)
    cos_j=cosine(vj,-sep)
    cos_pair=cosine(vi,vj)
    # Invariant to swapping the two endpoints.
    orientation=np.column_stack((
        norm(vi)+norm(vj),np.abs(norm(vi)-norm(vj)),
        cos_i+cos_j,np.abs(cos_i-cos_j),cos_pair,
        cosine(sep,ca_sep),norm(ca_sep),
    ))
    gate_prob=np.clip(g.x[:,-1],1e-6,1)
    n=s.n
    degree=np.bincount(np.concatenate((i,j)),minlength=n).astype(np.float32)
    mass=np.bincount(np.concatenate((i,j)),
                    weights=np.tile(gate_prob,2),minlength=n).astype(np.float32)
    max_score=np.zeros(n,dtype=np.float32)
    np.maximum.at(max_score,i,gate_prob)
    np.maximum.at(max_score,j,gate_prob)
    competitor_i=np.maximum(degree[i]-1,0)
    competitor_j=np.maximum(degree[j]-1,0)
    total_mass_i=np.maximum(mass[i]-gate_prob,0)
    total_mass_j=np.maximum(mass[j]-gate_prob,0)
    rivalry=np.column_stack((
        np.log1p(competitor_i+competitor_j),
        np.log1p(np.abs(competitor_i-competitor_j)),
        np.log1p(total_mass_i+total_mass_j),
        np.log1p(np.abs(total_mass_i-total_mass_j)),
        0.5*(max_score[i]+max_score[j]),
        np.abs(max_score[i]-max_score[j]),
    ))
    near=np.column_stack((
        (g.x[:,0]<=6).astype(np.float32),
        np.log1p(np.maximum(g.x[:,0]-6,0)),
        np.log1p(np.maximum(g.x[:,0]-12,0)),
        np.log1p(np.minimum(np.abs(s.atom_res[i]-s.atom_res[j]),300)),
        np.ones(len(i),np.float32),
    ))
    extras=np.column_stack((orientation,rivalry,near)).astype(np.float32)
    assert extras.shape==(len(i),18)
    enriched=np.concatenate((g.x,extras),axis=1).astype(np.float32)
    assert np.isfinite(enriched).all()
    return replace(g,x=enriched)


class T1_NN(nn.Module):
    """Task/time-conditioned relation-network with cautious residual refinement.

    Updates are symmetric in atom pair endpoints; all calculations only use
    the selected options and their shared atoms. An update gate prevents
    replacement of reliable independent logits by unstable relation messages.
    """
    def __init__(self,n_features,hidden=96,steps=3,dropout=0.,relation="symmetric"):
        super().__init__()
        if relation not in ("symmetric","shared"):
            raise ValueError(relation)
        self.steps=int(steps)
        self.hidden=int(hidden)
        self.relation=relation
        self.encoder=nn.Sequential(
            nn.Linear(n_features,hidden),nn.LayerNorm(hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU())
        self.initial=nn.Linear(hidden,1)
        fields=hidden*3+4 if relation=="symmetric" else hidden*2+2
        self.updates=nn.ModuleList(
            nn.Sequential(nn.Linear(fields,hidden),nn.SiLU(),
                          nn.Dropout(dropout),nn.Linear(hidden,hidden))
            for _ in range(self.steps))
        self.update_gates=nn.ModuleList(nn.Linear(fields,hidden) for _ in range(self.steps))
        self.logit_heads=nn.ModuleList(nn.Linear(hidden,1) for _ in range(self.steps))
        self.logit_gates=nn.ModuleList(nn.Linear(hidden,1) for _ in range(self.steps))
        for m in self.logit_gates:
            nn.init.constant_(m.bias,-2.0)
    def forward(self,x,ends,n_atoms):
        h=self.encoder(x)
        logits=self.initial(h).squeeze(-1)
        a,b=ends[:,0],ends[:,1]
        ones=torch.ones((len(h),1),dtype=h.dtype,device=h.device)
        for block,gate,head,logit_gate in zip(
                self.updates,self.update_gates,self.logit_heads,self.logit_gates):
            weights=torch.sigmoid(logits).unsqueeze(1)
            context_sum=h.new_zeros((n_atoms,h.shape[1]))
            mass=h.new_zeros((n_atoms,1))
            count=h.new_zeros((n_atoms,1))
            context_sum.index_add_(0,a,h*weights)
            context_sum.index_add_(0,b,h*weights)
            mass.index_add_(0,a,weights)
            mass.index_add_(0,b,weights)
            count.index_add_(0,a,ones)
            count.index_add_(0,b,ones)
            if self.relation=="symmetric":
                ml=(mass[a]-weights).clamp_min(0)
                mr=(mass[b]-weights).clamp_min(0)
                sl=(context_sum[a]-h*weights)/ml.clamp_min(1e-4)
                sr=(context_sum[b]-h*weights)/mr.clamp_min(1e-4)
                cl=(count[a]-1).clamp_min(0)
                cr=(count[b]-1).clamp_min(0)
                context=torch.cat((
                    h,(sl+sr)*.5,torch.abs(sl-sr),
                    torch.log1p(ml+mr),torch.log1p(torch.abs(ml-mr)),
                    torch.log1p(cl+cr),torch.log1p(torch.abs(cl-cr))
                ),dim=1)
            else:
                m=(mass[a]+mass[b]-2*weights).clamp_min(0)
                c=(count[a]+count[b]-2).clamp_min(0)
                avg=(context_sum[a]+context_sum[b]-2*h*weights)/m.clamp_min(1e-4)
                context=torch.cat((h,avg,torch.log1p(m),torch.log1p(c)),dim=1)
            dh=torch.tanh(block(context))
            h=h+.35*torch.sigmoid(gate(context))*dh
            logits=logits+.5*torch.sigmoid(logit_gate(h)).squeeze(-1)*head(h).squeeze(-1)
        return logits


def build_layout(atoms):
    residue = np.unique(atoms["residue_index"], return_inverse=True)[1]
    name = np.asarray(atoms["name"])
    rn = np.asarray(atoms["residue_name"])
    n_residues = int(residue.max()) + 1
    fallback = np.zeros(n_residues, dtype=np.int32)
    np.maximum.at(fallback, residue, np.arange(len(name), dtype=np.int32))
    ca = fallback.copy()
    ca[residue[name == "CA"]] = np.flatnonzero(name == "CA")
    _, _, context, atom_types = topology(atoms)
    donor = np.isin(np.arange(len(name)), context[5])
    acceptor = np.isin(np.arange(len(name)), context[9])
    positive = ((rn == "LYS") & (name == "NZ")) | ((rn == "ARG") & np.isin(name, ["NE", "NH1", "NH2"]))
    negative = ((rn == "ASP") & np.isin(name, ["OD1", "OD2"])) | ((rn == "GLU") & np.isin(name, ["OE1", "OE2"]))
    elements = np.full(len(name), 4, dtype=np.int8)
    for index, element in enumerate(("C", "N", "O", "S")):
        elements[np.char.startswith(name, element)] = index
    i, j = np.triu_indices(len(name), 1)
    i, j = i.astype(np.int32), j.astype(np.int32)
    separated = np.abs(residue[i] - residue[j]) >= 2
    opposite = (positive[i] & negative[j]) | (positive[j] & negative[i])
    masks = dict(salt=np.flatnonzero(separated & opposite), contact=np.flatnonzero(separated))
    return SimpleNamespace(n=len(name), atom_res=residue, atom_ca=ca[residue], ca=ca,
                           donor=donor, acceptor=acceptor, poscharge=positive, negcharge=negative,
                           elements=elements, atom_types=atom_types, i=i, j=j, physics_ids=masks,
                           residue_names=rn[ca])


def snapshot(layout, positions, box, name, origin):
    box = np.diag(box) if box.shape == (3,) else box
    residue_positions = positions[layout.ca]
    d = distance(residue_positions[:, None] - residue_positions[None, :], box)
    return SimpleNamespace(**vars(layout), pos=positions, box=box, name=name, t=origin,
                           density8=(np.count_nonzero(d <= 8, axis=1) - 1)[layout.atom_res],
                           density12=(np.count_nonzero(d <= 12, axis=1) - 1)[layout.atom_res])


def load_models(directory):
    directory = Path(directory)
    torch.set_num_threads(2)
    root = {"gate": ("ROD_full_stage1_gate.pt", "R8_SHA256SUMS.txt"),
            "salt": ("ROD_full_T3.pt", "R8_SHA256SUMS.txt"),
            "contact": ("T1_B_symmetric.pt", "T1_SHA256SUMS.txt")}
    loaded = {}
    for key, (name, manifest) in root.items():
        files = dict(line.split(maxsplit=1)[::-1] for line in
                     (directory / manifest).read_text(encoding="utf-8").splitlines() if line.strip())
        path = directory / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != files[name]:
            raise ValueError(f"Legacy checkpoint hash failed for {name}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if key == "contact":
            mean, std = (state[f"input_feature_{field}"].numpy() for field in ("mean", "std"))
            model = T1_NN(len(mean), state["hidden"], state["steps"], relation=state["relation"])
        else:
            mean, std = (state[f"feature_{field}"].numpy() for field in ("mean", "std"))
            model = Gate(len(mean)) if key == "gate" else OptionsNetwork(
                len(mean), state["steps"], hidden=state["hidden"], mode=state["mode"])
        model.load_state_dict(state["state"], strict=True)
        loaded[key] = (model.eval(), mean, std, actual)
    return loaded


@torch.inference_mode()
def legacy_scores(s, task, models):
    gate, gm, gs, _ = models["gate"]
    model, mean, std, _ = models[task]
    g, _ = screen_snapshot(s, TASKS.index(task), 10, gate, gm, gs, TOPK[task])
    if g is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    if task == "contact":
        g = enrich(s, g)
    x = torch.from_numpy(np.clip((g.x - mean) / std, -12, 12))
    endpoints = torch.from_numpy(np.column_stack((g.i, g.j)).astype(np.int64))
    p = torch.sigmoid(model(x, endpoints, g.n_atoms)).numpy()
    n_residues = len(s.ca)
    a, b = s.atom_res[g.i], s.atom_res[g.j]
    ids = np.minimum(a, b) * n_residues + np.maximum(a, b)
    return ids, p
