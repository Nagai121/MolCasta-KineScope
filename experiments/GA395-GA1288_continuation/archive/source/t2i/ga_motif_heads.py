"""Motif-specific residual mixture heads with a shared static/dynamic representation."""
import numpy as np
import torch
from torch import nn


class MotifOutput(nn.Module):
    def __init__(self,dim,out,groups):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(dim,128),nn.LayerNorm(128),nn.SiLU(),
                                  nn.Linear(128,64),nn.LayerNorm(64),nn.SiLU())
        self.head=nn.Linear(64,out)
        self.specialized=nn.Linear(64,groups*out)
        self.groups=groups
        self.width=out

    def forward(self,x,group):
        features=self.shared(x)
        index=torch.arange(len(x),device=x.device)
        delta=self.specialized(features).reshape(-1,self.groups,self.width)
        return self.head(features)+.5*delta[index,group]


def routing(x,kind):
    element=(x[:,1]==8).astype(np.int64)
    if kind=="chemistry":
        return element,2
    if kind=="separation":
        return element*2+(x[:,-1]>np.log(5)).astype(np.int64),4
    if kind=="distance":
        return element*2+(x[:,0]>4.25).astype(np.int64),4
    raise ValueError(kind)


if __name__=="__main__":
    from .ga_mixture import main

    main(stability=True,competitive=True,orient_capacity=True,motif_heads=True)
