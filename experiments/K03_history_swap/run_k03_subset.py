import numpy as np, json
exec(open("/home/hatch/workspace/kinescope/work/run_experiments.py").read().split("CONDITIONS=")[0].split('print("train"')[0])
train=load(TRAIN_P); test=load(TEST_P)
Xtr=cat(train,"X"); Xte=cat(test,"X"); Htr=cat(train,"hist"); Hte=cat(test,"hist")
Ytr=cat(train,"y"); Yte=cat(test,"y")
# rebuild swap replica with matched mask
def swap_replica_mask(parts):
    outs=[]; masks=[]
    for pi in range(0,len(parts),3):
        grp=parts[pi:pi+3]
        for ri in range(3):
            src=grp[(ri+1)%3]; dst=grp[ri]
            key_src={(int(a),int(b)):i for i,(a,b) in enumerate(zip(src["donor_atom"],src["acceptor_atom"]))}
            h=np.empty_like(dst["hist"].T); mask=np.zeros(len(h),bool)
            for i,(a,b) in enumerate(zip(dst["donor_atom"],dst["acceptor_atom"])):
                j=key_src.get((int(a),int(b)))
                if j is None: h[i]=dst["hist"].T[i]
                else: h[i]=src["hist"].T[j]; mask[i]=True
            outs.append(h); masks.append(mask)
    return np.concatenate(outs), np.concatenate(masks)
Hsw,mask=swap_replica_mask(test)
print("matched",mask.sum(),"of",len(mask))
res={}
for name,hte in [("base_matched_subset",Hte),("swap_replica_matched_subset",Hsw)]:
    res[name]={}
    for hi,h in enumerate(HORIZONS):
        p=predict(np.concatenate([Xtr,hist6(Htr)],axis=1),Ytr[:,hi],np.concatenate([Xte,hist6(hte)],axis=1))
        res[name][str(h)]=metrics(Yte[mask,hi],p[mask])
        print(name,h,res[name][str(h)])
json.dump(res,open("/home/hatch/workspace/kinescope/work/results_k03_subset.json","w"),indent=2)
