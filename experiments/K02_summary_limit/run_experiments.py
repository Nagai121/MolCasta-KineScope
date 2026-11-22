import numpy as np, json
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
CACHE=Path("/home/hatch/workspace/kinescope/work/data_cache")
TRAIN_P=("1bgf_A","1ifg_A","1j8r_A","2b0a_A"); TEST_P=("1mix_A","2qt4_A")
HORIZONS=(1,5,10,30,50,90)

def load(prots):
    parts=[]
    for p in prots:
        for r in (1,2,3):
            z=np.load(CACHE/f"{p}_R{r}.npz")
            parts.append({k:z[k] for k in z.files})
    return parts

def cat(parts,key):
    arrs=[p[key].T if key.startswith("hist") else p[key] for p in parts]
    return np.concatenate(arrs)

def hist6(series, bound=3.5):
    # series: (n, T) time axis 0 in D code is frame axis; here rows
    return np.stack([series[:,0],series.mean(1),series.std(1),series.min(1),series[:,-1]-series[:,0],(series<=bound).mean(1)],axis=1).astype(np.float32)

def gen8(series, origin):
    # summary() from hb_trajectory_screen.py adapted: single path, steps=frames, d=origin distance
    delta=series-origin[:,None]
    return np.column_stack((delta.mean(1),delta.std(1),delta.min(1),delta.max(1),
        (series<=3.5).mean(1),(series<=4.0).mean(1),delta[:,-1],
        (np.diff(series,axis=1)<0).mean(1))).astype(np.float32)

def predict(xtr,ytr,xte):
    if np.unique(ytr).size<2: return np.full(len(xte),float(ytr.mean()))
    m=HistGradientBoostingClassifier(max_iter=120,learning_rate=.075,max_leaf_nodes=31,min_samples_leaf=15,l2_regularization=.02,random_state=42)
    m.fit(xtr,ytr); return m.predict_proba(xte)[:,1]

def metrics(y,p):
    p=np.clip(p,1e-8,1-1e-8)
    return dict(n=int(len(y)),positives=int(y.sum()),auprc=float(average_precision_score(y,p)),auc=float(roc_auc_score(y,p)),brier=float(brier_score_loss(y,p)),logloss=float(log_loss(y,p,labels=[0,1])))

train=load(TRAIN_P); test=load(TEST_P)
Xtr=cat(train,"X"); Xte=cat(test,"X")
Htr=cat(train,"hist"); Hte=cat(test,"hist")
Ytr=cat(train,"y"); Yte=cat(test,"y")
print("train",Xtr.shape,"test",Xte.shape,"test pos",Yte.sum(0))

rng=np.random.default_rng(42)
# K01 transforms applied to both train and test hist series
def thin(series, idx): return series[:,idx]
def shuffled(series, seed):
    r=np.random.default_rng(seed); out=series.copy()
    for i in range(len(out)): out[i]=out[i][r.permutation(out.shape[1])]
    return out
def reversed_s(series): return series[:,::-1].copy()

# K03 test-only swaps: need per-protein/replica test parts aligned
# (a) within protein+replica permutation of test rows: build permuted Hte_swapa aligned with concatenated test order
def swap_within(parts):
    outs=[]; r=np.random.default_rng(123)
    for p in parts:
        h=p["hist"].T; perm=r.permutation(len(h)); outs.append(h[perm])
    return np.concatenate(outs)
Hte_swap_pair=swap_within(test)
# (b) other replica same pair: test parts order is prot-major, replica 1,2,3. Map R1<-R2,R2<-R3,R3<-R1 by matching donor/acceptor atom ids
def swap_replica(parts):
    outs=[]
    for pi in range(0,len(parts),3):
        grp=parts[pi:pi+3]
        for ri in range(3):
            src=grp[(ri+1)%3]; dst=grp[ri]
            key_src={(int(a),int(b)):i for i,(a,b) in enumerate(zip(src["donor_atom"],src["acceptor_atom"]))}
            h=np.empty_like(dst["hist"].T); missing=0
            for i,(a,b) in enumerate(zip(dst["donor_atom"],dst["acceptor_atom"])):
                j=key_src.get((int(a),int(b)))
                if j is None: h[i]=dst["hist"].T[i]; missing+=1
                else: h[i]=src["hist"].T[j]
            print("swap_replica missing",missing,"of",len(h))
            outs.append(h)
    return np.concatenate(outs)
Hte_swap_rep=swap_replica(test)

CONDITIONS={}
# each condition: function returning (train_hist_feats, test_hist_feats) or None for static
CONDITIONS["static_only"]=None
CONDITIONS["base_hist_D6"]=(hist6(Htr),hist6(Hte))
CONDITIONS["K01_thin6"]=(hist6(thin(Htr,[0,2,4,6,8,10])),hist6(thin(Hte,[0,2,4,6,8,10])))
CONDITIONS["K01_thin3"]=(hist6(thin(Htr,[0,5,10])),hist6(thin(Hte,[0,5,10])))
CONDITIONS["K01_shuffle"]=(hist6(shuffled(Htr,7)),hist6(shuffled(Hte,7)))
CONDITIONS["K01_reverse"]=(hist6(reversed_s(Htr)),hist6(reversed_s(Hte)))
CONDITIONS["K02_gen8"]=(gen8(Htr,Xtr[:,0]),gen8(Hte,Xte[:,0]))
CONDITIONS["K03_swap_pair_testonly"]=(hist6(Htr),hist6(Hte_swap_pair))
CONDITIONS["K03_swap_replica_testonly"]=(hist6(Htr),hist6(Hte_swap_rep))
CONDITIONS["K04_resmin"]=(hist6(cat(train,"hist_min")),hist6(cat(test,"hist_min")))
CONDITIONS["K04_ca"]=(hist6(cat(train,"hist_ca")),hist6(cat(test,"hist_ca")))

results={}
for name,feats in CONDITIONS.items():
    results[name]={}
    for hi,h in enumerate(HORIZONS):
        ytr=Ytr[:,hi]; yte=Yte[:,hi]
        if feats is None:
            p=predict(Xtr,ytr,Xte)
        else:
            ftr,fte=feats
            p=predict(np.concatenate([Xtr,ftr],axis=1),ytr,np.concatenate([Xte,fte],axis=1))
        results[name][str(h)]=metrics(yte,p)
        print(name,h, {k:round(v,4) if isinstance(v,float) else v for k,v in results[name][str(h)].items()},flush=True)
Path("/home/hatch/workspace/kinescope/work/results.json").write_text(json.dumps(results,indent=2))
print("saved")
