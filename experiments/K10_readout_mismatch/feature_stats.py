"""K10 補助診断: hist6 特徴量の分布のずれ (実5点 vs 生成5点, 学習+テスト全件)
モデル学習は行わない。run_experiments.py と同じ読み込み・同じ hist6 定義。"""
import numpy as np, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
# reuse definitions without running main: copy minimal loaders
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
GEN = Path(__file__).resolve().parent.parent / "K05_generated_quality" / "generated_series.npz"
z = np.load(GEN); tids, pair, gen = z["tid"], z["pair"], z["gen"]
gen5_all = np.column_stack([pair[:,2].astype(float), gen.astype(float)])
GMAP = {(str(t), int(pair[i,0]), int(pair[i,1])): gen5_all[i] for i,t in enumerate(tids)}
def hist6(s):
    return np.stack([s[:,0], s.mean(1), s.std(1), s.min(1), s[:,-1]-s[:,0], (s<=3.5).mean(1)], axis=1)
names = ["first","mean","std","min","last_minus_first","frac_le_3.5"]
R=[]; G=[]
for p in ("1bgf_A","1ifg_A","1j8r_A","2b0a_A","1mix_A","2qt4_A"):
    for r in (1,2,3):
        d=np.load(CACHE/f"{p}_R{r}.npz"); h=d["hist"].T
        R.append(hist6(h[:,[0,3,5,7,10]]))
        tid=f"atlas-{p}-analysis-R{r}"
        G.append(hist6(np.stack([GMAP[(tid,int(a),int(b))] for a,b in zip(d["donor_atom"],d["acceptor_atom"])])))
R=np.concatenate(R); G=np.concatenate(G)
out={}
for i,n in enumerate(names):
    out[n]={"real5_mean":float(R[:,i].mean()),"real5_std":float(R[:,i].std()),
            "gen5_mean":float(G[:,i].mean()),"gen5_std":float(G[:,i].std()),
            "corr":float(np.corrcoef(R[:,i],G[:,i])[0,1])}
    print(n, {k:round(v,4) for k,v in out[n].items()})
Path(__file__).parent.joinpath("feature_stats.json").write_text(json.dumps(out,indent=2))
