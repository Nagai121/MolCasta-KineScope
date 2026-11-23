"""K25: 環境の履歴特徴量の抽出。

定義は結果を見る前に固定した（README 参照）。
- 残基ペア距離系列 S[i,j,t]: 履歴フレーム 0..10 の重原子間最小距離（周期境界は軸ごとの箱）。
- 近傍集合 N(r): 起点フレーム（フレーム10）で Cα 間距離が 10.0 Å 以下、配列距離2以上の残基。
- 候補（ドナー残基 d、アクセプター残基 a）について、r in {d, a}、相手 p として
  E(r) = N(r) から p を除いた集合に対し、S[r,k,t]（k in E(r), 全11フレーム）から
  env_dmean, env_dmin, env_dfrac(<=4.5), 接触数系列 C_r(t)=#{k: S<=4.5} の
  env_cmean, env_cstd, env_cdiff(最後-最初) の6量を作る。ドナー側6＋アクセプター側6＝12特徴量。
- E(r) が空の場合、距離系3量は NaN、接触数系3量は 0。
候補の同定（donor_res / acceptor_res）は base_reproduction のキャッシュと同一。
"""
import sys
sys.path.insert(0, "/home/hatch/workspace/minitopol-read-opt/src")
import minitopol  # noqa: F401  (zarr codec 登録のため)
import numpy as np, pandas as pd, zarr
from pathlib import Path

BASE = Path("/home/hatch/workspace/gpt-model-check/data/trajectories")
CACHE = Path("/home/hatch/workspace/kinescope/work/data_cache")
OUT = Path("/home/hatch/workspace/kinescope/work/k25_env")
PROTEINS = ("1bgf_A", "1ifg_A", "1j8r_A", "2b0a_A", "1mix_A", "2qt4_A")
FRAMES = list(range(11))
CA_CUTOFF = 10.0
CONTACT = 4.5


def distance(v, box):
    box = np.asarray(box, dtype=np.float64)
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-5):
        lengths = np.diag(box)
        v = v - lengths * np.rint(v / lengths)
    else:
        inv = np.linalg.inv(box)
        fr = v @ inv
        fr -= np.rint(fr)
        v = fr @ box
    return np.linalg.norm(v, axis=-1)


def layout(df):
    # base_reproduction/extract.py と同一の残基並び（N,CA,C,O を欠く残基は除外）
    order = list(dict.fromkeys(df["residue_index"].tolist()))
    maps = []
    for r in order:
        idx = np.flatnonzero(df["residue_index"].values == r)
        nm = {df["name"].values[i]: int(i) for i in idx}
        if not {"N", "CA", "C", "O"}.issubset(nm):
            continue
        maps.append(nm)
    return maps


def residue_pair_series(pos, box, heavy, res_of_atom, n_res):
    """S[i,j,t]: 履歴フレームの残基ペア重原子最小距離。(n_res, n_res, 11)"""
    n_atoms = pos.shape[1]
    S = np.empty((n_res, n_res, len(FRAMES)), dtype=np.float32)
    for ti, f in enumerate(FRAMES):
        p = pos[f].astype(np.float64)
        best = np.full(n_res * n_res, np.inf, dtype=np.float64)
        step = 256
        for s in range(0, n_atoms, step):
            e = min(s + step, n_atoms)
            d = distance(p[s:e, None, :] - p[None, :, :], box[f])
            # 重原子同士だけを使う: 非重原子は inf にする
            d[~heavy[s:e], :] = np.inf
            d[:, ~heavy] = np.inf
            idx = (res_of_atom[s:e, None] * n_res + res_of_atom[None, :]).ravel()
            np.minimum.at(best, idx, d.ravel())
        S[:, :, ti] = best.reshape(n_res, n_res).astype(np.float32)
    return S


def env_features(S, ca_d10, d_res, a_res):
    n_res = S.shape[0]
    sep = np.abs(np.subtract.outer(np.arange(n_res), np.arange(n_res)))
    neigh = (ca_d10 <= CA_CUTOFF) & (sep >= 2)
    np.fill_diagonal(neigh, False)
    n_cand = len(d_res)
    env = np.full((n_cand, 12), np.nan, dtype=np.float64)
    nsize = np.zeros((n_cand, 2), dtype=int)
    for ci in range(n_cand):
        for side, (r, p) in enumerate(((d_res[ci], a_res[ci]), (a_res[ci], d_res[ci]))):
            ks = np.flatnonzero(neigh[r])
            ks = ks[ks != p]
            nsize[ci, side] = len(ks)
            base = side * 6
            if len(ks) == 0:
                env[ci, base + 3:base + 6] = 0.0
                continue
            dists = S[r, ks, :]  # (n_k, 11)
            env[ci, base + 0] = dists.mean()
            env[ci, base + 1] = dists.min()
            env[ci, base + 2] = (dists <= CONTACT).mean()
            cnt = (dists <= CONTACT).sum(axis=0)  # (11,)
            env[ci, base + 3] = cnt.mean()
            env[ci, base + 4] = cnt.std()
            env[ci, base + 5] = cnt[-1] - cnt[0]
    return env.astype(np.float32), nsize


def build(prot, rep):
    tid = f"atlas-{prot}-analysis-R{rep}"
    d = BASE / tid
    df = pd.read_parquet(d / "atoms.parquet")
    z = zarr.open_group(str(d / "A.zarr"), mode='r')
    pos = z['positions'][:].astype(np.float64)
    box = z['box'][:].astype(np.float64)
    maps = layout(df)
    n_res = len(maps)
    ca = np.array([m["CA"] for m in maps])
    heavy_lists = []
    for m in maps:
        heavy_lists.append(np.array([v for k, v in m.items()
                                     if not k.startswith("H") and not k[0].isdigit()], dtype=int))
    res_of_atom = np.full(pos.shape[1], -1, dtype=int)
    heavy = np.zeros(pos.shape[1], dtype=bool)
    for ri, hl in enumerate(heavy_lists):
        res_of_atom[hl] = ri
        heavy[hl] = True
    # レイアウト外の原子（重原子リストに無い原子）は距離計算から除外される（heavy=False）
    S = residue_pair_series(pos, box, heavy, res_of_atom, n_res)
    t = 10
    ca_d10 = distance(pos[t, ca[:, None], :] - pos[t, ca[None, :], :], box[t])
    cz = np.load(CACHE / f"{prot}_R{rep}.npz")
    env, nsize = env_features(S, ca_d10, cz["donor_res"], cz["acceptor_res"])
    np.savez_compressed(OUT / f"{prot}_R{rep}.npz", env=env, neigh_size=nsize)
    print(tid, "res", n_res, "cand", len(env),
          "neigh mean", nsize.mean(axis=0).round(2),
          "empty sides", int((nsize == 0).sum()), flush=True)


for p in PROTEINS:
    for r in (1, 2, 3):
        build(p, r)
print("done")
