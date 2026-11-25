# 03. 生成 MD・平衡アンサンブル生成モデルと評価・失敗モード

キネスコの知見 4（**「生成軌道が失敗するのは、ペア平均距離の約 97% が静的特徴で説明できてしまうため」**：K05, K11, K26, K34, K39, K43）の新規性と適用限界を評価するための先行研究一覧。

---

## 1. 主要な生成 MD / タンパク質アンサンブル生成モデル（2024–2025）

### AlphaFlow — Jing, Berger & Jaakkola (ICML 2024)
- **書誌**: B. Jing, B. Berger, T. Jaakkola, *"AlphaFold Meets Flow Matching for Generating Protein Ensembles"*, **ICML 2024**. GitHub: `bjing2016/alphaflow`
- **要点**: AlphaFold / ESMFold の単一構造予測器をフローマッチング枠組みへ転用し、MD アンサンブルを模倣する構造群を生成。

### MDGen — Jing, Stärk, Jaakkola & Berger (NeurIPS 2024)
- **書誌**: B. Jing, H. Stärk, T. Jaakkola, B. Berger, *"Generative Modeling of Molecular Dynamics Trajectories"*, **NeurIPS 2024**. GitHub: `bjing2016/mdgen`
- **要点**: MD 軌道を動画生成問題として扱い、**ATLAS データセット** を直接用いて前方シミュレーション・遷移経路サンプリング・時間アップサンプリング等を学習。

### Str2Str — Lu, Zhong, Zhang & Tang (ICLR 2024)
- **書誌**: J. Lu, B. Zhong, Z. Zhang, J. Tang, *"Str2Str: A Score-based Framework for Zero-shot Protein Conformation Sampling"*, **ICLR 2024**. `arXiv:2306.03117`
- **要点**: MD 軌道データを使わずにスコアベース構造間変換でゼロショットに構造アンサンブルをサンプリング。

### Distributional Graphormer (DiG) — Zheng et al. (Nature Machine Intelligence 2024)
- **書誌**: S. Zheng, J. He, C. Liu, Y. Shi, Z. Lu, W. Feng, F. Ju, J. Wang, J. Zhu, Y. Min, et al. (Microsoft Research AI4Science), *"Towards Predicting Equilibrium Distributions for Molecular Systems with Deep Learning"*, **Nature Machine Intelligence** 6, 195–208 (2024). DOI: `10.1038/s42256-024-00837-3`
- **要点**: 深層拡散／アニーリングネットワークにより、分子記述子から直接平衡分布をサンプリング。

### BioEmu — Lewis, Hempel, Jiménez-Luna, Gastegger, Clementi, Noé et al. (Science 2025)
- **書誌**: S. Lewis et al. (Frank Noé ほか, Microsoft Research AI4Science / FU Berlin), *"Scalable emulation of protein equilibrium ensembles with generative deep learning"*, **Science** (2025). DOI: `10.1126/science.adv9817`. GitHub: `microsoft/bioemu`
- **要点**: 静的構造・実験安定性データ・200 ms 超の MD 軌道を統合学習し、単一 GPU で 1 時間あたり数千の独立コンフォメーションを高速生成。クリプティックポケットや自由エネルギー差（~1 kcal/mol 精度）を再現。

### BBFlow (2024–2025) / BioMD (2025)
- **BBFlow**: ATLAS で学習・評価されたバックボーン幾何のみの SE(3) フローマッチング動態予測器。
- **BioMD** (Feng, Zhang, Zhang, Liu, Li): MISATO / DD-13M 上で階層的予測＋補間を行う全原子バイオ分子動態生成モデル。

---

## 2. 生成 MD の既知の失敗モードと評価指標

近年の評価研究（2024–2025）では、生成アンサンブル／生成軌道の主要な失敗モードとして以下がすでに広く議論されている：

1. **静的構造崩壊（Static Structure Collapse / Mode Collapse）**:
   - 過正則化や単一構造バイアス、自己回帰ステップの保守化により、生成軌道が入力の単一静的構造の周囲に張り付いてしまい、真の熱ゆらぎや機能的運動を生まない現象（Str2Str や自己回帰型生成器のトレードオフとして指摘）。
2. **ボルツマン重みの不一致（Boltzmann Distribution Mismatch）**:
   - 多様性は出ても物理的な自由エネルギー重みが合わず、高エネルギー状態を過大サンプリングしたり、点変異による平衡シフトを定量再現できない（BioEmu 等の独立ベンチマークでも指摘）。
3. **立体化学的破綻・誤差蓄積**:
   - 長時間ロールアウト時の原子衝突やペプチド結合歪み。
4. **既存の評価軸**:
   - 従来の評価は RMSD / RMSF、接触マップ、二次構造占有率、TICA / PCA 射影上の自由エネルギー地形、$J$-coupling / PRE などの実験観測量の一致度が中心。

---

## 3. キネスコとの照合

- **既知の部分**: 生成軌道が静的構造の近くに張り付いて動かなくなること（静的崩壊）自体は既に知られている。
- **キネスコの独自着想**: 「下流タスクで生成軌道が効かない理由」を、単なる多様性不足ではなく **『生成されたペア平均距離の静的特徴による決定係数 $R^2$（実 0.635 vs 生成 0.976）および静的残差のラベル相関ゼロ』** として直接診断した点（K39, K43）は、既存の評価指標（RMSF・TICA 等）の盲点を突く鋭い診断プロトコルである。
- **未加工での致命的限界**: キネスコが測定した生成器は **すべて MolCasta 内部の非公開・簡易生成器**（第 22 案＋第 2・8・11〜14 案）のみであり、上記の公開 SOTA 生成器（MDGen, BioEmu, AlphaFlow, Str2Str, DiG 等）を 1 つも評価していない。したがって未加工状態では「生成軌道一般の科学的主張」にはならず、公開生成器 1〜2 種への適用が不可欠である。
