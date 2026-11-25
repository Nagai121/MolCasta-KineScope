# 02. 大規模 MD データセット・静的構造からの動態予測・短期 MD からの長時間動態学習

キネスコの実験基盤（ATLAS）、静的構造からの平均距離・柔軟性予測の上限（K23, K40, G01, GA 系列）、および短期 MD からの長時間予測（G 系列・L 系列・K56/K75）を先行研究と照合するための文献一覧。

---

## 1. 標準化された大規模 MD データセット

### ATLAS — Vander Meersche et al. (2024)
- **書誌**: Y. Vander Meersche, G. Cretin, A. Gheeraert, J.-C. Gelly, T. Galochkina, *"ATLAS: protein flexibility description from atomistic molecular dynamics simulations"*, **Nucleic Acids Research** 52(D1), D384–D392 (2024). DOI: `10.1093/nar/gkad1084`
- **要点**:
  - 非冗長な単鎖タンパク質群（1,390 鎖規模）に対し、GROMACS + CHARMM36m + TIP3P + 150 mM NaCl で各 100 ns × 3 レプリカの標準化全原子 MD を提供（CC BY-NC 4.0）。
  - キネスコの全実験の元データであり、現在のタンパク質動態 ML の事実上の標準ベンチマーク。

### mdCATH — Mirarchi, Giorgino & De Fabritiis (2024)
- **書誌**: A. Mirarchi, T. Giorgino, G. De Fabritiis, *"mdCATH: A Large-Scale MD Dataset for Data-Driven Computational Biophysics"*, **Scientific Data** / arXiv (2024).
- **要点**:
  - CATH ドメイン 5,398 種 × 5 温度（320–450 K）× 5 レプリカ、総計 62 ms 超の全原子 MD 座標・力データセット。

### MISATO — Siebenmorgen et al. (2024)
- **書誌**: C. Siebenmorgen et al., *"MISATO: machine learning dataset of protein–ligand complexes for structure-based drug discovery"*, **Nature Computational Science** (2024).
- **要点**:
  - PDBbind 由来約 20,000 のタンパク質–リガンド複合体に対する半経験的 QM 精緻化＋明示溶媒 MD（総計 170 μs 超）。

---

## 2. 短期 MD 軌道から長時間キネティクス・遅い集団変数を抽出する ML

### VAMPnets — Mardt, Pasquali, Wu & Noé (2018)
- **書誌**: A. Mardt, L. Pasquali, H. Wu, F. Noé, *"VAMPnets for deep learning of molecular kinetics"*, **Nature Communications** 9, 5 (2018).
- **要点**:
  - マルコフ過程の変分アプローチ（VAMP）をエンドツーエンドの深層学習に統合し、多数の短い MD 軌道断片から長時間スケールの遅いモード・マルコフ状態モデル（MSM）を直接学習する基盤手法。

### SPIB — Wang & Tiwary (2021)
- **書誌**: D. Wang & P. Tiwary, *"State predictive information bottleneck"*, **J. Chem. Phys.** 154, 134111 (2021)（arXiv 2020; 発展版 2021–2024）.
- **要点**:
  - 情報ボトルネック原理を用い、高次元座標から「時間遅れ $\Delta t$ 先の準安定状態」を予測するための最小十分表現（反応座標 / CV）を抽出。コミッター関数と理論的に結びつき、未収束の短期軌道からのキネティクス推定に広く用いられる。

### 短期 MD からの大時間ステップ跳躍・長時間動態予測（2025–2026）
- **DeepJump (2025)**: ユークリッド同変フローマッチングを用い、mdCATH 等の短期軌道断片からマクロ時間ステップを跳躍して長時間動態を再現（~1,000× 高速化）。
- **TS-DAR (2025)**: 短期軌道から遷移状態・中間状態・反応座標・コミッターを自動抽出する変分正則化ニューラルネットワーク。
- **VelocityFM (2026)**: 残基フレームと二面角の速度空間での Rectified Flow Matching により、短期軌道予測を未見タンパク質へ汎化。
- **Dyna-1 (2026)**: μs–ms スケールのタンパク質動態（NMR 緩和データ対応）を高速予測。
- **Temporal GNN / 4D 記述子による結合予測 (2023–2025)**: 短期 MD から抽出した動的特徴を XGBoost や GNN に組み込んでクリプティックポケットや結合親和性・キネティクスを改善するパイプラインは既に多数報告されている。

- **キネスコとの照合**:
  - 「短い実 MD を ML に追加入力して将来の分子イベント予測を改善する」という大枠のアイデア自体は、VAMPnets / SPIB / Temporal GNN / 4D-MD-ML の流れで既に確立済み。
  - キネスコ固有の主張となり得るはずだった「わずか 100 ps で 90 ns 先の 50% 以上を回収・コスト 1/8」は、内部監査で **予測起点（1.0 ns）と入力（0.0–0.1 ns）の間に 1.0 ns の時間ギャップがある** ことが判明したため、未加工のままでは成立しない（詳細は [04_kinescope_novelty_audit.md](04_kinescope_novelty_audit.md)）。

---

## 3. 単一静的構造からの動態・柔軟性（RMSF / 接触 / 水素結合）予測

### RMSF-net (2024) / FastProtFlex (2025)
- **RMSF-net (2024)**: 静的原子構造や Cryo-EM 密度から残基レベル RMSF を予測し、古典的弾性ネットワークモデル（ENM）を上回る。
- **FastProtFlex / GDV (2025)**: ATLAS サブセットで学習した 15 次元局所パッキング特徴（Graphlet Degree Vector）の線形モデルだけで、静的座標から RMSF を秒単位で高精度予測。
- **キネスコとの照合**:
  - 残基レベルの柔軟性（RMSF）は静的パッキング幾何だけでかなり高精度に予測できることが知られている一方、キネスコ `K04`・`K23`・`K40`・`G01`・`K2-03`・`K3-01..06` は **「原子ペアレベルの 0〜1 ns 平均距離の静的残差（ずれ）」は Cα 粗視化で完全に消え、残基パッキング・Cα RMSF・99 静的特徴・点群 NN・5 大物理記述子（χ, φ/ψ, 電場, 橋渡し水, CHARMM36m エネルギー）のどれでも説明できない** ことを示している。ここは残基柔軟性予測の既存文献とは明確に異なる現象領域を突いている。
