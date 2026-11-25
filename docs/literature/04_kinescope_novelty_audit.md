# 04. キネスコ（MolCasta-KineScope）新規科学主張の厳密監査レポート

対象：`MolCasta-KineScope` の `main` ブランチ（529 実験：K / K2 / K3 / G / L 系列）およびリモートブランチ `research/ga301-static-pair-offset`（GA01〜GA1288）。

## 判定基準（無駄な保留を排するための 2 軸）

「未加工の状態で新規な科学的主張となせる」とは、**追加実験・パイプライン差し替えを一切行わず、リポジトリに現存する測定記録だけ**で以下 2 軸を同時に満たすことと定義する。

1. **内部妥当性・再現性（Internal Validity）**: 実験設計上の致命的欠陥（時間リーク／時間ギャップ、基準の歪み、極小サンプル $N=2$ タンパク質への過一般化、非公開モデルへの本質的依存）がないこと。
2. **先行研究に対する新規性（Novelty）**: [01](01_biophysics_hbond_and_energy_landscape.md)〜[03](03_generative_md_and_failure_modes.md) の既知理論・既知ベンチマークの結果を言い換えただけではないこと。

---

## 最終カウント

- **厳格な科学論文基準で未加工のまま成立する新規主張：0 件**
- **「非公開ベースモデル `T1I2006` への条件付け・強い土台でのレプリカ下限未測定」を許容した場合に、未加工データ自体が新規知見に最も近いもの：最大 1 件**（下記「候補 5：K3 系列の 100 タンパク質大規模棄却」）

---

## 全 6 主張候補の個別監査

### 候補 1：「わずか 100 ps (0.1 ns) の実 MD で 90 ns 先の指導力の半分以上を回収でき、コストは 1 ns の 1/8」（FINDINGS #1, G02〜G180, L01〜L74, G04）
- **判定**: **❌ 不成立（致命的な実験設計欠陥あり）**
- **根拠（リポジトリ内部検証）**:
  1. **1.0 ns 時間ギャップの罠**: `G02`, `G05`, `L02`〜`L74` すべてで、予測起点と静的特徴は **フレーム 10（$t = 1.0\text{ ns}$）** に固定されている一方、「0.1 ns 切り取り」は **フレーム `[0, 1]`（$t = 0.0, 0.1\text{ ns}$）** を使用している。実際に `L02`（1 ns 先）で **覆う長さ 0 ns（フレーム 0 単独）** が回復率 **0.353**、`L08`（90 ns 先）では **0.693** を記録している。すなわち、起点 $t=1.0\text{ ns}$ の構造と $t=0.0\text{ ns}$ の距離を同時入力しているため、**その入力を作るために物理的にはすでに 1.0 ns の実 MD を走らせる必要がある**。起点直前 0.1 ns（フレーム `[9, 10]`）や起点 $t = 0.1\text{ ns}$ への移動は全 529 実験で一度も測定されていない。
  2. **基準モデルの歪み**: 100 タンパク質強い土台（`G05`, `notes/A_real_trajectory_effect.md`）では、凍結本体 `T1I2006` 単独の pooled AUPRC が **0.4929** であるのに対し、8 タンパク質較正の静的補正モデルが **0.4591** へ −0.0338 劣化している。0.1 ns 切り取りの pooled AUPRC は **0.4974** であり、静的補正比では 54.5% 回収に見えるが、**凍結本体単独比の実上積みは +0.0045（実利得 +0.0364 の 12.4%）に過ぎない**。
  3. **コスト実測の限界**: `G04` は 1 タンパク質（`1mix_A`）・CPU のみ・平衡化 10 ps 短縮の測定であり、GPU 環境や標準 100 ps 平衡化での一般主張には不足。

---

### 候補 2：「生成軌道が失敗するのは、ペア平均距離の約 97% が静的特徴で説明できてしまうため（静的冗長性診断）」（FINDINGS #4, K05, K11, K26, K34, K39, K43）
- **判定**: **❌ 未加工では「生成軌道一般の科学主張」として不成立（診断法の着想のみ高価値）**
- **根拠**:
  1. 測定対象は MolCasta 内部の非公開・簡易生成器のみ（`K39` の 97% は第 22 案のみ、`K11`/`K26` も第 2・8・11〜14 案の 6 タンパク質検証のみ）。**BioEmu・AlphaFlow・MDGen・Str2Str・DiG などの公開標準モデルは 0 件**。
  2. 生成モデルが静的構造周辺に張り付く「静的構造崩壊（static structure collapse）」自体は先行研究で既知（[03](03_generative_md_and_failure_modes.md)）。
  3. 「ペア平均距離の静的説明率 $R^2$ と残差ラベル相関」で下流失敗を診断する枠組みは鋭いが、公開生成器への実証がない未加工状態では「内部生成器の事後解析」に留まる。

---

### 候補 3：「水素結合の指導力は平均距離 1 スカラーに圧縮され、接触・塩橋では分散する／時間順不変・Cα 粗視化で崩壊」（FINDINGS #2・#6, K01, K04, K18, K21, K27, K31, K56, K75）
- **判定**: **❌ 未加工では独立した新規科学主張として不成立**
- **根拠**:
  1. **比較条件の非対称性**: 水素結合は 100 タンパク質＋凍結非公開モデル `T1I2006`（`K21`, `K2-18`）だが、接触は **テスト 2 タンパク質**（`1mix_A`, `2qt4_A`）＋弱い静的 5 特徴（`K27`）、塩橋は 30 タンパク質＋弱い静的 5 特徴（`K31`）であり、現象間の構造差を一般法則として主張する比較条件が揃っていない。
  2. `K01`（時間順シャッフル・逆転不変）と `K04`（Cα 距離への粗視化で消失）もテスト 2 タンパク質のみ。かつ `K01` の 6 統計量中 4 つは定義上もともと順序不変。
  3. 角度条件を含まない距離代理ラベル（≤ 3.5 Å）で時間平均距離が将来形成確率のほぼ十分統計量になる点は、局所ポテンシャル井戸の平均位置が占有率を決める統計力学の基本帰結と整合的で、単独での新規性は低い。

---

### 候補 4：「瞬時距離は 100 ps 以内で忘却されるが、静的予測からの『ずれ（offset）』は窓の前半・後半および別レプリカ間で持続し、指導力はトレンドではなく水準である」（FINDINGS #3・#5, K03, K23, K39, K42, K44, K2-02, K2-08〜K2-60, K2-147〜182）
- **判定**: **❌ 未加工では証拠の土台が分断されており、物理像の核心は既存理論と重複**
- **根拠**:
  1. **サンプル規模の分断**: 「残差が指導力の 68〜94% を担う（`K42`/`K44`）」は 100 タンパク質（`T1I2006` 依存）だが、**自己相関時間 < 100 ps（`K2-02`）、残差の前半・後半相関 r = +0.703（`K2-08`）、別レプリカ入替（`K03`）、レプリカ間下限 0.249 Å（`K23`）はすべてテスト 2 タンパク質（資料 D 条件）のみ**（塩橋 `K2-22`/`K2-60` のみ 30 タンパク質）。100 タンパク質強い土台での一貫計測が存在しない。
  2. **同一初期構造の交絡**: `K03`・`K23` の 3 レプリカは同一初期構造から出発した 0〜1 ns であり、独立初期構造からの検証ではない（`K03` 自身も留保）。
  3. **先行研究との重複**: 数十 ps での水素結合内揺らぎの脱相関（Luzar & Chandler 1996; Sheu et al. 2003）と、ns 以上持続するコンフォメーション・サブステート平均位置（Frauenfelder et al. 1991）の分離自体は生物物理の標準像である。

---

### 候補 5：「未見 100 タンパク質（163,074 ペア、ATLAS CHARMM36m）において、将来 1〜90 ns の水素結合形成を予測する 0〜1 ns 平均距離の静的残差（SD = 0.6002 Å）は、起点構造の 5 大物理量（側鎖 χ 角・主鎖 φ/ψ 角・重原子局所電場・ペア間橋渡し水・CHARMM36m 静電/vdW エネルギー）のいずれともほとんど無相関（|r| ≤ 0.143）で説明不能」（K3-01, K3-03, K3-04, K3-05, K3-06 ＋ K39/K42）
- **判定**: **⚠️ 未加工データの中で新規知見に最も近い（条件付き最大 1 件）が、厳格基準ではあと 2 点の穴が残る**
- **強い点**:
  - 未見 100 タンパク質 × 3 レプリカ（163,074 ペア、配列一致度 < 0.35）という大規模かつ厳密な検証。
  - 事前登録ライン（|r| ≥ 0.30, SD ≤ 0.30 Å）、シャッフル対照、距離一致誤差 < 2×10⁻⁶ Å、`.tpr` による CHARMM36m＋NC2–OC NBFIX 実力場検証、フレーム 0・フレーム 10 両計測まで極めて丁寧。
  - Fried & Boxer (2017) の局所電場、Bellissent-Funel (2016) / Halle (2004) の橋渡し水、古典力場エネルギー、二面角という真っ先に疑われる 5 大物理仮説を一斉に 100 タンパク質で定量棄却した大規模ネガティブ結果は先行文献にほぼ見当たらない。
- **未加工のまま主張にする際の 2 つの穴**:
  1. **非公開モデル `T1I2006` 依存**: 残差の定義が「非公開 `T1I2006` の出力確率を含む 5 特徴からの予測残差」であり、第三者が同一残差を独立再現できない（かつ `T1I2006` 自体が 118+ 個の構造特徴を含むため、「単一構造全般で説明不能」ではなく「`T1I2006` 抽出後の残差に追加で効かない」という条件付き主張になる）。
  2. **強い土台でのレプリカ間下限が未測定**: 0.6002 Å の残差 SD のうち、何割が真に再現可能な物理信号（レプリカ共通成分）で、何割が有限 11 フレーム標本ノイズ・起点瞬時ズレなのかが 100 タンパク質側では測られていない（`K3` 各 README 訂正欄にも明記）。さらに `K3-02`（周囲水素結合ネットワーク）も未完。

---

### 候補 6（リモートブランチ `research/ga301-static-pair-offset`）：「同一の静的起点構造からの独立 100 ps 暗黙溶媒 MD（Amber/CHARMM 8 分岐）は、純粋静的 ML（300 構造探索・静的力/幾何）を大きく超えて過去 0〜1 ns 実 MD 平均残差をブラインドパネルで r = 0.62–0.67 予測するが、約 0.66–0.67 で頭打ちになる」（GA01〜GA1288）
- **判定**: **❌ 未加工では独立した科学主張として未統合**
- **根拠**:
  - **重要発見**: `GA305`〜`GA1145` は G/L 系列と異なり、**真に単一の静的起点構造から独立に前方 100 ps MD を回している**（時間ギャップの罠がない）。これにより、「単一スナップショットの静的物理記述子（K3）では取れないペア平均オフセットが、力場ポテンシャル上の短期多体緩和（100 ps MD）では r = 0.62–0.67 回収できる」という物理的解釈（＝オフセットの正体は局所自由エネルギー井戸への多体緩和平衡位置）が強く示唆される。
  - しかし未加工では、①評価が 8 配列（3,700〜4,000 ペア）かつ非公開 `T1I2007` 依存、②下流 100 配列での AUPRC 回復率が未測定、③`main` ブランチの K3/K42 と基準・データ分割が統合されていない。

---

## 最小加工で本物の新規科学主張（論文級）に化ける 3 本のルート

未加工では「0 件（緩めに見ても条件付き 1 件）」だが、キネスコの蓄積から **最小限の追試だけで強い論文・知見になるタネが 3 本** はっきり見えている：

1. **ルート A（最有力・K3＋GA305 の統合）**:
   - **主張**: 「タンパク質内水素結合ペアの 0〜1 ns 平均距離オフセット（1〜90 ns 先の形成を支配）は、単一スナップショットの 5 大物理量（χ、φ/ψ、局所電場、橋渡し水、CHARMM36m 静電/vdW エネルギー）では説明不能だが、同一静的構造からの極短多体緩和（100 ps 暗黙溶媒 MD）で r ≈ 0.62–0.67 回収できる」
   - **最小加工**: ① `T1I2006` を使わない公開静的ベースライン（起点距離＋公開構造特徴）で 100 タンパク質の残差と **レプリカ間下限** を計算、② `K3-02` を完了、③ 同一 100 タンパク質（または公開分割）上で K3 の棄却結果と GA305 型の短期緩和回収を 1 本の表に揃える。
2. **ルート B（生成モデル評価法・知見 4 の外部実証）**:
   - **主張**: 「生成 MD モデルが下流イベント予測で効かない原因を診断する『ペア平均距離の静的冗長性 $R^2$・残差ラベル相関テスト』」
   - **最小加工**: 公開モデル（**MDGen** や **BioEmu** / **AlphaFlow** のうち 1〜2 種）の ATLAS 出力に対して `K39`/`K43` と同じ診断スクリプトを 1 回走らせるだけ。
3. **ルート C（極短実 MD の正しい検証・G/L 設計ミスの修正）**:
   - **主張**: 「本当に起点直前の 0.1 ns（フレーム `[9, 10]`）または起点 $t = 0.1\text{ ns}$ での 100 ps 実 MD が 1〜90 ns 先をどれだけ予測できるか」
   - **最小加工**: 1.0 ns 時間ギャップを解消し、公開静的ベースラインに対して 100 タンパク質で再計測する。


---

## Addendum (Post-Verification Update: E01–E04 Completed)

Following the initial audit of unprocessed KineScope claims (**0 publication-grade claims as-is**), four 100% public, leak-free verification experiments (`E01–E04`) were executed in `d:\KineScope\experiments\` without touching `MolCasta-KineScope/`. The verified count of **publication-grade, 100% public scientific claims** now moves from **0 → 3 strong claims** (plus 2 cleanly retired artifacts):

### Newly Established Public Claims (3)
1. **Static-to-Equilibrium Wall & 5-Family Elimination ([E02](../../experiments/E02_public_k3_and_replica_floor/README.md))**: Across 100 unseen ATLAS proteins (`484,049` pairs, 4 phenomena, 0 proprietary features), independent MD replicas share a reproducible sub-ns pair-mean residual (`r = +0.388` HB form, `+0.568` HB diss, `+0.416` salt bridge; `2.8×` above the measured 100-protein single-replica thermal noise floor `0.201 Å`), yet **all 5 static physical-feature families** — including simultaneous combination (`68` features) and newly completed `K3-02` H-bond network topology — explain `<10%` of its variance.
2. **Water-Free Implicit-Solvent Relaxation Shatters the Static Wall & Recovers 1–90 ns Guidance ([E03](../../experiments/E03_implicit_md_relaxation_vs_k3/README.md))**: From single static frame 10 (`d0_err = 0.0000 Å`), `4×100 ps` OpenMM GBn2 implicit-solvent MD (no explicit water, `55 s/protein` on consumer GPU) recovers `r = +0.520` of the explicit-solvent residual (`SD: 0.603 → 0.513 Å`) and **55–86% (mean-only) / up to 116% (all 7 stats at 90 ns)** of explicit 1 ns MD's downstream 1–90 ns H-bond formation guidance.
3. **Two-Regime Benchmark: Implicit Physics vs SOTA Generative MD (`MDGen`) & True Short-MD Scaling ([E01](../../experiments/E01_true_short_md_guidance/README.md) + [E04](../../experiments/E04_public_generator_static_redundancy/README.md))**:
   - True short explicit MD exhibits a sharp **initial-relaxation vs steady-state asymmetry** (`0.0→0.1 ns` recovers `48–60%`, whereas steady-state `0.9→1.0 ns` recovers only `36–44%`; `0.5 ns` reaches `75–88%`; `1.0 ns mean` alone captures `90–96%`), while real GPU fixed overhead (`200 ps eq`) caps `0.1/1.0 ns` cost at `0.27–0.28` (`~1/3.6`, not `1/8`).
   - On 25 strictly unseen `atlas_test.csv` proteins, official `MDGen` (`40 s/protein`) is **not** statically redundant ($R^2 = 0.59 \to 0.49$), contradicting KineScope Finding 4's generalization, yet exposes a clean **physics-vs-ML two-regime split**: implicit physics (`E03`) dominates sub-ns local offset (`r = +0.520`, `1 ns rec = 58%`), whereas MDGen's 100 ns rollout is weak at `1 ns` (`-5% to +25%`, `r = +0.266`) while recovering **`55–78%`** at `5–90 ns`.

### Cleanly Retired KineScope Artifacts (2)
- **Retired**: "0.1 ns steady-state MD recovers ≥50% at all horizons at 1/8 cost" (proven to be a 1.0 ns time-gap artifact + CPU-only 20 ps eq benchmark in E01).
- **Retired**: "Generative MD models suffer universal $R^2 \approx 0.96$ static redundancy and 0% guidance" (proven specific to KineScope's internal generator; refuted on real SOTA `MDGen` in E04).
