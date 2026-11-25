# KineScope（キネスコ）先行研究文献アーカイブ & 新規性監査

キネスコ（[MolCasta-KineScope](../MolCasta-KineScope/README.md)）の主張候補が「未加工の状態で新規な科学的主張となり得るか」を判定するために収集・整理した先行研究群と、リポジトリ内部監査レポートを格納しています。

## ファイル一覧

| ファイル | 内容 |
|---|---|
| [01_biophysics_hbond_and_energy_landscape.md](01_biophysics_hbond_and_energy_landscape.md) | 生物物理・統計力学の基礎文献（エネルギー地形、水素結合寿命・自己相関、水和・橋渡し水、タンパク質内部電場、MD レプリカ分散） |
| [02_md_datasets_and_static_dynamics_prediction.md](02_md_datasets_and_static_dynamics_prediction.md) | 大規模 MD データセット（ATLAS, mdCATH, MISATO）、静的構造からの動態予測、短期 MD からの長時間動態・キネティクス学習（VAMPnets, SPIB, DeepJump 等） |
| [03_generative_md_and_failure_modes.md](03_generative_md_and_failure_modes.md) | 生成 MD・平衡アンサンブル生成モデル（BioEmu, AlphaFlow, MDGen, Str2Str, DiG, BBFlow, BioMD）と評価指標・失敗モード（静的構造崩壊など） |
| [04_kinescope_novelty_audit.md](04_kinescope_novelty_audit.md) | キネスコ全主張候補（`main`＋`research/ga301-static-pair-offset`）の未加工状態での内部妥当性×先行研究照合・最終判定 |

## 結論サマリ

- **未加工の状態で、査読耐性のある独立した新規科学主張として成立するもの：0 件**
- **「非公開ベースモデル依存・レプリカ下限未測定」を許容した場合に新規知見に最も近いもの：最大 1 件**（K3 系列の 100 タンパク質・163,074 ペアにおける 5 大物理量の網羅的棄却）
- **最小限の追加実験・公開基準への載せ替えで新規科学主張へ化けるタネ：3 件**（詳細は [04_kinescope_novelty_audit.md](04_kinescope_novelty_audit.md)）
