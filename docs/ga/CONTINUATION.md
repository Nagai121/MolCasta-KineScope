# GA395〜GA1288 研究レポート

[全894試行の索引](CONTINUATION_INDEX.md) · [実測・保存物台帳](continuation-manifest.json) · [GA01〜GA392](README.md)

単一の静的構造を起点に生成した軌道から、同じ原子ペアの実履歴0〜1 nsの平均距離をどこまで推定できるかを調べた。観測履歴は教師と採点に用いた。比較指標は、予測平均・観測平均の両方から固定の静的参照を引いた値のPearson相関。下流の水素結合形成AUPRCや指導力回復率の合格判定とは評価対象が異なる。

## 主な結果

| 評価 | モデル | 残差相関 | 比較元 | 残差相関 |
| --- | --- | ---: | --- | ---: |
| 継続検証3配列・9起点 | GA403 | 0.617960 | GA392 | 0.615994 |
| 継続検証3配列・9起点 | GA696 | 0.629649 | GA452 | 0.619664 |
| 継続検証3配列・9起点 | GA1086 | 0.653346 | GA1005 | 0.650370 |
| GA351 既存の確認8配列・3,980ペア | GA1039 | 0.643909 | GA348 | 0.621945 |
| GA841 新規8配列・3,864ペア | GA821 | 0.660857 | GA348 | 0.652839 |
| GA1145 別の新規8配列・3,715ペア | GA1086 | 0.663925 | GA348 | 0.668885 |

新規の確認評価は配列・起点・候補集合を固定して各パネル内で比較した。GA841でのGA821の差は+0.008018、配列単位の対応付き再標本化区間は−0.013984〜+0.025594。別のGA1145ではGA1086の差は−0.004960。いずれも相関0.700は未達。GA1145では一配列のCHARMM末端ヒスチジンのテンプレートの曖昧さを採点前に解消し、登録した全8配列を維持した。 [GA841](../../experiments/GA395-GA1288_continuation/archive/runs/ga841-fresh-panel/confirmation.json) · [GA1145](../../experiments/GA395-GA1288_continuation/archive/runs/ga1145-prospective/confirmation.json) · [GA1145の復旧記録](../../experiments/GA395-GA1288_continuation/archive/runs/ga1145-prospective/charmm-setup-recovery.json)

## 改良の経緯

- **GA403・GA452。** AmberとCHARMMの平均距離応答を中心とした読み出しを比較した。GA452は後続起点を4倍に重み付けして学習し、固定検証では後続起点で改善した。GA351は0.623947。 [GA403](../../experiments/GA395-GA1288_continuation/archive/runs/ga395-444/README.md) · [GA452](../../experiments/GA395-GA1288_continuation/archive/runs/ga445-494/README.md)
- **GA696。** 履歴の複数時間窓・ばらつき・近距離滞在率を補助教師に加えた。GA351は0.631175。対応させた8初期値では補助教師の有無で4勝4敗、平均差は+0.002608。 [GA696](../../experiments/GA395-GA1288_continuation/archive/runs/ga688-multiwindow/ga351-confirmation.json) · [初期値比較](../../experiments/GA395-GA1288_continuation/archive/runs/ga789-window-seed-check/ensembles.json)
- **GA821・GA898。** 同一ペアの別起点や別レプリカで教師が揺れる場合に学習重みを下げた。さらに混合分布を使ったGA898は検証0.640523、GA351は0.633468。重み付けは対応8初期値中7つで同方向に改善した。 [GA898](../../experiments/GA395-GA1288_continuation/archive/runs/ga890-robust-mixture/ga351-confirmation.json) · [初期値比較](../../experiments/GA395-GA1288_continuation/archive/runs/ga929-mixture-seeds/seed-comparison.json)
- **GA1005・GA1039・GA1086。** 原子を共有する候補ペアの競合を特徴にし、GA1039以降は原子間の向きも反映した。方向の追加は、8初期値を平均した検証では0.633667から0.633399へ微減した。単独の最良値と初期値をまたいだ効果を区別する。 [競合モデル](../../experiments/GA395-GA1288_continuation/archive/runs/ga1001-competitive-capacity/ga351-confirmation.json) · [方向の対応比較](../../experiments/GA395-GA1288_continuation/archive/runs/ga1049-orientation-audit/seed-comparison.json)
- **学習集合と物理処理。** 同じ設定で74配列のGA898は0.640523、24配列の先頭起点を足したGA969は0.620175、同じ24配列の後続起点まで加えたGA1032は0.638708。GA1236の明示溶媒は先頭起点で改善したが、係数を固定した後続起点では0.651487から0.646808へ低下した。25・60・100 psの比較でも大きな改善はなかった。 [学習集合の比較](../../experiments/GA395-GA1288_continuation/archive/runs/ga1029-later-expansion/trials.jsonl) · [明示溶媒の後続起点](../../experiments/GA395-GA1288_continuation/archive/runs/ga1237-explicit-later/development.json) · [生成長の比較](../../experiments/GA395-GA1288_continuation/archive/runs/ga1176-long-readout/trials.jsonl)

## 成果物と再現

各バッチの実測、保存していた重み、評価出力、生成キャッシュ、実行ログをランディレクトリの相対パスを保って収録した。OSの一時PIDのファイルは除外した。1件の試行がバッチ内の比較、交差検証、固定モデルの再採点に対応する場合がある。索引の保存個別成果物数はそのID名を持つファイルの数で、バッチ共通の重みや予測は個別記録を参照。

コードは作業ツリーの再現用スナップショットで、各試行を実行した当時のコミットを証明するものではない。元のパッケージを保つため、InternalResearch側の実行環境で source の t2i と tseries を参照する。MiniTopolのATLAS環境、凍結T1重み、既存GAの入力キャッシュ、選んだ処理に対応するCUDA・OpenMMが必要。モデルの選択・数値の再計算は今回のアーカイブ作業では行っていない。

GA1289の146配列への拡張は別の作業として進行中。台帳はGA395〜GA1288に対応する。 [全試行索引](CONTINUATION_INDEX.md) · [台帳](continuation-manifest.json)
