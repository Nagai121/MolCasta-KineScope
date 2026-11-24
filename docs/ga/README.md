# GA01〜GA392 研究レポート

[全試行索引](INDEX.md) · [成果物とSHA-256の台帳](manifest.json)

静的構造から、実軌道が運ぶ指導力をどこまで再現できるかを調べた。初期300試行の最良方式GA259は、検証で凍結T1I2007のAUPRCを+0.000550改善した。選定後の4配列ではAUPRCが−0.000453だった。その後の平均距離残差の探索では、静的起点から短いMDを生成するGA305で相関0.500を超え、GA386は繰り返し参照した8配列の先頭起点で0.694461に達した。別の予約8配列では0.610223、従来8配列の後続起点では0.626572だった。相関0.700、誤差SD 0.30 Å以下、未見100配列×3レプリカでの指導力50%回復は、この範囲の記録から確認できていない。

## GA01〜GA300 · 下流の水素結合形成予測

12入力表現、5種類の生成目標、5構造で300通りを比較した。静的な仮想迂回で競合する候補の情報を入力し、エキスパート混合で平均・ばらつき・滞在割合などを出力するGA259を検証AUPRCで選定した。予測対象は1 ns先の水素結合形成の距離代理ラベル。候補集合と凍結T1I2007の確率を共通にした。[条件と観測履歴対照](../../experiments/GA01-GA300_static_guidance_screen/README.md)

| 評価集合 | モデル | AUC | AUPRC | Brier | Log Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| 検証10配列・153,212候補 | 凍結T1I2007 | 0.932855 | 0.500030 | 0.043723 | 0.144245 |
| 同上 | GA259 | 0.932870 | 0.500580 | 0.043710 | 0.144201 |
| 選定後4配列 | 凍結T1I2007 | 0.928058 | 0.470244 | 0.041071 | 0.138314 |
| 同上 | GA259 | 0.928012 | 0.469791 | 0.041057 | 0.138292 |

300試行のうち93試行で検証AUPRCが上がった。補正器の学習条件を合わせた追加対照の観測履歴AUPRCは0.514223。GA259の上積みはこの対照の約3.9%に相当するが、検証選定後の値であり、未見100配列×3レプリカの判定とは条件が異なる。[全試行の実測](../../experiments/GA01-GA300_static_guidance_screen/results.json) · [選定後評価](../../experiments/GA01-GA300_static_guidance_screen/test.json)

## GA301〜GA392 · ペアごとの平均距離残差

KineScopeで実履歴の平均距離に着目した結果を受け、同一ペアの過去0〜1 nsの平均距離を予測対象にした。報告するPearson相関は、予測平均と観測平均の双方から固定した静的対照の値を引き、対応する候補ペアの残差を比較したもの。誤差SDは予測平均と観測平均の差のばらつきで、下流AUPRCによる指導力の回復率とも別の指標である。生成MDを使う試行では静的起点からの生成結果を入力し、観測履歴は教師や採点に使用した。観測側を入力する診断は各記録で条件を確認できる。[GA301の定義と分割](../../experiments/GA301_static_pair_offset/README.md) · [KineScopeの測定事実](../facts.md)

| モデル | マイルストーンの特徴 | 残差Pearson相関と評価条件 |
| --- | --- | --- |
| [GA301](../../experiments/GA301_static_pair_offset/README.md) | 静的Ridge対照に対し、構造・仮想変形を用いる残差回帰 | 0.33278、従来8配列・9,843ペア。誤差SD 0.504069 Å |
| [GA302](../../experiments/GA302_local_relations/README.md) | MiniCalcの局所座標に候補周辺の関係を配置 | 静的関係表現の単独試行。条件別の成績は個別記録 |
| [GA304](../../experiments/GA304_static_force_geometry/README.md) | 静的な力と幾何を拡張 | 0.432747、従来確認パネル |
| [GA305](../../experiments/GA305_generated_native_pair_offset/README.md) | 静的起点からAmberで100 ps×2、生成統計から静的成分を引いて回帰 | 0.550534、従来8配列・9,843ペア。誤差SD 0.446856 Å |
| [GA309](../../experiments/GA309_static_force_native_blend/README.md) | GA304の静的残差とGA305の生成残差を加法融合 | 0.580911、GA305と同じ9,843ペア。誤差SD 0.435609 Å |
| [GA312](../../experiments/GA312_eight_seed_confirmation/README.md) | 100 psの生成を8分岐に拡大 | 0.637262、従来8配列・先頭3,586ペア。同一起点のGA305は0.576715 |
| [GA321](../../experiments/GA321_scaled_mixed_force_field_confirmation/README.md) | 二つの力場と拡張した静的予測を融合 | 従来先頭0.674013、予約8配列の先頭4,167ペアで0.594380 |
| [GA332](../../experiments/GA332_expanded_prior_confirmation/README.md) | 211配列へ拡張した静的予測を混合し、動的出力はGA321と共通 | 従来先頭0.675531、予約先頭0.596249 |
| [GA348](../../experiments/GA348_two_force_field_ensemble_confirmation/README.md) | 短い二力場生成の学習済み読み出しを20%混合 | 従来先頭0.685008、予約先頭0.596732、従来後続3,550ペアで0.630489 |
| [GA384](../../experiments/GA384_neural_response_confirmation/README.md) | 二力場出力にHuber学習のニューラル読み出しを20%混合 | 従来先頭0.688395、予約先頭0.604937、従来後続0.627662 |
| [GA351固定比較](../../experiments/GA351_sequence_disjoint_reserve/README.md) | 前もって予約した学習配列と重複しない8配列でGA384と親GA348を採点 | 3,980ペアで0.622139対0.621945。RMSE 0.500598 Å対0.499672 Å |
| [GA386](../../experiments/GA386_later_origin_neural_training/README.md) | 先頭74配列に後続24配列×2起点の教師を追加。40エポック、固定混合35% | 従来先頭0.694461、予約先頭0.610223、従来後続0.626572。GA351は未採点 |
| [GA392](../../experiments/GA392_expanded_later_origin_training/README.md) | 後続学習を74配列に増やし、先頭41,376・後続82,987ペアで読み出しを学習 | 検証3配列・9起点で全体0.615994、先頭0.667649、後続0.600755。確認パネル未採点 |

GA305からGA309の比較は起点・候補が一致し、静的な力の残差を加える上積みを確認できる。GA312以降の先頭起点だけの数字は、複数起点を含む結果と分けて解釈する。GA312は100 ps×8の生成・準備に並行実行下で131〜188秒/起点かかっており、分岐増加に伴う計算負荷も記録した。[GA309](../../experiments/GA309_static_force_native_blend/README.md) · [GA312](../../experiments/GA312_eight_seed_confirmation/README.md)

GA348からGA386にかけて先頭起点の相関は上がった一方、同じ従来配列の後続起点はGA348の0.630489がGA386の0.626572を上回った。GA384とGA348の新規GA351比較でも相関差は+0.000194、RMSEは+0.000926 Åだった。GA392はGA386から検証全体で+0.000972で、未採点パネルでの改善は判断できない。[GA351](../../experiments/GA351_sequence_disjoint_reserve/README.md) · [GA386](../../experiments/GA386_later_origin_neural_training/README.md) · [GA392](../../experiments/GA392_expanded_later_origin_training/README.md)

## 成果物と再現条件

GA01〜GA300は300個のチェックポイントと検証・選定後評価・観測対照を収録した。GA301〜GA392は各実験の記録に、利用できる学習済み重み、評価出力、実行コード・ログ、生成距離キャッシュを対応付けた。GA351のAmber／CHARMM各8件、GA385の後続起点48件も収録した。元のランディレクトリからの相対パスとGit上の収録先は台帳を参照。既存フォルダに同一内容があるファイルは、台帳からそのファイルにリンクした。[実験索引](INDEX.md) · [ファイル台帳](manifest.json)

GA323・GA325・GA326・GA328はGA320、GA385はGA386の記録に収録。GA349は照合したローカル作業ツリーと対象ブランチに記録が見つからず、索引上の欠番とした。GA303・GA304・GA335・GA336の配列別の派生キャッシュとGA361のモデルダウンロードキャッシュはアーカイブ対象外。再実行には各記録が指定するInternalResearchの作業ツリー、MiniTopolとATLAS、凍結モデル、試行に応じてMiniCalcやMD環境が必要になる。この整理では新規学習・再採点を実行していない。[GA301の再現手順](../../experiments/GA301_static_pair_offset/README.md) · [除外件数を含む台帳](manifest.json)

台帳の sha256 と bytes はGitに保存された内容に対する値。改行の正規化で元ファイルのハッシュと異なる場合は workspace_sha256 を併記した。
