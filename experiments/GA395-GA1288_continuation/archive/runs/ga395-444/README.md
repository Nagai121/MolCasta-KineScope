# GA395〜GA444 逐次探索

KineScopeのK21・K42/K44・K2-02/K2-05を踏まえ、水素結合形成候補の平均距離残差を、二力場の生成応答から推定した。[KineScopeの知見](https://github.com/Nagai121/MolCasta-KineScope/blob/27539ca7f2f0956077e6f53c61ab206e31cbfe97/FINDINGS.md)

GA392の開発用データ124,363ペア、検証3配列・9起点・4,820ペア。学習と検証は配列で分離。CUDA学習10エポック、CPU推論。GA348を65%、各読み出しを35%混合。既存の短い生成結果を再利用した。

1モデルずつ学習して全起点の検証相関を計測し、暫定最良を更新した。次の試行は暫定最良の近傍から選び、4回ごとに試行の少ない表現も比較した。仮説と採否は trials.jsonl に記録。学習時間にキャッシュ生成とデータ組み立ては含めていない。

| モデル | 全起点 | 先頭起点 | 後続起点 | 全体RMSE Å |
| --- | ---: | ---: | ---: | ---: |
| GA348 | 0.605836 | 0.666916 | 0.587522 | 0.514172 |
| GA392 | 0.615994 | 0.667649 | 0.600755 | 0.511661 |
| GA403 | 0.617960 | 0.665704 | 0.604983 | 0.508378 |

GA401からGA403では、二力場の平均距離応答と力場間の差との積を追加して、相関が0.616800から0.617960に上がった。GA403と同じ損失・構造でペアを均等に重み付けしたGA409は0.615007。タンパク質ごとの均等化がこの検証では有利だった。関係特徴・全時系列応答を使うGA402・GA407・GA410はGA403に届かなかった。

同じ3配列で50回選定した探索結果。独立確認パネルは未採点で、指導力回復率と下流AUPRCは今回測定していない。

## 試行順

| ID | 表現 / 損失 / 重み / 構造 | 全起点 | 先頭 | 後続 | 暫定最良との差 | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| GA395 | level / huber / later / mlp | 0.608560 | 0.656909 | 0.594690 | — | 採用 |
| GA396 | level / huber / uniform / mlp | 0.608210 | 0.658518 | 0.593594 | -0.000351 | 据え置き |
| GA397 | level / huber / later / linear | 0.601943 | 0.651692 | 0.587726 | -0.006617 | 据え置き |
| GA398 | level / mse / later / mlp | 0.611862 | 0.654760 | 0.600347 | +0.003301 | 採用 |
| GA399 | compact_level / mse / later / mlp | 0.609221 | 0.657602 | 0.595448 | -0.002640 | 据え置き |
| GA400 | level / mse / later / deep | 0.615857 | 0.659868 | 0.603967 | +0.003995 | 採用 |
| GA401 | level / mse / protein / deep | 0.616800 | 0.659418 | 0.605402 | +0.000943 | 採用 |
| GA402 | level_graph / mse / protein / deep | 0.610749 | 0.660018 | 0.597380 | -0.006051 | 据え置き |
| GA403 | level_interaction / mse / protein / deep | 0.617960 | 0.665704 | 0.604983 | +0.001160 | 採用 |
| GA404 | level_interaction / smooth_mae / protein / deep | 0.608985 | 0.667625 | 0.591375 | -0.008976 | 据え置き |
| GA405 | level_interaction / mse / protein / linear | 0.603309 | 0.651419 | 0.589574 | -0.014652 | 据え置き |
| GA406 | level_interaction / mse / protein_later / deep | 0.616001 | 0.661544 | 0.603382 | -0.001960 | 据え置き |
| GA407 | full_response / mse / protein / deep | 0.614588 | 0.660140 | 0.602245 | -0.003373 | 据え置き |
| GA408 | level_interaction / huber / protein / deep | 0.611448 | 0.667976 | 0.594688 | -0.006513 | 据え置き |
| GA409 | level_interaction / mse / uniform / deep | 0.615007 | 0.665018 | 0.601105 | -0.002954 | 据え置き |
| GA410 | full_graph / mse / protein / deep | 0.613710 | 0.659455 | 0.601650 | -0.004251 | 据え置き |
| GA411 | compact_level / mse / protein / deep | 0.607331 | 0.654916 | 0.593721 | -0.010629 | 据え置き |
| GA412 | level_interaction / mse / protein / mlp | 0.610977 | 0.655330 | 0.598946 | -0.006984 | 据え置き |
| GA413 | level_interaction / mse / later / deep | 0.614557 | 0.661850 | 0.601267 | -0.003404 | 据え置き |
| GA414 | level_interaction / smooth_mae / protein / linear | 0.601109 | 0.651102 | 0.586479 | -0.016851 | 据え置き |
| GA415 | level_graph / huber / protein / deep | 0.608767 | 0.666884 | 0.591065 | -0.009193 | 据え置き |
| GA416 | level_interaction / smooth_mae / protein_later / deep | 0.608215 | 0.665463 | 0.591085 | -0.009746 | 据え置き |
| GA417 | level_interaction / mse / uniform / linear | 0.602893 | 0.651505 | 0.588986 | -0.015068 | 据え置き |
| GA418 | full_response / smooth_mae / protein / deep | 0.610190 | 0.667187 | 0.593069 | -0.007771 | 据え置き |
| GA419 | full_graph / huber / protein / deep | 0.606425 | 0.668053 | 0.587344 | -0.011535 | 据え置き |
| GA420 | level_interaction / mse / protein_later / linear | 0.603730 | 0.651772 | 0.589983 | -0.014231 | 据え置き |
| GA421 | level_interaction / smooth_mae / uniform / deep | 0.609485 | 0.668262 | 0.591814 | -0.008476 | 据え置き |
| GA422 | compact_level / mse / protein / linear | 0.604270 | 0.649719 | 0.591708 | -0.013690 | 据え置き |
| GA423 | level_graph / mse / uniform / deep | 0.609600 | 0.660182 | 0.595653 | -0.008360 | 据え置き |
| GA424 | level_interaction / smooth_mae / protein / mlp | 0.609215 | 0.663424 | 0.593085 | -0.008746 | 据え置き |
| GA425 | level_interaction / huber / protein_later / deep | 0.610577 | 0.664836 | 0.594613 | -0.007383 | 据え置き |
| GA426 | full_response / mse / protein / linear | 0.604457 | 0.653343 | 0.590687 | -0.013504 | 据え置き |
| GA427 | full_graph / mse / uniform / deep | 0.611226 | 0.660354 | 0.597914 | -0.006734 | 据え置き |
| GA428 | level_interaction / huber / protein / mlp | 0.608905 | 0.660551 | 0.593769 | -0.009055 | 据え置き |
| GA429 | compact_level / mse / protein_later / deep | 0.609530 | 0.656998 | 0.596027 | -0.008430 | 据え置き |
| GA430 | level_graph / smooth_mae / protein / deep | 0.606823 | 0.668050 | 0.587892 | -0.011137 | 据え置き |
| GA431 | full_response / huber / protein / deep | 0.612307 | 0.666207 | 0.596366 | -0.005653 | 据え置き |
| GA432 | level_interaction / mse / protein_later / mlp | 0.613100 | 0.654118 | 0.602325 | -0.004860 | 据え置き |
| GA433 | level_interaction / mse / later / linear | 0.603148 | 0.652062 | 0.589088 | -0.014812 | 据え置き |
| GA434 | full_graph / smooth_mae / protein / deep | 0.606458 | 0.670602 | 0.586428 | -0.011502 | 据え置き |
| GA435 | compact_level / huber / protein / deep | 0.605212 | 0.656994 | 0.589900 | -0.012749 | 据え置き |
| GA436 | level_interaction / mse / uniform / mlp | 0.610329 | 0.655754 | 0.597900 | -0.007632 | 据え置き |
| GA437 | level_graph / mse / protein / linear | 0.604077 | 0.650929 | 0.590970 | -0.013884 | 据え置き |
| GA438 | level_interaction / smooth_mae / later / deep | 0.608948 | 0.665155 | 0.592228 | -0.009012 | 据え置き |
| GA439 | full_response / mse / uniform / deep | 0.614445 | 0.660605 | 0.601882 | -0.003516 | 据え置き |
| GA440 | level_interaction / huber / protein / linear | 0.601415 | 0.650879 | 0.587055 | -0.016545 | 据え置き |
| GA441 | full_graph / mse / protein / mlp | 0.608998 | 0.657513 | 0.595936 | -0.008963 | 据え置き |
| GA442 | level_graph / mse / protein_later / deep | 0.611632 | 0.656458 | 0.599701 | -0.006328 | 据え置き |
| GA443 | compact_level / mse / uniform / deep | 0.607018 | 0.655283 | 0.593157 | -0.010943 | 据え置き |
| GA444 | level_interaction / mse / later / mlp | 0.611452 | 0.653488 | 0.600337 | -0.006508 | 据え置き |
