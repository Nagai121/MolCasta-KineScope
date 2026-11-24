# GA01〜GA392 実験索引

[総合レポート](README.md) · [ファイル台帳](manifest.json)

## GA01〜GA300

凍結T1I2007を補正する1 ns先の水素結合形成予測。表は10配列での検証AUPRC。全指標・計時・GPUメモリは[実測台帳](../../experiments/GA01-GA300_static_guidance_screen/results.json)を参照。

| ID | 入力 | 生成目標 | 構造 | 検証AUPRC |
| --- | --- | --- | --- | ---: |
| [GA01](../../experiments/GA01-GA300_static_guidance_screen/GA01.pt) | distance | point | additive | 0.500010 |
| [GA02](../../experiments/GA01-GA300_static_guidance_screen/GA02.pt) | distance | point | lowrank | 0.499890 |
| [GA03](../../experiments/GA01-GA300_static_guidance_screen/GA03.pt) | distance | point | gated | 0.500025 |
| [GA04](../../experiments/GA01-GA300_static_guidance_screen/GA04.pt) | distance | point | experts | 0.500033 |
| [GA05](../../experiments/GA01-GA300_static_guidance_screen/GA05.pt) | distance | point | residual | 0.499930 |
| [GA06](../../experiments/GA01-GA300_static_guidance_screen/GA06.pt) | distance | moments | additive | 0.500007 |
| [GA07](../../experiments/GA01-GA300_static_guidance_screen/GA07.pt) | distance | moments | lowrank | 0.499758 |
| [GA08](../../experiments/GA01-GA300_static_guidance_screen/GA08.pt) | distance | moments | gated | 0.500016 |
| [GA09](../../experiments/GA01-GA300_static_guidance_screen/GA09.pt) | distance | moments | experts | 0.499236 |
| [GA10](../../experiments/GA01-GA300_static_guidance_screen/GA10.pt) | distance | moments | residual | 0.499942 |
| [GA11](../../experiments/GA01-GA300_static_guidance_screen/GA11.pt) | distance | hurdle | additive | 0.500031 |
| [GA12](../../experiments/GA01-GA300_static_guidance_screen/GA12.pt) | distance | hurdle | lowrank | 0.499917 |
| [GA13](../../experiments/GA01-GA300_static_guidance_screen/GA13.pt) | distance | hurdle | gated | 0.499901 |
| [GA14](../../experiments/GA01-GA300_static_guidance_screen/GA14.pt) | distance | hurdle | experts | 0.499992 |
| [GA15](../../experiments/GA01-GA300_static_guidance_screen/GA15.pt) | distance | hurdle | residual | 0.499833 |
| [GA16](../../experiments/GA01-GA300_static_guidance_screen/GA16.pt) | distance | distribution | additive | 0.499897 |
| [GA17](../../experiments/GA01-GA300_static_guidance_screen/GA17.pt) | distance | distribution | lowrank | 0.499833 |
| [GA18](../../experiments/GA01-GA300_static_guidance_screen/GA18.pt) | distance | distribution | gated | 0.500015 |
| [GA19](../../experiments/GA01-GA300_static_guidance_screen/GA19.pt) | distance | distribution | experts | 0.500002 |
| [GA20](../../experiments/GA01-GA300_static_guidance_screen/GA20.pt) | distance | distribution | residual | 0.499909 |
| [GA21](../../experiments/GA01-GA300_static_guidance_screen/GA21.pt) | distance | joint | additive | 0.499906 |
| [GA22](../../experiments/GA01-GA300_static_guidance_screen/GA22.pt) | distance | joint | lowrank | 0.499821 |
| [GA23](../../experiments/GA01-GA300_static_guidance_screen/GA23.pt) | distance | joint | gated | 0.499744 |
| [GA24](../../experiments/GA01-GA300_static_guidance_screen/GA24.pt) | distance | joint | experts | 0.499874 |
| [GA25](../../experiments/GA01-GA300_static_guidance_screen/GA25.pt) | distance | joint | residual | 0.499935 |
| [GA26](../../experiments/GA01-GA300_static_guidance_screen/GA26.pt) | existing_static | point | additive | 0.500100 |
| [GA27](../../experiments/GA01-GA300_static_guidance_screen/GA27.pt) | existing_static | point | lowrank | 0.500272 |
| [GA28](../../experiments/GA01-GA300_static_guidance_screen/GA28.pt) | existing_static | point | gated | 0.500367 |
| [GA29](../../experiments/GA01-GA300_static_guidance_screen/GA29.pt) | existing_static | point | experts | 0.500246 |
| [GA30](../../experiments/GA01-GA300_static_guidance_screen/GA30.pt) | existing_static | point | residual | 0.499615 |
| [GA31](../../experiments/GA01-GA300_static_guidance_screen/GA31.pt) | existing_static | moments | additive | 0.499571 |
| [GA32](../../experiments/GA01-GA300_static_guidance_screen/GA32.pt) | existing_static | moments | lowrank | 0.500225 |
| [GA33](../../experiments/GA01-GA300_static_guidance_screen/GA33.pt) | existing_static | moments | gated | 0.499934 |
| [GA34](../../experiments/GA01-GA300_static_guidance_screen/GA34.pt) | existing_static | moments | experts | 0.500019 |
| [GA35](../../experiments/GA01-GA300_static_guidance_screen/GA35.pt) | existing_static | moments | residual | 0.500401 |
| [GA36](../../experiments/GA01-GA300_static_guidance_screen/GA36.pt) | existing_static | hurdle | additive | 0.500169 |
| [GA37](../../experiments/GA01-GA300_static_guidance_screen/GA37.pt) | existing_static | hurdle | lowrank | 0.500169 |
| [GA38](../../experiments/GA01-GA300_static_guidance_screen/GA38.pt) | existing_static | hurdle | gated | 0.500261 |
| [GA39](../../experiments/GA01-GA300_static_guidance_screen/GA39.pt) | existing_static | hurdle | experts | 0.500323 |
| [GA40](../../experiments/GA01-GA300_static_guidance_screen/GA40.pt) | existing_static | hurdle | residual | 0.500207 |
| [GA41](../../experiments/GA01-GA300_static_guidance_screen/GA41.pt) | existing_static | distribution | additive | 0.500319 |
| [GA42](../../experiments/GA01-GA300_static_guidance_screen/GA42.pt) | existing_static | distribution | lowrank | 0.500096 |
| [GA43](../../experiments/GA01-GA300_static_guidance_screen/GA43.pt) | existing_static | distribution | gated | 0.499970 |
| [GA44](../../experiments/GA01-GA300_static_guidance_screen/GA44.pt) | existing_static | distribution | experts | 0.499883 |
| [GA45](../../experiments/GA01-GA300_static_guidance_screen/GA45.pt) | existing_static | distribution | residual | 0.500201 |
| [GA46](../../experiments/GA01-GA300_static_guidance_screen/GA46.pt) | existing_static | joint | additive | 0.500059 |
| [GA47](../../experiments/GA01-GA300_static_guidance_screen/GA47.pt) | existing_static | joint | lowrank | 0.500117 |
| [GA48](../../experiments/GA01-GA300_static_guidance_screen/GA48.pt) | existing_static | joint | gated | 0.500247 |
| [GA49](../../experiments/GA01-GA300_static_guidance_screen/GA49.pt) | existing_static | joint | experts | 0.500415 |
| [GA50](../../experiments/GA01-GA300_static_guidance_screen/GA50.pt) | existing_static | joint | residual | 0.500221 |
| [GA51](../../experiments/GA01-GA300_static_guidance_screen/GA51.pt) | local_response | point | additive | 0.500053 |
| [GA52](../../experiments/GA01-GA300_static_guidance_screen/GA52.pt) | local_response | point | lowrank | 0.500064 |
| [GA53](../../experiments/GA01-GA300_static_guidance_screen/GA53.pt) | local_response | point | gated | 0.499912 |
| [GA54](../../experiments/GA01-GA300_static_guidance_screen/GA54.pt) | local_response | point | experts | 0.499856 |
| [GA55](../../experiments/GA01-GA300_static_guidance_screen/GA55.pt) | local_response | point | residual | 0.499534 |
| [GA56](../../experiments/GA01-GA300_static_guidance_screen/GA56.pt) | local_response | moments | additive | 0.499878 |
| [GA57](../../experiments/GA01-GA300_static_guidance_screen/GA57.pt) | local_response | moments | lowrank | 0.499306 |
| [GA58](../../experiments/GA01-GA300_static_guidance_screen/GA58.pt) | local_response | moments | gated | 0.499849 |
| [GA59](../../experiments/GA01-GA300_static_guidance_screen/GA59.pt) | local_response | moments | experts | 0.498964 |
| [GA60](../../experiments/GA01-GA300_static_guidance_screen/GA60.pt) | local_response | moments | residual | 0.499470 |
| [GA61](../../experiments/GA01-GA300_static_guidance_screen/GA61.pt) | local_response | hurdle | additive | 0.500031 |
| [GA62](../../experiments/GA01-GA300_static_guidance_screen/GA62.pt) | local_response | hurdle | lowrank | 0.499781 |
| [GA63](../../experiments/GA01-GA300_static_guidance_screen/GA63.pt) | local_response | hurdle | gated | 0.499513 |
| [GA64](../../experiments/GA01-GA300_static_guidance_screen/GA64.pt) | local_response | hurdle | experts | 0.499741 |
| [GA65](../../experiments/GA01-GA300_static_guidance_screen/GA65.pt) | local_response | hurdle | residual | 0.499509 |
| [GA66](../../experiments/GA01-GA300_static_guidance_screen/GA66.pt) | local_response | distribution | additive | 0.499811 |
| [GA67](../../experiments/GA01-GA300_static_guidance_screen/GA67.pt) | local_response | distribution | lowrank | 0.499739 |
| [GA68](../../experiments/GA01-GA300_static_guidance_screen/GA68.pt) | local_response | distribution | gated | 0.499944 |
| [GA69](../../experiments/GA01-GA300_static_guidance_screen/GA69.pt) | local_response | distribution | experts | 0.499832 |
| [GA70](../../experiments/GA01-GA300_static_guidance_screen/GA70.pt) | local_response | distribution | residual | 0.500297 |
| [GA71](../../experiments/GA01-GA300_static_guidance_screen/GA71.pt) | local_response | joint | additive | 0.499430 |
| [GA72](../../experiments/GA01-GA300_static_guidance_screen/GA72.pt) | local_response | joint | lowrank | 0.499726 |
| [GA73](../../experiments/GA01-GA300_static_guidance_screen/GA73.pt) | local_response | joint | gated | 0.499326 |
| [GA74](../../experiments/GA01-GA300_static_guidance_screen/GA74.pt) | local_response | joint | experts | 0.499159 |
| [GA75](../../experiments/GA01-GA300_static_guidance_screen/GA75.pt) | local_response | joint | residual | 0.500044 |
| [GA76](../../experiments/GA01-GA300_static_guidance_screen/GA76.pt) | network_response | point | additive | 0.499655 |
| [GA77](../../experiments/GA01-GA300_static_guidance_screen/GA77.pt) | network_response | point | lowrank | 0.499949 |
| [GA78](../../experiments/GA01-GA300_static_guidance_screen/GA78.pt) | network_response | point | gated | 0.499610 |
| [GA79](../../experiments/GA01-GA300_static_guidance_screen/GA79.pt) | network_response | point | experts | 0.499902 |
| [GA80](../../experiments/GA01-GA300_static_guidance_screen/GA80.pt) | network_response | point | residual | 0.499440 |
| [GA81](../../experiments/GA01-GA300_static_guidance_screen/GA81.pt) | network_response | moments | additive | 0.499211 |
| [GA82](../../experiments/GA01-GA300_static_guidance_screen/GA82.pt) | network_response | moments | lowrank | 0.499369 |
| [GA83](../../experiments/GA01-GA300_static_guidance_screen/GA83.pt) | network_response | moments | gated | 0.499512 |
| [GA84](../../experiments/GA01-GA300_static_guidance_screen/GA84.pt) | network_response | moments | experts | 0.499238 |
| [GA85](../../experiments/GA01-GA300_static_guidance_screen/GA85.pt) | network_response | moments | residual | 0.499035 |
| [GA86](../../experiments/GA01-GA300_static_guidance_screen/GA86.pt) | network_response | hurdle | additive | 0.499685 |
| [GA87](../../experiments/GA01-GA300_static_guidance_screen/GA87.pt) | network_response | hurdle | lowrank | 0.499886 |
| [GA88](../../experiments/GA01-GA300_static_guidance_screen/GA88.pt) | network_response | hurdle | gated | 0.499875 |
| [GA89](../../experiments/GA01-GA300_static_guidance_screen/GA89.pt) | network_response | hurdle | experts | 0.499319 |
| [GA90](../../experiments/GA01-GA300_static_guidance_screen/GA90.pt) | network_response | hurdle | residual | 0.499839 |
| [GA91](../../experiments/GA01-GA300_static_guidance_screen/GA91.pt) | network_response | distribution | additive | 0.499780 |
| [GA92](../../experiments/GA01-GA300_static_guidance_screen/GA92.pt) | network_response | distribution | lowrank | 0.499867 |
| [GA93](../../experiments/GA01-GA300_static_guidance_screen/GA93.pt) | network_response | distribution | gated | 0.499595 |
| [GA94](../../experiments/GA01-GA300_static_guidance_screen/GA94.pt) | network_response | distribution | experts | 0.499687 |
| [GA95](../../experiments/GA01-GA300_static_guidance_screen/GA95.pt) | network_response | distribution | residual | 0.500156 |
| [GA96](../../experiments/GA01-GA300_static_guidance_screen/GA96.pt) | network_response | joint | additive | 0.499683 |
| [GA97](../../experiments/GA01-GA300_static_guidance_screen/GA97.pt) | network_response | joint | lowrank | 0.499390 |
| [GA98](../../experiments/GA01-GA300_static_guidance_screen/GA98.pt) | network_response | joint | gated | 0.499639 |
| [GA99](../../experiments/GA01-GA300_static_guidance_screen/GA99.pt) | network_response | joint | experts | 0.499275 |
| [GA100](../../experiments/GA01-GA300_static_guidance_screen/GA100.pt) | network_response | joint | residual | 0.499457 |
| [GA101](../../experiments/GA01-GA300_static_guidance_screen/GA101.pt) | occupancy | point | additive | 0.499901 |
| [GA102](../../experiments/GA01-GA300_static_guidance_screen/GA102.pt) | occupancy | point | lowrank | 0.499934 |
| [GA103](../../experiments/GA01-GA300_static_guidance_screen/GA103.pt) | occupancy | point | gated | 0.500114 |
| [GA104](../../experiments/GA01-GA300_static_guidance_screen/GA104.pt) | occupancy | point | experts | 0.499763 |
| [GA105](../../experiments/GA01-GA300_static_guidance_screen/GA105.pt) | occupancy | point | residual | 0.500020 |
| [GA106](../../experiments/GA01-GA300_static_guidance_screen/GA106.pt) | occupancy | moments | additive | 0.499653 |
| [GA107](../../experiments/GA01-GA300_static_guidance_screen/GA107.pt) | occupancy | moments | lowrank | 0.499472 |
| [GA108](../../experiments/GA01-GA300_static_guidance_screen/GA108.pt) | occupancy | moments | gated | 0.500257 |
| [GA109](../../experiments/GA01-GA300_static_guidance_screen/GA109.pt) | occupancy | moments | experts | 0.499927 |
| [GA110](../../experiments/GA01-GA300_static_guidance_screen/GA110.pt) | occupancy | moments | residual | 0.500122 |
| [GA111](../../experiments/GA01-GA300_static_guidance_screen/GA111.pt) | occupancy | hurdle | additive | 0.499995 |
| [GA112](../../experiments/GA01-GA300_static_guidance_screen/GA112.pt) | occupancy | hurdle | lowrank | 0.500065 |
| [GA113](../../experiments/GA01-GA300_static_guidance_screen/GA113.pt) | occupancy | hurdle | gated | 0.500115 |
| [GA114](../../experiments/GA01-GA300_static_guidance_screen/GA114.pt) | occupancy | hurdle | experts | 0.500041 |
| [GA115](../../experiments/GA01-GA300_static_guidance_screen/GA115.pt) | occupancy | hurdle | residual | 0.499968 |
| [GA116](../../experiments/GA01-GA300_static_guidance_screen/GA116.pt) | occupancy | distribution | additive | 0.499793 |
| [GA117](../../experiments/GA01-GA300_static_guidance_screen/GA117.pt) | occupancy | distribution | lowrank | 0.500043 |
| [GA118](../../experiments/GA01-GA300_static_guidance_screen/GA118.pt) | occupancy | distribution | gated | 0.500003 |
| [GA119](../../experiments/GA01-GA300_static_guidance_screen/GA119.pt) | occupancy | distribution | experts | 0.500028 |
| [GA120](../../experiments/GA01-GA300_static_guidance_screen/GA120.pt) | occupancy | distribution | residual | 0.500151 |
| [GA121](../../experiments/GA01-GA300_static_guidance_screen/GA121.pt) | occupancy | joint | additive | 0.499856 |
| [GA122](../../experiments/GA01-GA300_static_guidance_screen/GA122.pt) | occupancy | joint | lowrank | 0.499919 |
| [GA123](../../experiments/GA01-GA300_static_guidance_screen/GA123.pt) | occupancy | joint | gated | 0.500103 |
| [GA124](../../experiments/GA01-GA300_static_guidance_screen/GA124.pt) | occupancy | joint | experts | 0.499880 |
| [GA125](../../experiments/GA01-GA300_static_guidance_screen/GA125.pt) | occupancy | joint | residual | 0.500068 |
| [GA126](../../experiments/GA01-GA300_static_guidance_screen/GA126.pt) | competitors | point | additive | 0.499933 |
| [GA127](../../experiments/GA01-GA300_static_guidance_screen/GA127.pt) | competitors | point | lowrank | 0.499940 |
| [GA128](../../experiments/GA01-GA300_static_guidance_screen/GA128.pt) | competitors | point | gated | 0.500131 |
| [GA129](../../experiments/GA01-GA300_static_guidance_screen/GA129.pt) | competitors | point | experts | 0.500061 |
| [GA130](../../experiments/GA01-GA300_static_guidance_screen/GA130.pt) | competitors | point | residual | 0.500146 |
| [GA131](../../experiments/GA01-GA300_static_guidance_screen/GA131.pt) | competitors | moments | additive | 0.499925 |
| [GA132](../../experiments/GA01-GA300_static_guidance_screen/GA132.pt) | competitors | moments | lowrank | 0.499663 |
| [GA133](../../experiments/GA01-GA300_static_guidance_screen/GA133.pt) | competitors | moments | gated | 0.500043 |
| [GA134](../../experiments/GA01-GA300_static_guidance_screen/GA134.pt) | competitors | moments | experts | 0.499878 |
| [GA135](../../experiments/GA01-GA300_static_guidance_screen/GA135.pt) | competitors | moments | residual | 0.499807 |
| [GA136](../../experiments/GA01-GA300_static_guidance_screen/GA136.pt) | competitors | hurdle | additive | 0.500150 |
| [GA137](../../experiments/GA01-GA300_static_guidance_screen/GA137.pt) | competitors | hurdle | lowrank | 0.500038 |
| [GA138](../../experiments/GA01-GA300_static_guidance_screen/GA138.pt) | competitors | hurdle | gated | 0.500062 |
| [GA139](../../experiments/GA01-GA300_static_guidance_screen/GA139.pt) | competitors | hurdle | experts | 0.500171 |
| [GA140](../../experiments/GA01-GA300_static_guidance_screen/GA140.pt) | competitors | hurdle | residual | 0.499922 |
| [GA141](../../experiments/GA01-GA300_static_guidance_screen/GA141.pt) | competitors | distribution | additive | 0.499964 |
| [GA142](../../experiments/GA01-GA300_static_guidance_screen/GA142.pt) | competitors | distribution | lowrank | 0.498948 |
| [GA143](../../experiments/GA01-GA300_static_guidance_screen/GA143.pt) | competitors | distribution | gated | 0.500025 |
| [GA144](../../experiments/GA01-GA300_static_guidance_screen/GA144.pt) | competitors | distribution | experts | 0.499940 |
| [GA145](../../experiments/GA01-GA300_static_guidance_screen/GA145.pt) | competitors | distribution | residual | 0.500178 |
| [GA146](../../experiments/GA01-GA300_static_guidance_screen/GA146.pt) | competitors | joint | additive | 0.500068 |
| [GA147](../../experiments/GA01-GA300_static_guidance_screen/GA147.pt) | competitors | joint | lowrank | 0.499012 |
| [GA148](../../experiments/GA01-GA300_static_guidance_screen/GA148.pt) | competitors | joint | gated | 0.500068 |
| [GA149](../../experiments/GA01-GA300_static_guidance_screen/GA149.pt) | competitors | joint | experts | 0.499827 |
| [GA150](../../experiments/GA01-GA300_static_guidance_screen/GA150.pt) | competitors | joint | residual | 0.499487 |
| [GA151](../../experiments/GA01-GA300_static_guidance_screen/GA151.pt) | combined | point | additive | 0.499887 |
| [GA152](../../experiments/GA01-GA300_static_guidance_screen/GA152.pt) | combined | point | lowrank | 0.499998 |
| [GA153](../../experiments/GA01-GA300_static_guidance_screen/GA153.pt) | combined | point | gated | 0.500072 |
| [GA154](../../experiments/GA01-GA300_static_guidance_screen/GA154.pt) | combined | point | experts | 0.500030 |
| [GA155](../../experiments/GA01-GA300_static_guidance_screen/GA155.pt) | combined | point | residual | 0.500094 |
| [GA156](../../experiments/GA01-GA300_static_guidance_screen/GA156.pt) | combined | moments | additive | 0.499658 |
| [GA157](../../experiments/GA01-GA300_static_guidance_screen/GA157.pt) | combined | moments | lowrank | 0.499586 |
| [GA158](../../experiments/GA01-GA300_static_guidance_screen/GA158.pt) | combined | moments | gated | 0.499944 |
| [GA159](../../experiments/GA01-GA300_static_guidance_screen/GA159.pt) | combined | moments | experts | 0.499720 |
| [GA160](../../experiments/GA01-GA300_static_guidance_screen/GA160.pt) | combined | moments | residual | 0.499753 |
| [GA161](../../experiments/GA01-GA300_static_guidance_screen/GA161.pt) | combined | hurdle | additive | 0.500153 |
| [GA162](../../experiments/GA01-GA300_static_guidance_screen/GA162.pt) | combined | hurdle | lowrank | 0.500092 |
| [GA163](../../experiments/GA01-GA300_static_guidance_screen/GA163.pt) | combined | hurdle | gated | 0.499976 |
| [GA164](../../experiments/GA01-GA300_static_guidance_screen/GA164.pt) | combined | hurdle | experts | 0.500051 |
| [GA165](../../experiments/GA01-GA300_static_guidance_screen/GA165.pt) | combined | hurdle | residual | 0.500119 |
| [GA166](../../experiments/GA01-GA300_static_guidance_screen/GA166.pt) | combined | distribution | additive | 0.500122 |
| [GA167](../../experiments/GA01-GA300_static_guidance_screen/GA167.pt) | combined | distribution | lowrank | 0.500176 |
| [GA168](../../experiments/GA01-GA300_static_guidance_screen/GA168.pt) | combined | distribution | gated | 0.500238 |
| [GA169](../../experiments/GA01-GA300_static_guidance_screen/GA169.pt) | combined | distribution | experts | 0.500237 |
| [GA170](../../experiments/GA01-GA300_static_guidance_screen/GA170.pt) | combined | distribution | residual | 0.500031 |
| [GA171](../../experiments/GA01-GA300_static_guidance_screen/GA171.pt) | combined | joint | additive | 0.500092 |
| [GA172](../../experiments/GA01-GA300_static_guidance_screen/GA172.pt) | combined | joint | lowrank | 0.499613 |
| [GA173](../../experiments/GA01-GA300_static_guidance_screen/GA173.pt) | combined | joint | gated | 0.500037 |
| [GA174](../../experiments/GA01-GA300_static_guidance_screen/GA174.pt) | combined | joint | experts | 0.500136 |
| [GA175](../../experiments/GA01-GA300_static_guidance_screen/GA175.pt) | combined | joint | residual | 0.500462 |
| [GA176](../../experiments/GA01-GA300_static_guidance_screen/GA176.pt) | straight | point | additive | 0.500038 |
| [GA177](../../experiments/GA01-GA300_static_guidance_screen/GA177.pt) | straight | point | lowrank | 0.499971 |
| [GA178](../../experiments/GA01-GA300_static_guidance_screen/GA178.pt) | straight | point | gated | 0.499922 |
| [GA179](../../experiments/GA01-GA300_static_guidance_screen/GA179.pt) | straight | point | experts | 0.499972 |
| [GA180](../../experiments/GA01-GA300_static_guidance_screen/GA180.pt) | straight | point | residual | 0.499583 |
| [GA181](../../experiments/GA01-GA300_static_guidance_screen/GA181.pt) | straight | moments | additive | 0.500036 |
| [GA182](../../experiments/GA01-GA300_static_guidance_screen/GA182.pt) | straight | moments | lowrank | 0.500003 |
| [GA183](../../experiments/GA01-GA300_static_guidance_screen/GA183.pt) | straight | moments | gated | 0.499854 |
| [GA184](../../experiments/GA01-GA300_static_guidance_screen/GA184.pt) | straight | moments | experts | 0.499768 |
| [GA185](../../experiments/GA01-GA300_static_guidance_screen/GA185.pt) | straight | moments | residual | 0.499977 |
| [GA186](../../experiments/GA01-GA300_static_guidance_screen/GA186.pt) | straight | hurdle | additive | 0.500010 |
| [GA187](../../experiments/GA01-GA300_static_guidance_screen/GA187.pt) | straight | hurdle | lowrank | 0.499809 |
| [GA188](../../experiments/GA01-GA300_static_guidance_screen/GA188.pt) | straight | hurdle | gated | 0.499924 |
| [GA189](../../experiments/GA01-GA300_static_guidance_screen/GA189.pt) | straight | hurdle | experts | 0.499912 |
| [GA190](../../experiments/GA01-GA300_static_guidance_screen/GA190.pt) | straight | hurdle | residual | 0.499742 |
| [GA191](../../experiments/GA01-GA300_static_guidance_screen/GA191.pt) | straight | distribution | additive | 0.499971 |
| [GA192](../../experiments/GA01-GA300_static_guidance_screen/GA192.pt) | straight | distribution | lowrank | 0.499814 |
| [GA193](../../experiments/GA01-GA300_static_guidance_screen/GA193.pt) | straight | distribution | gated | 0.499445 |
| [GA194](../../experiments/GA01-GA300_static_guidance_screen/GA194.pt) | straight | distribution | experts | 0.499956 |
| [GA195](../../experiments/GA01-GA300_static_guidance_screen/GA195.pt) | straight | distribution | residual | 0.499773 |
| [GA196](../../experiments/GA01-GA300_static_guidance_screen/GA196.pt) | straight | joint | additive | 0.499627 |
| [GA197](../../experiments/GA01-GA300_static_guidance_screen/GA197.pt) | straight | joint | lowrank | 0.499872 |
| [GA198](../../experiments/GA01-GA300_static_guidance_screen/GA198.pt) | straight | joint | gated | 0.498840 |
| [GA199](../../experiments/GA01-GA300_static_guidance_screen/GA199.pt) | straight | joint | experts | 0.499777 |
| [GA200](../../experiments/GA01-GA300_static_guidance_screen/GA200.pt) | straight | joint | residual | 0.499883 |
| [GA201](../../experiments/GA01-GA300_static_guidance_screen/GA201.pt) | detour | point | additive | 0.499505 |
| [GA202](../../experiments/GA01-GA300_static_guidance_screen/GA202.pt) | detour | point | lowrank | 0.499976 |
| [GA203](../../experiments/GA01-GA300_static_guidance_screen/GA203.pt) | detour | point | gated | 0.499703 |
| [GA204](../../experiments/GA01-GA300_static_guidance_screen/GA204.pt) | detour | point | experts | 0.499523 |
| [GA205](../../experiments/GA01-GA300_static_guidance_screen/GA205.pt) | detour | point | residual | 0.499615 |
| [GA206](../../experiments/GA01-GA300_static_guidance_screen/GA206.pt) | detour | moments | additive | 0.500265 |
| [GA207](../../experiments/GA01-GA300_static_guidance_screen/GA207.pt) | detour | moments | lowrank | 0.500054 |
| [GA208](../../experiments/GA01-GA300_static_guidance_screen/GA208.pt) | detour | moments | gated | 0.499566 |
| [GA209](../../experiments/GA01-GA300_static_guidance_screen/GA209.pt) | detour | moments | experts | 0.499977 |
| [GA210](../../experiments/GA01-GA300_static_guidance_screen/GA210.pt) | detour | moments | residual | 0.499940 |
| [GA211](../../experiments/GA01-GA300_static_guidance_screen/GA211.pt) | detour | hurdle | additive | 0.499458 |
| [GA212](../../experiments/GA01-GA300_static_guidance_screen/GA212.pt) | detour | hurdle | lowrank | 0.499964 |
| [GA213](../../experiments/GA01-GA300_static_guidance_screen/GA213.pt) | detour | hurdle | gated | 0.499853 |
| [GA214](../../experiments/GA01-GA300_static_guidance_screen/GA214.pt) | detour | hurdle | experts | 0.499563 |
| [GA215](../../experiments/GA01-GA300_static_guidance_screen/GA215.pt) | detour | hurdle | residual | 0.499821 |
| [GA216](../../experiments/GA01-GA300_static_guidance_screen/GA216.pt) | detour | distribution | additive | 0.499612 |
| [GA217](../../experiments/GA01-GA300_static_guidance_screen/GA217.pt) | detour | distribution | lowrank | 0.500131 |
| [GA218](../../experiments/GA01-GA300_static_guidance_screen/GA218.pt) | detour | distribution | gated | 0.499921 |
| [GA219](../../experiments/GA01-GA300_static_guidance_screen/GA219.pt) | detour | distribution | experts | 0.499477 |
| [GA220](../../experiments/GA01-GA300_static_guidance_screen/GA220.pt) | detour | distribution | residual | 0.499565 |
| [GA221](../../experiments/GA01-GA300_static_guidance_screen/GA221.pt) | detour | joint | additive | 0.499725 |
| [GA222](../../experiments/GA01-GA300_static_guidance_screen/GA222.pt) | detour | joint | lowrank | 0.499822 |
| [GA223](../../experiments/GA01-GA300_static_guidance_screen/GA223.pt) | detour | joint | gated | 0.499716 |
| [GA224](../../experiments/GA01-GA300_static_guidance_screen/GA224.pt) | detour | joint | experts | 0.500157 |
| [GA225](../../experiments/GA01-GA300_static_guidance_screen/GA225.pt) | detour | joint | residual | 0.499473 |
| [GA226](../../experiments/GA01-GA300_static_guidance_screen/GA226.pt) | anisotropy | point | additive | 0.500131 |
| [GA227](../../experiments/GA01-GA300_static_guidance_screen/GA227.pt) | anisotropy | point | lowrank | 0.499995 |
| [GA228](../../experiments/GA01-GA300_static_guidance_screen/GA228.pt) | anisotropy | point | gated | 0.499934 |
| [GA229](../../experiments/GA01-GA300_static_guidance_screen/GA229.pt) | anisotropy | point | experts | 0.500055 |
| [GA230](../../experiments/GA01-GA300_static_guidance_screen/GA230.pt) | anisotropy | point | residual | 0.499796 |
| [GA231](../../experiments/GA01-GA300_static_guidance_screen/GA231.pt) | anisotropy | moments | additive | 0.500022 |
| [GA232](../../experiments/GA01-GA300_static_guidance_screen/GA232.pt) | anisotropy | moments | lowrank | 0.499898 |
| [GA233](../../experiments/GA01-GA300_static_guidance_screen/GA233.pt) | anisotropy | moments | gated | 0.499896 |
| [GA234](../../experiments/GA01-GA300_static_guidance_screen/GA234.pt) | anisotropy | moments | experts | 0.499977 |
| [GA235](../../experiments/GA01-GA300_static_guidance_screen/GA235.pt) | anisotropy | moments | residual | 0.499662 |
| [GA236](../../experiments/GA01-GA300_static_guidance_screen/GA236.pt) | anisotropy | hurdle | additive | 0.499696 |
| [GA237](../../experiments/GA01-GA300_static_guidance_screen/GA237.pt) | anisotropy | hurdle | lowrank | 0.499641 |
| [GA238](../../experiments/GA01-GA300_static_guidance_screen/GA238.pt) | anisotropy | hurdle | gated | 0.499826 |
| [GA239](../../experiments/GA01-GA300_static_guidance_screen/GA239.pt) | anisotropy | hurdle | experts | 0.499609 |
| [GA240](../../experiments/GA01-GA300_static_guidance_screen/GA240.pt) | anisotropy | hurdle | residual | 0.499764 |
| [GA241](../../experiments/GA01-GA300_static_guidance_screen/GA241.pt) | anisotropy | distribution | additive | 0.499656 |
| [GA242](../../experiments/GA01-GA300_static_guidance_screen/GA242.pt) | anisotropy | distribution | lowrank | 0.499868 |
| [GA243](../../experiments/GA01-GA300_static_guidance_screen/GA243.pt) | anisotropy | distribution | gated | 0.499745 |
| [GA244](../../experiments/GA01-GA300_static_guidance_screen/GA244.pt) | anisotropy | distribution | experts | 0.499630 |
| [GA245](../../experiments/GA01-GA300_static_guidance_screen/GA245.pt) | anisotropy | distribution | residual | 0.499859 |
| [GA246](../../experiments/GA01-GA300_static_guidance_screen/GA246.pt) | anisotropy | joint | additive | 0.499109 |
| [GA247](../../experiments/GA01-GA300_static_guidance_screen/GA247.pt) | anisotropy | joint | lowrank | 0.500126 |
| [GA248](../../experiments/GA01-GA300_static_guidance_screen/GA248.pt) | anisotropy | joint | gated | 0.499774 |
| [GA249](../../experiments/GA01-GA300_static_guidance_screen/GA249.pt) | anisotropy | joint | experts | 0.499885 |
| [GA250](../../experiments/GA01-GA300_static_guidance_screen/GA250.pt) | anisotropy | joint | residual | 0.499778 |
| [GA251](../../experiments/GA01-GA300_static_guidance_screen/GA251.pt) | availability | point | additive | 0.500198 |
| [GA252](../../experiments/GA01-GA300_static_guidance_screen/GA252.pt) | availability | point | lowrank | 0.500006 |
| [GA253](../../experiments/GA01-GA300_static_guidance_screen/GA253.pt) | availability | point | gated | 0.500290 |
| [GA254](../../experiments/GA01-GA300_static_guidance_screen/GA254.pt) | availability | point | experts | 0.500012 |
| [GA255](../../experiments/GA01-GA300_static_guidance_screen/GA255.pt) | availability | point | residual | 0.500104 |
| [GA256](../../experiments/GA01-GA300_static_guidance_screen/GA256.pt) | availability | moments | additive | 0.500004 |
| [GA257](../../experiments/GA01-GA300_static_guidance_screen/GA257.pt) | availability | moments | lowrank | 0.500109 |
| [GA258](../../experiments/GA01-GA300_static_guidance_screen/GA258.pt) | availability | moments | gated | 0.499979 |
| [GA259](../../experiments/GA01-GA300_static_guidance_screen/GA259.pt) | availability | moments | experts | 0.500580 |
| [GA260](../../experiments/GA01-GA300_static_guidance_screen/GA260.pt) | availability | moments | residual | 0.499983 |
| [GA261](../../experiments/GA01-GA300_static_guidance_screen/GA261.pt) | availability | hurdle | additive | 0.500027 |
| [GA262](../../experiments/GA01-GA300_static_guidance_screen/GA262.pt) | availability | hurdle | lowrank | 0.499844 |
| [GA263](../../experiments/GA01-GA300_static_guidance_screen/GA263.pt) | availability | hurdle | gated | 0.499742 |
| [GA264](../../experiments/GA01-GA300_static_guidance_screen/GA264.pt) | availability | hurdle | experts | 0.500031 |
| [GA265](../../experiments/GA01-GA300_static_guidance_screen/GA265.pt) | availability | hurdle | residual | 0.499930 |
| [GA266](../../experiments/GA01-GA300_static_guidance_screen/GA266.pt) | availability | distribution | additive | 0.499840 |
| [GA267](../../experiments/GA01-GA300_static_guidance_screen/GA267.pt) | availability | distribution | lowrank | 0.499791 |
| [GA268](../../experiments/GA01-GA300_static_guidance_screen/GA268.pt) | availability | distribution | gated | 0.499911 |
| [GA269](../../experiments/GA01-GA300_static_guidance_screen/GA269.pt) | availability | distribution | experts | 0.499859 |
| [GA270](../../experiments/GA01-GA300_static_guidance_screen/GA270.pt) | availability | distribution | residual | 0.500022 |
| [GA271](../../experiments/GA01-GA300_static_guidance_screen/GA271.pt) | availability | joint | additive | 0.498980 |
| [GA272](../../experiments/GA01-GA300_static_guidance_screen/GA272.pt) | availability | joint | lowrank | 0.499839 |
| [GA273](../../experiments/GA01-GA300_static_guidance_screen/GA273.pt) | availability | joint | gated | 0.500186 |
| [GA274](../../experiments/GA01-GA300_static_guidance_screen/GA274.pt) | availability | joint | experts | 0.499101 |
| [GA275](../../experiments/GA01-GA300_static_guidance_screen/GA275.pt) | availability | joint | residual | 0.500232 |
| [GA276](../../experiments/GA01-GA300_static_guidance_screen/GA276.pt) | hybrid | point | additive | 0.499909 |
| [GA277](../../experiments/GA01-GA300_static_guidance_screen/GA277.pt) | hybrid | point | lowrank | 0.499697 |
| [GA278](../../experiments/GA01-GA300_static_guidance_screen/GA278.pt) | hybrid | point | gated | 0.500074 |
| [GA279](../../experiments/GA01-GA300_static_guidance_screen/GA279.pt) | hybrid | point | experts | 0.500027 |
| [GA280](../../experiments/GA01-GA300_static_guidance_screen/GA280.pt) | hybrid | point | residual | 0.499916 |
| [GA281](../../experiments/GA01-GA300_static_guidance_screen/GA281.pt) | hybrid | moments | additive | 0.500076 |
| [GA282](../../experiments/GA01-GA300_static_guidance_screen/GA282.pt) | hybrid | moments | lowrank | 0.499813 |
| [GA283](../../experiments/GA01-GA300_static_guidance_screen/GA283.pt) | hybrid | moments | gated | 0.500039 |
| [GA284](../../experiments/GA01-GA300_static_guidance_screen/GA284.pt) | hybrid | moments | experts | 0.500040 |
| [GA285](../../experiments/GA01-GA300_static_guidance_screen/GA285.pt) | hybrid | moments | residual | 0.499833 |
| [GA286](../../experiments/GA01-GA300_static_guidance_screen/GA286.pt) | hybrid | hurdle | additive | 0.499945 |
| [GA287](../../experiments/GA01-GA300_static_guidance_screen/GA287.pt) | hybrid | hurdle | lowrank | 0.499978 |
| [GA288](../../experiments/GA01-GA300_static_guidance_screen/GA288.pt) | hybrid | hurdle | gated | 0.500064 |
| [GA289](../../experiments/GA01-GA300_static_guidance_screen/GA289.pt) | hybrid | hurdle | experts | 0.499955 |
| [GA290](../../experiments/GA01-GA300_static_guidance_screen/GA290.pt) | hybrid | hurdle | residual | 0.500056 |
| [GA291](../../experiments/GA01-GA300_static_guidance_screen/GA291.pt) | hybrid | distribution | additive | 0.500154 |
| [GA292](../../experiments/GA01-GA300_static_guidance_screen/GA292.pt) | hybrid | distribution | lowrank | 0.500189 |
| [GA293](../../experiments/GA01-GA300_static_guidance_screen/GA293.pt) | hybrid | distribution | gated | 0.499976 |
| [GA294](../../experiments/GA01-GA300_static_guidance_screen/GA294.pt) | hybrid | distribution | experts | 0.499816 |
| [GA295](../../experiments/GA01-GA300_static_guidance_screen/GA295.pt) | hybrid | distribution | residual | 0.500192 |
| [GA296](../../experiments/GA01-GA300_static_guidance_screen/GA296.pt) | hybrid | joint | additive | 0.500120 |
| [GA297](../../experiments/GA01-GA300_static_guidance_screen/GA297.pt) | hybrid | joint | lowrank | 0.500026 |
| [GA298](../../experiments/GA01-GA300_static_guidance_screen/GA298.pt) | hybrid | joint | gated | 0.500130 |
| [GA299](../../experiments/GA01-GA300_static_guidance_screen/GA299.pt) | hybrid | joint | experts | 0.500322 |
| [GA300](../../experiments/GA01-GA300_static_guidance_screen/GA300.pt) | hybrid | joint | residual | 0.500204 |

## GA301〜GA392

過去0〜1 nsの平均距離残差の開発系列。診断・評価パネルの準備・比較も含む。台帳に結果、コード、保存された重みと関連配列を掲載。

| ID | 実験記録 | 台帳のファイル数 |
| --- | --- | ---: |
| GA301 | [static pair offset](../../experiments/GA301_static_pair_offset/README.md) | 29 |
| GA302 | [local relations](../../experiments/GA302_local_relations/README.md) | 17 |
| GA303 | [static training scale](../../experiments/GA303_static_training_scale/README.md) | 30 |
| GA304 | [static force geometry](../../experiments/GA304_static_force_geometry/README.md) | 12 |
| GA305 | [generated native pair offset](../../experiments/GA305_generated_native_pair_offset/README.md) | 20 |
| GA306 | [temporal readout](../../experiments/GA306_temporal_readout/README.md) | 3 |
| GA307 | [generated window length](../../experiments/GA307_generated_window_length/README.md) | 9 |
| GA308 | [dynamic candidate relations](../../experiments/GA308_dynamic_candidate_relations/README.md) | 3 |
| GA309 | [static force native blend](../../experiments/GA309_static_force_native_blend/README.md) | 6 |
| GA310 | [seed count diagnostic](../../experiments/GA310_seed_count_diagnostic/README.md) | 6 |
| GA311 | [replica expanded temporal models](../../experiments/GA311_replica_expanded_temporal_models/README.md) | 29 |
| GA312 | [eight seed confirmation](../../experiments/GA312_eight_seed_confirmation/README.md) | 12 |
| GA313 | [force field mixture](../../experiments/GA313_force_field_mixture/README.md) | 6 |
| GA314 | [seed aggregation](../../experiments/GA314_seed_aggregation/README.md) | 5 |
| GA315 | [observed future diagnostic](../../experiments/GA315_observed_future_diagnostic/README.md) | 2 |
| GA316 | [implicit force minimization](../../experiments/GA316_implicit_force_minimization/README.md) | 3 |
| GA317 | [static hydrogen positions](../../experiments/GA317_static_hydrogen_positions/README.md) | 6 |
| GA318 | [scaled static prior](../../experiments/GA318_scaled_static_prior/README.md) | 10 |
| GA319 | [adaptive force field fusion](../../experiments/GA319_adaptive_force_field_fusion/README.md) | 2 |
| GA320 | [physics variant screen](../../experiments/GA320_physics_variant_screen/README.md) | 6 |
| GA321 | [scaled mixed force field confirmation](../../experiments/GA321_scaled_mixed_force_field_confirmation/README.md) | 34 |
| GA322 | [train only fusion](../../experiments/GA322_train_only_fusion/README.md) | 2 |
| GA323 | [GA323の関連試行](../../experiments/GA320_physics_variant_screen/README.md) | 4 |
| GA324 | [trajectory budget](../../experiments/GA324_trajectory_budget/README.md) | 2 |
| GA325 | [GA325の関連試行](../../experiments/GA320_physics_variant_screen/README.md) | 4 |
| GA326 | [GA326の関連試行](../../experiments/GA320_physics_variant_screen/README.md) | 4 |
| GA327 | [symmetric training targets](../../experiments/GA327_symmetric_training_targets/README.md) | 10 |
| GA328 | [GA328の関連試行](../../experiments/GA320_physics_variant_screen/README.md) | 4 |
| GA329 | [neural static prior](../../experiments/GA329_neural_static_prior/README.md) | 18 |
| GA330 | [all available static prior](../../experiments/GA330_all_available_static_prior/README.md) | 10 |
| GA331 | [compatible atomic displacement](../../experiments/GA331_compatible_atomic_displacement/README.md) | 2 |
| GA332 | [expanded prior confirmation](../../experiments/GA332_expanded_prior_confirmation/README.md) | 8 |
| GA333 | [scaled short trajectory training](../../experiments/GA333_scaled_short_trajectory_training/README.md) | 85 |
| GA334 | [robust residual weighting](../../experiments/GA334_robust_residual_weighting/README.md) | 2 |
| GA335 | [directional solvent accessibility](../../experiments/GA335_directional_solvent_accessibility/README.md) | 6 |
| GA336 | [scaled local minicalc relations](../../experiments/GA336_scaled_local_minicalc_relations/README.md) | 15 |
| GA337 | [temperature response](../../experiments/GA337_temperature_response/README.md) | 9 |
| GA338 | [scaled dynamic ROD relations](../../experiments/GA338_scaled_dynamic_ROD_relations/README.md) | 11 |
| GA339 | [temperature response confirmation](../../experiments/GA339_temperature_response_confirmation/README.md) | 20 |
| GA340 | [short trajectory model combination](../../experiments/GA340_short_trajectory_model_combination/README.md) | 2 |
| GA341 | [dielectric response](../../experiments/GA341_dielectric_response/README.md) | 9 |
| GA342 | [replica denoised labels](../../experiments/GA342_replica_denoised_labels/README.md) | 8 |
| GA343 | [generalization diagnostic](../../experiments/GA343_generalization_diagnostic/README.md) | 2 |
| GA344 | [learned two force fields](../../experiments/GA344_learned_two_force_fields/development.json) | 46 |
| GA345 | [elastic velocity modes](../../experiments/GA345_elastic_velocity_modes/README.md) | 4 |
| GA346 | [other replica diagnostic](../../experiments/GA346_other_replica_diagnostic/README.md) | 2 |
| GA347 | [more training origins](../../experiments/GA347_more_training_origins/README.md) | 6 |
| GA348 | [two force field ensemble confirmation](../../experiments/GA348_two_force_field_ensemble_confirmation/README.md) | 6 |
| GA349 | 作業ツリーと対象ブランチに記録なし | — |
| GA350 | [many short charmm branches](../../experiments/GA350_many_short_charmm_branches/README.md) | 6 |
| GA351 | [sequence disjoint reserve](../../experiments/GA351_sequence_disjoint_reserve/README.md) | 30 |
| GA352 | [generated motion weight](../../experiments/GA352_generated_motion_weight/README.md) | 2 |
| GA353 | [relaxation readout](../../experiments/GA353_relaxation_readout/README.md) | 2 |
| GA354 | [ensemble short readout](../../experiments/GA354_ensemble_short_readout/README.md) | 2 |
| GA355 | [scaled two force field readout](../../experiments/GA355_scaled_two_force_field_readout/README.md) | 85 |
| GA356 | [regularized short readout](../../experiments/GA356_regularized_short_readout/README.md) | 2 |
| GA357 | [balanced dynamic readout](../../experiments/GA357_balanced_dynamic_readout/README.md) | 8 |
| GA358 | [conditional dynamic residual](../../experiments/GA358_conditional_dynamic_residual/README.md) | 8 |
| GA359 | [shared neural response](../../experiments/GA359_shared_neural_response/README.md) | 27 |
| GA360 | [multiple validation origins](../../experiments/GA360_multiple_validation_origins/README.md) | 26 |
| GA361 | [sequence embedding prior](../../experiments/GA361_sequence_embedding_prior/README.md) | 106 |
| GA362 | [matched structure control](../../experiments/GA362_matched_structure_control/README.md) | 7 |
| GA363 | [history auxiliary targets](../../experiments/GA363_history_auxiliary_targets/README.md) | 15 |
| GA364 | [cached later origin training](../../experiments/GA364_cached_later_origin_training/README.md) | 14 |
| GA365 | [expanded later training](../../experiments/GA365_expanded_later_training/README.md) | 65 |
| GA366 | [future motion by origin](../../experiments/GA366_future_motion_by_origin/README.md) | 2 |
| GA367 | [early motion blend](../../experiments/GA367_early_motion_blend/README.md) | 2 |
| GA368 | [early motion confirmation](../../experiments/GA368_early_motion_confirmation/README.md) | 26 |
| GA369 | [training only forward response](../../experiments/GA369_training_only_forward_response/README.md) | 15 |
| GA370 | [later origin embedding and future aux](../../experiments/GA370_later_origin_embedding_and_future_aux/README.md) | 3 |
| GA371 | [static distance gate](../../experiments/GA371_static_distance_gate/README.md) | 2 |
| GA372 | [matched solvent surface term](../../experiments/GA372_matched_solvent_surface_term/README.md) | 25 |
| GA373 | [two force field relations](../../experiments/GA373_two_force_field_relations/README.md) | 11 |
| GA374 | [snapshot conditioning](../../experiments/GA374_snapshot_conditioning/README.md) | 14 |
| GA375 | [no cutoff motion](../../experiments/GA375_no_cutoff_motion/README.md) | 14 |
| GA376 | [smooth kernel readout](../../experiments/GA376_smooth_kernel_readout/README.md) | 29 |
| GA377 | [generated local vectors](../../experiments/GA377_generated_local_vectors/README.md) | 75 |
| GA378 | [axial and transverse targets](../../experiments/GA378_axial_and_transverse_targets/README.md) | 5 |
| GA379 | [moving backbone frames](../../experiments/GA379_moving_backbone_frames/README.md) | 8 |
| GA380 | [snapshot force field reliability](../../experiments/GA380_snapshot_force_field_reliability/README.md) | 3 |
| GA381 | [neural local motion](../../experiments/GA381_neural_local_motion/README.md) | 14 |
| GA382 | [motion consistency](../../experiments/GA382_motion_consistency/development.json) | 2 |
| GA383 | [motion consistency](../../experiments/GA383_motion_consistency/development.json) | 6 |
| GA384 | [neural response confirmation](../../experiments/GA384_neural_response_confirmation/README.md) | 30 |
| GA385 | [GA385の関連試行](../../experiments/GA386_later_origin_neural_training/README.md) | 53 |
| GA386 | [later origin neural training](../../experiments/GA386_later_origin_neural_training/README.md) | 26 |
| GA387 | [protein group weighting](../../experiments/GA387_protein_group_weighting/README.md) | 18 |
| GA388 | [confirmation distance diagnostics](../../experiments/GA388_confirmation_distance_diagnostics/README.md) | 2 |
| GA389 | [smooth distance experts](../../experiments/GA389_smooth_distance_experts/README.md) | 20 |
| GA390 | [angular hbond response](../../experiments/GA390_angular_hbond_response/development.json) | 8 |
| GA391 | [physical hydrogen masses](../../experiments/GA391_physical_hydrogen_masses/development.json) | 14 |
| GA392 | [expanded later origin training](../../experiments/GA392_expanded_later_origin_training/README.md) | 225 |
