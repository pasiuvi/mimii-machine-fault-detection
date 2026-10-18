# Q5 – explanation analysis (deployed model E026_s43)

Built 2026-10-18T10:51:49 (commit 86c816d). Validation machine id_04 for all choices; test clips only described.

## 1. Methods compared on validation clips (8 x 20 cell grid, hide 20 %)

| method               |   clips |   drop_top |   drop_random |   gap |   pass_rate |   rho_vs_occlusion |   ms_per_clip |   empty_maps |
|:---------------------|--------:|-----------:|--------------:|------:|------------:|-------------------:|--------------:|-------------:|
| Grad-CAM features.8  |     100 |      0.356 |         0.151 | 0.205 |       0.59  |              0.07  |        33.284 |            0 |
| Grad-CAM features.5  |     100 |      0.685 |         0.151 | 0.534 |       0.76  |              0.019 |        33.519 |            0 |
| Occlusion            |      24 |      1.571 |         0.285 | 1.286 |       1     |            nan     |       210.959 |            0 |
| Integrated Gradients |      24 |      0.485 |         0.285 | 0.2   |       0.542 |              0.2   |       146.211 |            0 |

gap = logit drop when the most highlighted 20 % is hidden minus mean drop for 10 random 20 % masks; pass_rate = share of clips where the highlighted area mattered more than random; rho_vs_occlusion = Spearman agreement with the occlusion map.

**Layer rule -> app uses features.5** (mean gap on the 100 clips where both layers give a map: {'features.8': 0.205, 'features.5': 0.534}). An empty map means no region pushed the fault score up (common for clearly normal clips).

## 2. Deletion curves

| method              |   fraction |   drop_top |   drop_random |
|:--------------------|-----------:|-----------:|--------------:|
| Grad-CAM features.5 |       0.05 |      0.152 |         0.131 |
| Grad-CAM features.5 |       0.1  |      0.314 |         0.209 |
| Grad-CAM features.5 |       0.2  |      0.685 |         0.102 |
| Grad-CAM features.5 |       0.3  |      1.002 |         0.326 |
| Grad-CAM features.5 |       0.5  |      1.121 |         0.093 |
| Grad-CAM features.8 |       0.05 |      0.162 |         0.131 |
| Grad-CAM features.8 |       0.1  |      0.261 |         0.209 |
| Grad-CAM features.8 |       0.2  |      0.356 |         0.102 |
| Grad-CAM features.8 |       0.3  |      0.373 |         0.326 |
| Grad-CAM features.8 |       0.5  |     -0.329 |         0.093 |

## 3. Model-randomisation sanity check (Spearman with the original map)

| randomised                                  |   mean |   std |   count |
|:--------------------------------------------|-------:|------:|--------:|
| head                                        | -0.093 | 0.264 |      24 |
| head + features.8                           |  0.138 | 0.331 |      24 |
| head + features.8 + features.7              | -0.133 | 0.324 |      24 |
| head + features.8 + features.7 + features.6 | -0.419 | 0.175 |      24 |

## 4. Average Grad-CAM frequency profile (validation)

| band         |   ('pump', 'normal') |   ('pump', 'abnormal') |   ('fan', 'normal') |   ('fan', 'abnormal') |
|:-------------|---------------------:|-----------------------:|--------------------:|----------------------:|
| 0-100 Hz     |                0.004 |                  0.009 |               0     |                 0.03  |
| 100-300 Hz   |                0.014 |                  0.016 |               0.006 |                 0.052 |
| 300-600 Hz   |                0.109 |                  0.048 |               0.062 |                 0.13  |
| 600-1000 Hz  |                0.128 |                  0.082 |               0.232 |                 0.234 |
| 1000-2000 Hz |                0.329 |                  0.234 |               0.189 |                 0.174 |
| 2000-4000 Hz |                0.241 |                  0.287 |               0.168 |                 0.132 |
| 4000-8000 Hz |                0.175 |                  0.324 |               0.342 |                 0.249 |

## 5. Case studies (demo clips + hardest errors)

| source        | file         | machine_type   | true_label   | q3_result    |   score |   threshold | verdict   | correct   | ood             | seed_votes   | top_band   |   top_band_share | time             | matches_q1_band   |   score_top_hidden |   score_random_hidden |   logit_drop_top |   logit_drop_random | check_passed   |
|:--------------|:-------------|:---------------|:-------------|:-------------|--------:|------------:|:----------|:----------|:----------------|:-------------|:-----------|-----------------:|:-----------------|:------------------|-------------------:|----------------------:|-----------------:|--------------------:|:---------------|
| demo          | 00000587.wav | pump           | normal       | normal       |  0.9728 |      0.9904 | no_alarm  | True      | in_distribution | 0/3          | 4–8 kHz    |            0.325 | spread over clip | False             |             0.9512 |                0.9572 |            0.608 |               0.421 | True           |
| demo          | 00000081.wav | pump           | abnormal     | abnormal     |  0.9921 |      0.9904 | alarm     | True      | unusual         | 1/3          | 1–2 kHz    |            0.327 | spread over clip | False             |             0.9943 |                0.9936 |           -0.323 |              -0.217 | False          |
| demo          | 00000058.wav | pump           | abnormal     | missed fault |  0.9755 |      0.9904 | no_alarm  | False     | in_distribution | 0/3          | 300–600 Hz |            0.223 | spread over clip | True              |             0.8529 |                0.9763 |            1.926 |              -0.09  | True           |
| demo          | 00000731.wav | pump           | normal       | false alarm  |  0.9939 |      0.9904 | alarm     | False     | in_distribution | 1/3          | 1–2 kHz    |            0.236 | spread over clip | False             |             0.9583 |                0.9919 |            1.958 |               0.252 | True           |
| demo          | 00000459.wav | fan            | normal       | normal       |  0.7385 |      0.8151 | no_alarm  | True      | in_distribution | 2/3          | 1–2 kHz    |            0.34  | spread over clip | True              |             0.7926 |                0.5971 |           -0.303 |               0.63  | False          |
| demo          | 00000177.wav | fan            | abnormal     | abnormal     |  0.9709 |      0.8151 | alarm     | True      | unusual         | 3/3          | 1–2 kHz    |            0.291 | spread over clip | True              |             0.97   |                0.9528 |            0.031 |               0.485 | False          |
| demo          | 00000063.wav | fan            | abnormal     | missed fault |  0.755  |      0.8151 | no_alarm  | False     | unusual         | 2/3          | 4–8 kHz    |            0.33  | spread over clip | False             |             0.5896 |                0.8477 |            0.763 |              -0.658 | True           |
| demo          | 00000083.wav | fan            | normal       | false alarm  |  0.9696 |      0.8151 | alarm     | False     | unusual         | 3/3          | 1–2 kHz    |            0.276 | spread over clip | True              |             0.9612 |                0.9693 |            0.253 |              -0.005 | True           |
| hardest error | 00000002.wav | pump           | abnormal     | missed fault |  0.4833 |      0.9904 | no_alarm  | False     | unusual         | 0/3          | 4–8 kHz    |            0.525 | spread over clip | False             |             0.3793 |                0.28   |            0.427 |               1.015 | False          |
| hardest error | 00000322.wav | fan            | abnormal     | missed fault |  0.5507 |      0.8151 | no_alarm  | False     | unusual         | 1/3          | 2–4 kHz    |            0.308 | spread over clip | True              |             0.3799 |                0.6432 |            0.694 |              -0.401 | True           |
| hardest error | 00000035.wav | fan            | abnormal     | missed fault |  0.616  |      0.8151 | no_alarm  | False     | unusual         | 1/3          | 1–2 kHz    |            0.346 | spread over clip | True              |             0.6869 |                0.5481 |           -0.312 |               0.258 | False          |
