# Q6 – CLAP on the held-out test machine (id_06), evaluated once

2026-10-29T22:32:56 · laion/clap-htsat-unfused rev unknown · selections from experiments/q6_config.json

## Comparison with Question 3 (same test clips, same metric, thresholds from validation)

| Model                           |   Test macro TPR@FPR5 | 95% CI      |   Pump TPR@FPR5 |   Fan TPR@FPR5 |   Test pooled TPR@FPR5 |   Test macro ROC-AUC |   Test macro pAUC(0.1) |   Val macro TPR@FPR5 |
|:--------------------------------|----------------------:|:------------|----------------:|---------------:|-----------------------:|---------------------:|-----------------------:|---------------------:|
| Random scores (reference)       |                0.05   | -           |          0.05   |         0.05   |                 0.05   |               0.5    |                 0.5    |               0.05   |
| Trivial: loudness only          |                0.3919 | 0.337–0.442 |          0.4902 |         0.2936 |                 0.3369 |               0.5002 |                 0.6782 |               0.475  |
| Classical baseline (SVM-RBF)    |                0.2433 | 0.183–0.331 |          0.1569 |         0.3296 |                 0.108  |               0.7801 |                 0.6008 |               0.5275 |
| Initial ANN (E001)              |                0.3638 | 0.306–0.426 |          0.4118 |         0.3158 |                 0.3413 |               0.7882 |                 0.6457 |               0.2776 |
| Final ANN (E026_s43, deployed)  |                0.0319 | 0.019–0.046 |          0      |         0.0637 |                 0.0022 |               0.461  |                 0.4908 |               0.5107 |
| Final ANN, mean over 3 seeds    |                0.1058 | sd 0.080    |          0.134  |         0.0776 |                 0.0612 |               0.5007 |                 0.5278 |               0.4411 |
| CLAP zero-shot (P4_fault_types) |                0.2114 | 0.162–0.266 |          0.4118 |         0.0111 |                 0.0194 |               0.4666 |                 0.5817 |               0.1137 |
| CLAP linear probe (C=0.1)       |                0.4451 | 0.410–0.488 |          0.8431 |         0.0471 |                 0.1987 |               0.7399 |                 0.6987 |               0.2305 |
| CLAP normal-only kNN (k=1)      |                0.3487 | 0.293–0.403 |          0.6863 |         0.0111 |                 0.1728 |               0.5887 |                 0.6406 |               0.1295 |

## Operating points (validation threshold applied to test)

- **CLAP zero-shot (P4_fault_types)**: pump: TPR 0.0, FPR 0.0; fan: TPR 0.2964, FPR 0.998
- **CLAP linear probe (C=0.1)**: pump: TPR 0.5392, FPR 0.0077; fan: TPR 0.8338, FPR 0.8404
- **CLAP normal-only kNN (k=1)**: pump: TPR 0.2157, FPR 0.0029; fan: TPR 0.3047, FPR 0.8236

## Extra (6.b opportunity): enrolment with 20 normal clips of the same machine

Not comparable with the table above: the 20 reference clips per machine are removed from the test set.

| setting                                    |   clips_scored |   test_macro_TPR@FPR5 |   pump |    fan |   test_macro_AUC |   val_macro_TPR@FPR5 |
|:-------------------------------------------|---------------:|----------------------:|-------:|-------:|-----------------:|---------------------:|
| k=3, 20 normal reference clips per machine |           2474 |                 0.622 | 0.5294 | 0.7147 |           0.8936 |               0.5024 |

## Cost and latency

|   CLAP latency per clip (ms), device |   CLAP latency CPU (ms) |   CLAP model only (ms), device | device   |   params_total_M |   params_audio_M |
|-------------------------------------:|------------------------:|-------------------------------:|:---------|-----------------:|-----------------:|
|                                 79.7 |                   255.5 |                          110.3 | cuda     |            153.5 |             28.2 |
