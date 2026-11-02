# AcoustiGuard v2 – one-time evaluation on the test machines id_06

Configuration chosen on the 6 development machines and tagged v2-before-test before this run. id_06 was already used for the locked v1 evaluations, so this is not a blind test of a new machine; nothing was tuned on it. 2-of-3 numbers assume independent clips.

## Mean of pump and fan

| approach                                                |   TPR@FPR5 |   AUC |   clip_recall |   clip_FA_day |   recall_2of3 |   FA_day_2of3 |   recall_2of3_sd |   FA_day_2of3_sd |
|:--------------------------------------------------------|-----------:|------:|--------------:|--------------:|--------------:|--------------:|-----------------:|-----------------:|
| v2 selected (clap, n=50, k=3, c=2)                      |      0.774 | 0.948 |         0.732 |          6.19 |         0.812 |         0.858 |            0.083 |            0.558 |
| reference: fine-tuned FaultCNN embedding, same protocol |      0.564 | 0.896 |         0.558 |          7.07 |         0.568 |         1.123 |            0.049 |            0.703 |

## Per machine (mean of 5 enrolment draws)

| machine    | approach                                                |   TPR@FPR5 |   AUC |   clip_recall |   clip_FA_day |   recall_2of3 |   FA_day_2of3 |   recall_2of3_sd |   FA_day_2of3_sd |
|:-----------|:--------------------------------------------------------|-----------:|------:|--------------:|--------------:|--------------:|--------------:|-----------------:|-----------------:|
| fan id_06  | v2 selected (clap, n=50, k=3, c=2)                      |      0.712 | 0.925 |         0.712 |         6.655 |         0.8   |         0.97  |            0.004 |            0.597 |
| fan id_06  | reference: fine-tuned FaultCNN embedding, same protocol |      0.823 | 0.97  |         0.784 |         5.581 |         0.878 |         0.682 |            0.036 |            0.486 |
| pump id_06 | v2 selected (clap, n=50, k=3, c=2)                      |      0.835 | 0.97  |         0.751 |         5.725 |         0.824 |         0.747 |            0.162 |            0.518 |
| pump id_06 | reference: fine-tuned FaultCNN embedding, same protocol |      0.304 | 0.822 |         0.331 |         8.558 |         0.258 |         1.564 |            0.062 |            0.92  |

Locked v1 for comparison: final ANN 0.032 TPR@FPR5; post-hoc CLAP enrolment (20 consecutive clips) 0.622 with 70.8 false alarms per day.
