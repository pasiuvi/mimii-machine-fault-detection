# Q3 extra analysis (deployed final ANN, test machine id_06)

Computed from reports/q3_test_predictions.csv - no model was evaluated again.

## Recall at 1 %, 5 % and 10 % false-alarm rate, and ranking quality
| machine   |   recall@1%FPR |   recall@5%FPR |   recall@10%FPR |   ROC-AUC |   median score normal |   median score abnormal |
|:----------|---------------:|---------------:|----------------:|----------:|----------------------:|------------------------:|
| pump      |          0     |          0     |           0     |     0.443 |                 0.981 |                   0.976 |
| fan       |          0.014 |          0.064 |           0.133 |     0.479 |                 0.969 |                   0.97  |

## Loudness relative to the median normal clip of the same machine (data-quality check)
| machine_type   |   label |   clips |   share_more_than_3dB_away |   median_rel_db |
|:---------------|--------:|--------:|---------------------------:|----------------:|
| fan            |       0 |    1015 |                      0     |           0     |
| fan            |       1 |     361 |                      1     |          -4.666 |
| pump           |       0 |    1036 |                      0.013 |           0     |
| pump           |       1 |     102 |                      0.48  |           2.802 |

Diagnostic only (uses test labels, never used to build or choose a model): ROC-AUC of |level difference| = pump 0.716, fan 1.000

## Overlap of detected faults with the classical baseline (at validation thresholds)
both detect 323, ANN only 31, baseline only 41, both miss 68 (of 463 abnormal clips)