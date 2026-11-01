# Post-hoc study: per-machine calibration and normal-only detection

Run AFTER the locked Q3/Q6 test evaluations; rules fixed in scripts/posthoc_calibration.py before running. Enrolment = first 20 normal clips of each machine; threshold = highest enrolment score; k = 3; false alarms per day assume one clip every 10 minutes (144 clips).

## Test machines (id_06), mean of pump and fan

| approach                                      |   machines |   TPR_at_FPR5 |   AUC |   recall |   FPR |   false_alarms_per_day |
|:----------------------------------------------|-----------:|--------------:|------:|---------:|------:|-----------------------:|
| ANN score, validation threshold (as deployed) |          2 |         0.032 | 0.466 |    0.504 | 0.617 |                  88.95 |
| ANN score, per-machine threshold              |          2 |         0.032 | 0.466 |    0.007 | 0.02  |                   2.9  |
| ANN embedding kNN (enrolment)                 |          2 |         0.437 | 0.825 |    0.758 | 0.408 |                  58.8  |
| Loudness deviation (enrolment)                |          2 |         0.524 | 0.792 |    0.951 | 0.87  |                 125.2  |
| CLAP kNN (enrolment)                          |          2 |         0.622 | 0.894 |    0.858 | 0.492 |                  70.8  |

## All machines each approach can be evaluated on (ANN: id_04 + id_06 only)

| approach                                      |   machines |   TPR_at_FPR5 |   AUC |   recall |   FPR |   false_alarms_per_day |
|:----------------------------------------------|-----------:|--------------:|------:|---------:|------:|-----------------------:|
| Loudness deviation (enrolment)                |          8 |         0.566 | 0.793 |    0.846 | 0.629 |                 90.562 |
| CLAP kNN (enrolment)                          |          8 |         0.584 | 0.868 |    0.707 | 0.404 |                 58.15  |
| ANN score, validation threshold (as deployed) |          4 |         0.271 | 0.558 |    0.507 | 0.332 |                 47.925 |
| ANN score, per-machine threshold              |          4 |         0.271 | 0.558 |    0.25  | 0.02  |                  2.9   |
| ANN embedding kNN (enrolment)                 |          4 |         0.644 | 0.89  |    0.879 | 0.466 |                 67.05  |

## Every machine

| machine    | machine_type   | role               | approach                                      |   TPR@FPR5 |   AUC |   recall_at_enrol_thr |   FPR_at_enrol_thr |   false_alarms_per_day |   scored |   faults |
|:-----------|:---------------|:-------------------|:----------------------------------------------|-----------:|------:|----------------------:|-------------------:|-----------------------:|---------:|---------:|
| fan id_00  | fan            | train machine      | Loudness deviation (enrolment)                |      0.545 | 0.803 |                 0.74  |              0.339 |                   48.8 |     1398 |      407 |
| fan id_00  | fan            | train machine      | CLAP kNN (enrolment)                          |      0.088 | 0.694 |                 0.13  |              0.084 |                   12.1 |     1398 |      407 |
| fan id_02  | fan            | train machine      | Loudness deviation (enrolment)                |      0.663 | 0.773 |                 0.671 |              0.137 |                   19.7 |     1355 |      359 |
| fan id_02  | fan            | train machine      | CLAP kNN (enrolment)                          |      0.507 | 0.803 |                 0.63  |              0.136 |                   19.5 |     1355 |      359 |
| fan id_04  | fan            | validation machine | ANN score, validation threshold (as deployed) |      0.991 | 0.997 |                 0.991 |              0.047 |                    6.8 |     1361 |      348 |
| fan id_04  | fan            | validation machine | ANN score, per-machine threshold              |      0.991 | 0.997 |                 0.986 |              0.035 |                    5   |     1361 |      348 |
| fan id_04  | fan            | validation machine | ANN embedding kNN (enrolment)                 |      1     | 0.998 |                 1     |              0.163 |                   23.5 |     1361 |      348 |
| fan id_04  | fan            | validation machine | Loudness deviation (enrolment)                |      0.684 | 0.781 |                 0.753 |              0.53  |                   76.3 |     1361 |      348 |
| fan id_04  | fan            | validation machine | CLAP kNN (enrolment)                          |      0.345 | 0.782 |                 0.195 |              0.028 |                    4   |     1361 |      348 |
| fan id_06  | fan            | TEST machine       | ANN score, validation threshold (as deployed) |      0.064 | 0.483 |                 0.97  |              0.999 |                  143.9 |     1356 |      361 |
| fan id_06  | fan            | TEST machine       | ANN score, per-machine threshold              |      0.064 | 0.483 |                 0.014 |              0.002 |                    0.3 |     1356 |      361 |
| fan id_06  | fan            | TEST machine       | ANN embedding kNN (enrolment)                 |      0.776 | 0.945 |                 0.515 |              0.001 |                    0.1 |     1356 |      361 |
| fan id_06  | fan            | TEST machine       | Loudness deviation (enrolment)                |      1     | 1     |                 1     |              0.931 |                  134   |     1356 |      361 |
| fan id_06  | fan            | TEST machine       | CLAP kNN (enrolment)                          |      0.715 | 0.881 |                 0.717 |              0.061 |                    8.8 |     1356 |      361 |
| pump id_00 | pump           | train machine      | Loudness deviation (enrolment)                |      0.86  | 0.959 |                 0.993 |              0.795 |                  114.5 |     1129 |      143 |
| pump id_00 | pump           | train machine      | CLAP kNN (enrolment)                          |      0.839 | 0.949 |                 0.993 |              0.715 |                  103   |     1129 |      143 |
| pump id_02 | pump           | train machine      | Loudness deviation (enrolment)                |      0.55  | 0.912 |                 0.991 |              0.511 |                   73.5 |     1096 |      111 |
| pump id_02 | pump           | train machine      | CLAP kNN (enrolment)                          |      0.991 | 0.996 |                 1     |              0.55  |                   79.2 |     1096 |      111 |
| pump id_04 | pump           | validation machine | ANN score, validation threshold (as deployed) |      0.03  | 0.302 |                 0.03  |              0.048 |                    7   |      782 |      100 |
| pump id_04 | pump           | validation machine | ANN score, per-machine threshold              |      0.03  | 0.302 |                 0     |              0.006 |                    0.8 |      782 |      100 |
| pump id_04 | pump           | validation machine | ANN embedding kNN (enrolment)                 |      0.7   | 0.91  |                 1     |              0.883 |                  127.1 |      782 |      100 |
| pump id_04 | pump           | validation machine | Loudness deviation (enrolment)                |      0.18  | 0.533 |                 0.72  |              0.981 |                  141.3 |      782 |      100 |
| pump id_04 | pump           | validation machine | CLAP kNN (enrolment)                          |      0.66  | 0.929 |                 0.99  |              0.735 |                  105.8 |      782 |      100 |
| pump id_06 | pump           | TEST machine       | ANN score, validation threshold (as deployed) |      0     | 0.448 |                 0.039 |              0.236 |                   34   |     1118 |      102 |
| pump id_06 | pump           | TEST machine       | ANN score, per-machine threshold              |      0     | 0.448 |                 0     |              0.038 |                    5.5 |     1118 |      102 |
| pump id_06 | pump           | TEST machine       | ANN embedding kNN (enrolment)                 |      0.098 | 0.705 |                 1     |              0.816 |                  117.5 |     1118 |      102 |
| pump id_06 | pump           | TEST machine       | Loudness deviation (enrolment)                |      0.049 | 0.584 |                 0.902 |              0.808 |                  116.4 |     1118 |      102 |
| pump id_06 | pump           | TEST machine       | CLAP kNN (enrolment)                          |      0.529 | 0.907 |                 1     |              0.922 |                  132.8 |     1118 |      102 |
