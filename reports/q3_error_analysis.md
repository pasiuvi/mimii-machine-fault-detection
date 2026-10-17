# Error analysis – deployed final ANN on the test machine (id_06)

Thresholds chosen on the validation machine for 5 % false alarms, one per machine type: **pump 0.9904** (test TPR 0.039, FPR 0.241), **fan 0.8151** (test TPR 0.970, FPR 0.999)

## 1. Confusion by machine type (at the validation thresholds)
| machine           |   TP |   FN |   FP |   TN |   recall |   false-alarm rate |
|:------------------|-----:|-----:|-----:|-----:|---------:|-------------------:|
| pump (test id_06) |    4 |   98 |  250 |  786 |    0.039 |              0.241 |
| fan (test id_06)  |  350 |   11 | 1014 |    1 |    0.97  |              0.999 |

Note: the false-alarm rate on test can differ from 5 % because the threshold was set on a different machine (id_04). This gap is the real-world cost of machine-to-machine shift.

## 2. Errors by loudness (relative to the median normal clip of the same machine)
| machine   | abnormal clips      |   n |   recall (final ANN) |   recall (baseline) |
|:----------|:--------------------|----:|---------------------:|--------------------:|
| pump      | quieter than normal |  28 |                0.036 |               0.036 |
| pump      | louder than normal  |  74 |                0.041 |               0.392 |
| fan       | quieter than normal | 255 |                0.957 |               0.902 |
| fan       | louder than normal  | 106 |                1     |               0.981 |

By loudness quartile (all test clips):
| rms_db      |   n |   abnormal |   recall |   false-alarm rate |
|:------------|----:|-----------:|---------:|-------------------:|
| Q1 quietest | 629 |        282 |    0.869 |              0.13  |
| Q2          | 628 |         12 |    0.25  |              0.537 |
| Q3          | 628 |          3 |    0     |              0.965 |
| Q4 loudest  | 629 |        166 |    0.639 |              0.616 |

## 3. Overlap of missed faults with the classical baseline
| pred        |   baseline detects |   baseline misses |
|:------------|-------------------:|------------------:|
| ANN detects |                323 |                31 |
| ANN misses  |                 41 |                68 |

## 4. Hardest errors
20 clips saved to reports/q3_hardest_errors.csv (lowest-scored missed faults and highest-scored false alarms) for listening and for Q5 (XAI).