# Q3 decision record – Round 2: combine the one-factor winners (written BEFORE any Round 2 run)

**Status:** accepted · **Data used:** validation machine id_04 only · **Test set (id_06):** never used

## 1. Why
Round 1 (E001–E023) changed **one setting at a time** around the reference E005. The winners were never tried together:

| Factor | Winner | Stable val macro TPR@FPR5 | Reference E005 |
|---|---|---|---|
| Optimiser | SGD-Nesterov, LR 1e-2 (E015) | **0.435** | 0.234 |
| Learning rate (AdamW) | 1e-3 (E008) | 0.362 | 0.234 |
| Fine-tuning depth | last 3 of 9 blocks (E012) | 0.358 | 0.234 |
| Batch size | 16 (E009) | 0.339 | 0.234 |
| EMA ablation | without EMA (E017) | 0.288 | 0.234 |

Factors can interact, so combining them is a standard next step. It's limited to **4 runs**, so total ANN effort
(27 development runs) stays the same order of magnitude as the 30 baseline configurations.

## 2. Plan (fixed before running)
| Run | Combination | Hypothesis |
|---|---|---|
| E024 | SGD 1e-2 + last 3 blocks | best optimiser + best depth add up |
| E025 | SGD 1e-2 + batch 16 | more updates help, or overshoot |
| E026 | SGD 1e-2 without EMA | EMA held E005 back; does it also hold SGD back? |
| E027 | AdamW 1e-3 + last 3 blocks | the depth gain also holds at a larger LR |

## 3. Selection rule (unchanged from Round 1)
The final design is the run with the highest **stable** validation macro TPR@FPR5 among **all** E002–E027,
then re-trained with seeds 43 and 44. If no combination beats E015, E015 is selected. The negative result is still reported.

## 4. What does not change
Split, features, architecture, metric, baseline, the one-time locked test evaluation.
