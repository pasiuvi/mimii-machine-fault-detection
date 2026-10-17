# Q3 decision record – Round 1 plan (written BEFORE any Round 1 run)

**Status:** accepted · **Data used for this decision:** validation machine id_04 only · **Test set (id_06):** never used

## 1. What Round 0 showed (archived in `experiments/archive/round0_*`)
| Run | What | Val pooled TPR@FPR5 | Pump | Fan | Observation |
|---|---|---|---|---|---|
| E000 | Loudness only (trivial) | 0.578 | – | – | strong because loudness differs on the val machine |
| B001 | MFCC + LogReg (best of 30) | 0.587 | – | – | the classical model was ahead |
| E001 | Head-only ANN | 0.074 | 0.170 | 0.385 | pooled score far below both per-type scores |
| E002 | Full fine-tuning, no aug | 0.368 | 0.240 | 0.419 | train loss 0.002, val loss 3.7–9.2 → over-fitting |
| E003 | + Q2 augmentation | 0.288 | 0.050 | 0.376 | val metric jumped 0.00 ↔ 0.37 between epochs |

## 2. Problems and decisions
| # | Problem (evidence) | Decision | Why this is legitimate |
|---|---|---|---|
| 1 | Pooled ranking mixes pumps and fans whose scores sit on different scales (E001: 0.17 / 0.39 per type but 0.07 pooled) | **Primary metric → macro TPR@FPR5** = mean of pump and fan TPR@FPR5, i.e. **one alarm threshold per machine type**, chosen on validation. Pooled TPR@FPR5 (Q1 definition) is still reported as a secondary metric. | The engineer always knows which machine type a recording comes from, so a per-type threshold is what the deployed app (Q4) does. Decided on validation, before any test use; applied to **every** model (trivial, baseline, ANN). |
| 2 | Over-fitting to the 2 training machines (E002) | Add ANN regularisation aimed at an *unseen* machine: lower LR 1e-4, weight decay 0.05, **EMA of weights**, **mixup within machine type** | Standard transfer-learning practice for small, shifted data. Each is added **one step at a time** (E004, E005) and removed again in ablations (E017–E019). |
| 3 | Noisy validation metric → the "best epoch" is partly luck (E003) | EMA smooths the weights; model selection uses the **stable score** = mean of the best epoch and its two neighbours | Penalises one-epoch spikes; rule fixed here, before the runs. |
| 4 | Domain shift between machines | One explicit experiment: per-clip band-mean removal (E020) | Outcome uncertain (it may also remove the steady fault tone, Q1 Finding 2) → tested, not assumed. |

## 3. Unchanged
Same split (train id_00 + id_02 · val id_04 · test id_06), same features and cache, same FaultCNN architecture (Q2),
same 30-configuration baseline search, at least 3 hyperparameters (LR, batch size, depth), 3 optimisers, designed failures,
3 seeds for the final model, one-time test evaluation with a lock file.

## 4. Pre-registered prediction
`reports/Q2_prediction.md` (commits `30e09e6`, `ced01a4`) is **not edited**. It was written for the pooled metric,
which is still reported, so the prediction can be checked exactly as written. The macro metric is reported next to it.
