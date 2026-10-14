# Pre-registered prediction (Question 2b)

**Written and committed to Git BEFORE any model was trained or evaluated.**
The commit date of this file is the evidence that the prediction came first.

## Which model family do I expect to score higher on the primary metric?

**Primary metric:** recall at a 5 % false-alarm rate (TPR @ FPR = 0.05) on the held-out test machines (fan id_06 + pump id_06).

**Prediction:** I expect the **fine-tuned CNN (ANN)** to score **higher than the classical baseline** (MFCC statistics + tuned Random Forest / SVM / logistic regression), but **only by a small margin**. I expect the 95 % bootstrap confidence intervals of the two models to overlap. I also think it is quite possible that the baseline wins on **one machine type**, most likely the fan.

## Why

**Reasons the ANN should be ahead**
1. **Low false-alarm region.** The metric only counts faults caught while false alarms stay at 5 %. This rewards models that rank *subtle* faults above all normal clips. A CNN that sees the full time–frequency picture can use patterns such as tonal lines and the shape of the spectrum, which a set of MFCC averages compresses away.
2. **Generalising to an unseen machine.** Finding 1 showed that the machine's identity dominates the sound. ImageNet-pretrained filters, random gain and SpecAugment should reduce how much the CNN relies on machine-specific details. The baseline gets no such help.
3. **Different fault signatures per machine type** (Finding 2: pumps louder at 431–910 Hz; fans with more low-frequency and less mid-frequency energy). A non-linear model with a machine-type input can learn both patterns.

**Reasons the margin should be small, or reversed**
1. **Faults are steady, not transient** (Finding 2, all 5,058 training clips: abnormal clips are *less* spiky than normal ones). Steady changes survive averaging, so clip-level MFCC statistics capture much of the signal.
2. **Few training machines.** There are only two machines per type in training. A 4.2 M-parameter network can still over-fit those machines, while a regularised classical model with fewer degrees of freedom may transfer just as well.
3. **Loudness alone already gives validation AUC 0.796.** A baseline can exploit level differences easily. I deliberately weaken this cue for the CNN with ±6 dB gain augmentation, which could cost the CNN some score on machines where loudness happens to work.

## What would change my mind

- If the baseline is ahead on **both** machine types with non-overlapping intervals, my assumption that time–frequency structure matters beyond steady averages is wrong for this data. I will report that honestly (brief section 6).
- If the CNN is far ahead on validation but not on test, the gain came from fitting the validation machine rather than learning the fault.
