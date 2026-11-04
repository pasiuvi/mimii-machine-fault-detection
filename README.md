# AcoustiGuard – Acoustic Fault Screening for Industrial Pumps and Fans

An end-to-end deep-learning project that listens to 10-second recordings of pumps and fans and helps a **plant maintenance engineer** decide whether to raise an inspection work order. It includes a web app with honest uncertainty, live-checked Grad-CAM explanations, a comparison with an audio-language foundation model (CLAP), and **AcoustiGuard v2**: a per-machine enrolment detector that compares each machine with its own normal sound.

> MSc Data Science (NIBM / Coventry University) – AI coursework · Author: **B. D. P. A. Karunarathna (Don Ashan)**, COMScDS251P-001

> **Status.** v1 (one classifier for all machines) failed on the held-out test machine (macro recall 3.2 % at a 5 % false-alarm rate) and must not be used on new machines. **v2** (per-machine enrolment, chosen on 6 development machines, tested once) caught **81 %** of faults on the test machines at about **0.9 false alarms per machine per day** (2-of-3 rule). v2 is ready for a supervised shadow-mode pilot, **not for production**. See [Results](#results) and [Limitations](#limitations-and-what-production-would-need).

| | |
|---|---|
| User / decision | Plant maintenance engineer: inspect now, or wait for the scheduled service |
| Data | MIMII (Purohit et al., 2019, CC BY-SA 4.0): pump + fan, 6 dB SNR, 16 kHz, microphone channel 0 |
| Split | By **machine ID** (no leakage): train id_00 + id_02 · validation id_04 · **test id_06** (used once, locked) |
| Model | FaultCNN: ImageNet-pretrained EfficientNet-B0 on 128-band log-mel spectrograms + machine-type input (4.17 M parameters) |
| Primary metric | **Macro TPR@FPR5**: fault recall at a 5 % false-alarm rate, threshold per machine type, set on validation |
| Web app | FastAPI + HTML/JS (`app/`): v1 single-clip analysis (`/`) and v2 machine monitoring with enrolment (`/monitor`) |
| XAI | Grad-CAM (layer chosen on validation) with a live deletion check in the app (`app/xai.py`) |
| Foundation model | LAION CLAP `laion/clap-htsat-unfused` @ `8fa0f1c…`: zero-shot, linear probe, normal-only kNN, enrolment |

## Results
Test machine id_06, same clips and metric for every model (95 % bootstrap CI). Full tables: `reports/q3_comparison.md`, `reports/q6_results.md`.

| Model | Test macro TPR@FPR5 | Pump | Fan |
|---|---|---|---|
| Loudness only (trivial) | 0.392 (0.337–0.442) | 0.490 | 0.294 |
| Classical baseline (MFCC + SVM) | 0.243 (0.183–0.331) | 0.157 | 0.330 |
| Initial ANN (E001) | 0.364 (0.306–0.426) | 0.412 | 0.316 |
| **Final ANN (E026_s43, deployed)** | **0.032 (0.019–0.046)** | 0.000 | 0.064 |
| CLAP zero-shot (prompts chosen on validation) | 0.211 (0.162–0.266) | 0.412 | 0.011 |
| CLAP linear probe | 0.445 (0.410–0.488) | 0.843 | 0.047 |
| CLAP enrolment, 20 normal clips of the same machine* | 0.622 | 0.529 | 0.715 |
| **v2: CLAP enrolment, 50 spread clips, z-score, 2-of-3 rule*** | **0.774** | **0.835** | **0.712** |

\* Not directly comparable: the enrolment clips of each machine are removed from the scored set. The post-hoc study (`reports/posthoc_calibration.md`) extends this to every machine; **v2** (`reports/v2_selection.md`, `reports/v2_test_results.md`) chose its settings on the 6 development machines (rule: highest recall with ≤ 1 false alarm per machine per day) and was tested once on id_06: 81 % of faults caught with 0.86 false alarms per day (2-of-3 rule, assuming independent clips; single-clip rate 6.2 per day). id_06 had been used before for the locked v1 evaluations, so this is not a blind new-site test, and two development fans were weak (1 % and 9 % caught).

**Main lesson:** the v1 ANN was selected on one validation machine and over-fitted to it; one classifier for all machines learns the machines, not the faults. Asking instead "does this machine still sound like itself?" (v2) solved most of the problem.

## Repository structure
```
src/          config (ALL settings + the split), audio utils, features, dataset, model, training, metrics, evaluation
scripts/      download, Q1 pipeline, Q3 baseline/experiments/final evaluation, Q4 reference + tests,
              Q5 XAI analysis, Q6 CLAP, post-hoc calibration study, submission check
app/          FastAPI backend (main.py), v1 inference (inference.py), Grad-CAM + live check (xai.py),
              v2 enrolment monitoring (monitor.py), static UI (index.html = v1, monitor.html = v2)
tests/        pytest: 30 app tests + 8 XAI tests + 6 v2 monitoring tests
notebooks/    01_eda, 02_solution_design, 03_model_development
experiments/  experiment_log.csv (every run, with hypothesis, result, lesson and Git commit), configs
reports/      decision records, result tables, figures, lock files of the one-time test evaluations
deploy/       Dockerfile (CPU, non-root, health check) and Space build script with smoke test
```

## Reproduce (Windows PowerShell; macOS/Linux similar)
```powershell
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt

# Q1 – data (about 18 GB download, MD5-verified), metadata, split by machine ID, leakage report
python -m scripts.download_mimii
python -m scripts.run_q1 --notebook

# Q2 – features and architecture report
python -m src.features
python -m scripts.q2_architecture_report

# Q3 – baseline, experiments, selection on validation, ONE-TIME locked test
python -m scripts.q3_baseline
python -m scripts.q3_run_experiments --all          # Round 1 + Round 2 (see reports/Q3_round*_plan.md)
python -m scripts.q3_run_experiments --final        # pick the best config on validation + 2 extra seeds
python -m scripts.q3_final_evaluate                  # refuses to run twice (reports/.test_evaluated.json)

# Q4 – web app
python -m scripts.q4_build_reference
python -m uvicorn app.main:app --port 8000           # open http://127.0.0.1:8000
python -m pytest tests -v                            # 44 tests

# Q5 – explanation analysis
python -m scripts.q5_xai_analysis

# Q6 – CLAP (embed -> select on validation -> one-time test)
python -m scripts.q6_clap embed; python -m scripts.q6_clap select; python -m scripts.q6_clap test

# Post-hoc study (after the locked tests): per-machine calibration and normal-only detection
python -m scripts.posthoc_calibration

# v2 – enrolment detector: select on 6 development machines -> tag -> ONE test on id_06
python -m scripts.v2_enrolment embed; python -m scripts.v2_enrolment select
python -m scripts.v2_enrolment test                  # refuses to run twice (reports/.v2_test_evaluated.json)
# then open http://127.0.0.1:8000/monitor and enrol a machine (or a demo machine)

# Check the submission package
python -m scripts.check_submission --tests
```
## Release 1.0 – trained models and data sample
[Release v1.0](https://github.com/pasiuvi/mimii-machine-fault-detection/releases/tag/v1.0) is the **first published package** of this project. It contains **both approaches**: v1 (FaultCNN classifier, page `/`) and v2 (per-machine enrolment monitoring, page `/monitor`). "v1.0" is the release number, not the model version.

| File | Put it in | Used by |
|---|---|---|
| `final_model.pt`, `final_model.json` | `models/` | Deployed FaultCNN (E026, seed 43) and its thresholds (v1 page, XAI) |
| `E026.pt`, `E026_s44.pt` | `models/` | The two extra seeds for seed agreement (v1 page) |
| `sample_data.zip` | unzip into `data/sample/` | 320 representative MIMII clips (pump + fan, normal + abnormal, channel 0) to test the code without the 17.9 GB download |

`models/` and the audio are not in Git because of their size. v2 downloads the pinned CLAP model from the Hugging Face Hub on first use. The full dataset is downloaded and MD5-checked with `python -m scripts.download_mimii`; the sample is made with `python -m scripts.make_sample --per-group 20`. The data sample is from the MIMII dataset (Purohit et al., 2019, DOI 10.5281/zenodo.3384388), CC BY-SA 4.0, redistributed unchanged under the same licence.

## Rules followed (reproducibility and honesty)
- One fixed split by machine ID; the test machine was used once per model family and locked (Git tags `q3-test-done`, `q6-test-done`).
- Every choice (model, threshold, Grad-CAM layer, CLAP prompts) was made on validation and committed before testing (`q3-before-test`, `q6-before-test`).
- Every run is logged with its hypothesis, result and commit (`experiments/experiment_log.csv`); failures are kept.
- Post-hoc analyses and v2 are labelled as such and never replace the locked results (tags `posthoc-plan`, `v2-plan`, `v2-before-test`, `v2-test-done`).

## Limitations and what production would need
- v1 is not usable on new machines. v2 is evaluated on only 4 machines per type (one test machine per type, seen before), one noise level (6 dB), one microphone, 10-s clips; two development fans were weak.
- v2 needs enrolment clips that are confirmed healthy and cover normal operating conditions; re-enrol after maintenance or sensor changes. The 2-of-3 false-alarm figure assumes independent clips.
- Thresholds must be calibrated per machine; the explanation shows *where* the model looked, not the physical cause.
- Before any operational use: validation on many more machines, sensors, loads and noise levels; false alarms per operating day agreed with maintenance; drift monitoring; access control and audit; and a safe escalation path (the app never replaces a scheduled inspection).

## Licence and data
Code: for assessment and portfolio use. Data: MIMII dataset, CC BY-SA 4.0. The full dataset is not redistributed (use the download script); only the small sample in Release v1.0 is shared, unchanged and under the same licence. CLAP weights: LAION, from the Hugging Face Hub.
