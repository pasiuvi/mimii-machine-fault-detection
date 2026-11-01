# AcoustiGuard – Acoustic Fault Screening for Industrial Pumps and Fans

An end-to-end deep-learning project that listens to a 10-second recording of a pump or fan and helps a **plant maintenance engineer** decide whether to raise an inspection work order. It includes a web app with honest uncertainty, live-checked Grad-CAM explanations, and a comparison with an audio-language foundation model (CLAP).

> MSc Data Science (NIBM / Coventry University) – AI coursework · Author: **B. D. P. A. Karunarathna (Don Ashan)**, COMScDS251P-001

> **Status: research prototype, not a production system.** On the held-out test machine, the deployed network was no better than chance (macro recall 3.2 % at a 5 % false-alarm rate), and thresholds set on one machine do not transfer to another. See [Results](#results) and [Limitations](#limitations-and-what-production-would-need).

| | |
|---|---|
| User / decision | Plant maintenance engineer: inspect now, or wait for the scheduled service |
| Data | MIMII (Purohit et al., 2019, CC BY-SA 4.0): pump + fan, 6 dB SNR, 16 kHz, microphone channel 0 |
| Split | By **machine ID** (no leakage): train id_00 + id_02 · validation id_04 · **test id_06** (used once, locked) |
| Model | FaultCNN: ImageNet-pretrained EfficientNet-B0 on 128-band log-mel spectrograms + machine-type input (4.17 M parameters) |
| Primary metric | **Macro TPR@FPR5**: fault recall at a 5 % false-alarm rate, threshold per machine type, set on validation |
| Web app | FastAPI + HTML/JS (`app/`): live predictions, input and out-of-distribution checks, uncertainty, limitations |
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

\* Not directly comparable: the 20 reference clips per machine are removed from the scored set. The post-hoc study (`reports/posthoc_calibration.md`) extends this to every machine and to per-machine thresholds.

**Main lesson:** the final ANN was selected on one validation machine and over-fitted to it. Ranking on validation reversed on the test machine. One validation machine per type is not enough to choose a model.

## Repository structure
```
src/          config (ALL settings + the split), audio utils, features, dataset, model, training, metrics, evaluation
scripts/      download, Q1 pipeline, Q3 baseline/experiments/final evaluation, Q4 reference + tests,
              Q5 XAI analysis, Q6 CLAP, post-hoc calibration study, submission check
app/          FastAPI backend (main.py), inference engine (inference.py), Grad-CAM + live check (xai.py), static UI
tests/        pytest: 30 app tests + 8 XAI tests (unit tests with a known-answer network + API tests)
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
python -m pytest tests -v                            # 38 tests

# Q5 – explanation analysis
python -m scripts.q5_xai_analysis

# Q6 – CLAP (embed -> select on validation -> one-time test)
python -m scripts.q6_clap embed; python -m scripts.q6_clap select; python -m scripts.q6_clap test

# Post-hoc study (after the locked tests): per-machine calibration and normal-only detection
python -m scripts.posthoc_calibration

# Check the submission package
python -m scripts.check_submission --tests
```
**Model files:** `models/` is not in Git because of its size. The trained models (`final_model.pt/.json`, `E026.pt`, `E026_s44.pt`) are attached to the GitHub **Release v1.0**. Download them into `models/`, or retrain with the Q3 commands.

## Rules followed (reproducibility and honesty)
- One fixed split by machine ID; the test machine was used once per model family and locked (Git tags `q3-test-done`, `q6-test-done`).
- Every choice (model, threshold, Grad-CAM layer, CLAP prompts) was made on validation and committed before testing (`q3-before-test`, `q6-before-test`).
- Every run is logged with its hypothesis, result and commit (`experiments/experiment_log.csv`); failures are kept.
- Post-hoc analyses are labelled as such and never replace the locked results.

## Limitations and what production would need
- Not proven on new machines: only 4 machines per type, one noise level (6 dB), one microphone, 10-s clips.
- Thresholds must be calibrated per machine; the explanation shows *where* the model looked, not the physical cause.
- Before any operational use: validation on many more machines, sensors, loads and noise levels; false alarms per operating day agreed with maintenance; drift monitoring; access control and audit; and a safe escalation path (the app never replaces a scheduled inspection).

## Licence and data
Code: for assessment and portfolio use. Data: MIMII dataset, CC BY-SA 4.0 (not redistributed; use the download script). CLAP weights: LAION, from the Hugging Face Hub.
