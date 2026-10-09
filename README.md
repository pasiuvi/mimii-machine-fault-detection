# Acoustic Fault Detection for Industrial Pumps and Fans

Deep-learning system that listens to a 10-second machine recording and tells a **plant maintenance engineer** whether the machine sounds **abnormal**, so they can raise an inspection work order **before** the machine breaks down. It also shows **which part of the sound** triggered the alarm.

> MSc Data Science (NIBM / Coventry University) — Artificial Neural Networks coursework, built to industry standard.
> Author: **Don Ashan** · [GitHub](https://github.com/pasiuvi) · [LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN)

| | |
|---|---|
| **User** | Plant maintenance engineer |
| **Decision** | Raise an inspection work order now vs. wait for the next scheduled service |
| **Data** | MIMII dataset (Hitachi), pump + fan, 6 dB SNR, 16 kHz audio |
| **Model** | Pretrained CNN on log-mel spectrograms *(in progress)* |
| **Baseline** | MFCC features + Random Forest / SVM *(in progress)* |
| **Explainability** | Grad-CAM on the spectrogram *(in progress)* |
| **GenAI comparison** | Zero-shot CLAP (audio-language foundation model) *(in progress)* |
| **Primary metric** | Recall at 5 % false-alarm rate (TPR @ FPR 0.05) |

## Project status
- [x] Q1 – Dataset, EDA, leakage checks
- [ ] Q2 – ANN design and alternatives
- [ ] Q3 – Training, baseline, optimisation study
- [ ] Q4 – Web application
- [ ] Q5 – Explainable AI
- [ ] Q6 – Generative / foundation model comparison

## Repository structure
```
├── scripts/
│   ├── download_mimii.py      # reproducible download (resume + MD5 check)
│   └── make_sample.py         # small sample for submission
├── src/
│   ├── config.py              # ALL settings: paths, seed, audio params, THE split
│   ├── audio_utils.py         # load audio, log-mel, loudness
│   ├── build_metadata.py      # wav files -> data/processed/metadata.csv
│   ├── make_splits.py         # group split by machine ID -> splits.csv
│   ├── eda_features.py        # cached clip summaries for EDA
│   ├── leakage_checks.py      # leakage + data-quality report
│   └── metrics.py             # one metric implementation for every model
├── notebooks/01_eda.ipynb     # Question 1
├── experiments/experiment_log.csv
├── reports/                   # figures + generated reports
├── tests/make_fake_mimii.py   # tiny fake dataset to test the pipeline
└── requirements.txt
```

## Reproduce
> New to Python, VS Code or Git? Start with **[BEGINNER_GUIDE.md](BEGINNER_GUIDE.md)**.
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python -m scripts.download_mimii  # ~18 GB: 6_dB_pump.zip + 6_dB_fan.zip, MD5-verified
python -m src.build_metadata
python -m src.make_splits
python -m src.leakage_checks
# then open notebooks/01_eda.ipynb and Run All
```

## Data split (fixed before any modelling)
Split **by machine ID** so no physical machine appears in more than one split; the test machine is never heard during development.

| Split | Machines |
|---|---|
| train | id_00, id_02 (pump + fan) |
| validation | id_04 (pump + fan) |
| test | id_06 (pump + fan) |

## Dataset & licence
Purohit, H., Tanabe, R., Ichige, K., Endo, T., Nikaido, Y., Suefusa, K., & Kawaguchi, Y. (2019). *MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection.* Proc. DCASE 2019 Workshop. Zenodo, DOI [10.5281/zenodo.3384388](https://doi.org/10.5281/zenodo.3384388). Licence: **CC BY-SA 4.0**. The audio is not stored in this repository.

## AI-tool declaration
Generative AI tools (Claude) were used as a development aid for code scaffolding and explanations; all code and results were run, checked and understood by the author (declared in the report appendix as required).
