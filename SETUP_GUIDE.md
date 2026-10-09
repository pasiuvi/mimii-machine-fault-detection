# Setup Guide (Windows): VS Code + Python + Git/GitHub, then Question 1

Follow the steps in order. Type every command in the **VS Code terminal** (menu **Terminal → New Terminal**), from inside the project folder. Do not type the `>`.

---

## Part A: Install the tools (one time only)

| # | Install | Where | Notes |
|---|---|---|---|
| 1 | **Python 3.11** (64-bit) | python.org/downloads/windows | On the first installer screen, tick **"Add python.exe to PATH"**. Use 3.11, not the newest version, because PyTorch and the audio libraries work best with it. |
| 2 | **Git** | git-scm.com/download/win | Keep all the default options. |
| 3 | **VS Code** | code.visualstudio.com | |
| 4 | VS Code extensions | Click the Extensions icon (left bar) | Install **Python**, **Jupyter**, and **GitLens** (optional). |
| 5 | GitHub account | github.com | You already have one: `pasiuvi`. |

Check that it worked. Open a new terminal and run:
```powershell
> py -3.11 --version      # should print Python 3.11.x
> git --version
```

Tell Git who you are (one time only):
```powershell
> git config --global user.name "Don Ashan"
> git config --global user.email "pasiuvi@gmail.com"
> git config --global init.defaultBranch main
```

---

## Part B: Open the project and create the Python environment

1. Unzip `mimii-machine-fault-detection.zip` into a short path, for example `C:\Projects\mimii-machine-fault-detection`.
   **Do not use OneDrive or Desktop folders.** The dataset is large and OneDrive would try to sync it.
2. In VS Code, use **File → Open Folder…** and choose that folder.
3. In the terminal, create and activate a **virtual environment** (a private Python just for this project):
```powershell
> py -3.11 -m venv .venv
> .venv\Scripts\activate
```
   If you get *"running scripts is disabled"*, run this once and then activate again:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
   When it works, `(.venv)` appears at the start of the prompt.
4. Install the packages:
```powershell
(.venv) > python -m pip install --upgrade pip
(.venv) > pip install -r requirements.txt
```
5. Tell VS Code to use this Python: press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, then pick **.venv**.

---

## Part C: Test the pipeline on FAKE data (about 2 minutes, before the big download)

This proves the code works on your laptop before you spend hours downloading.
```powershell
(.venv) > python -m tests.make_fake_mimii
(.venv) > python -m src.build_metadata
(.venv) > python -m src.make_splits
(.venv) > python -m src.leakage_checks
```
Open `notebooks/01_eda.ipynb`. Click **Select Kernel** (top right) and choose **.venv**, then click **Run All**. You should see the charts.

Then **delete the fake data** before using the real dataset:
```powershell
(.venv) > Remove-Item -Recurse -Force data\raw\6_dB_*, data\processed\*.csv, data\processed\*.npz, reports\figures\*.png, reports\*.md, reports\*.csv, reports\*.json
```

---

## Part D: Put the project on GitHub (version control)

1. On github.com, click **New repository**:
   - Name: `mimii-machine-fault-detection`
   - Visibility: **Private** for now. Make it Public **after** your marks are released, so other students can't copy it before submission.
   - Do **not** tick "Add README" (you already have one).
2. In the VS Code terminal:
```powershell
(.venv) > git init
(.venv) > git add .
(.venv) > git commit -m "Initial project structure: download, metadata, split, leakage checks, EDA notebook"
(.venv) > git branch -M main
(.venv) > git remote add origin https://github.com/pasiuvi/mimii-machine-fault-detection.git
(.venv) > git push -u origin main
```
   A browser window opens the first time so you can log in to GitHub.
3. Run `git status`. It should **not** list any `.wav` or `.zip` files. `.gitignore` keeps the data off GitHub.

### Daily Git routine (use this after every work session)
```powershell
> git status                      # what changed?
> git add .
> git commit -m "Q1: add class-balance and spectrum plots"
> git push
```
Good commit messages say **what** changed and **which question** it is for. Put the commit ID of each experiment in the `git_commit` column of `experiments/experiment_log.csv`. This makes every result traceable (brief section 6). You can get the ID with `git rev-parse --short HEAD`.

You can also use the **Source Control** icon (left bar in VS Code): type a message, click ✓ Commit, then **Sync**.

---

## Part E: Download the real dataset

You need about **40 GB free** disk space. The download is about 18 GB, and you need the same again while it unzips.
```powershell
(.venv) > python -m scripts.download_mimii --list            # shows what will download
(.venv) > python -m scripts.download_mimii --machines pump   # 7.7 GB first
(.venv) > python -m scripts.download_mimii --machines fan    # 10.2 GB next
```
- If the internet drops, **run the same command again**. It continues from where it stopped.
- Each zip is checked against the official MD5 checksum, then unzipped and deleted to save space.

---

## Part F: Question 1, step by step

```powershell
(.venv) > python -m src.build_metadata    # 1. table of all clips   -> data/processed/metadata.csv
(.venv) > python -m src.make_splits       # 2. the ONE fixed split  -> data/processed/splits.csv
(.venv) > python -m src.leakage_checks    # 3. leakage report       -> reports/leakage_report.md  (first run ~10 min)
```
4. Open `notebooks/01_eda.ipynb` and click **Run All**. All the figures are saved to `reports/figures/`.
5. Commit:
```powershell
> git add data/processed/metadata.csv data/processed/splits.csv notebooks reports
> git commit -m "Q1: EDA and leakage checks on real MIMII data (pump+fan, 6 dB)"
> git push
```
6. **Write the Q1 section of the report (about 4–5 pages)** in this order:
   1. Dataset description: copy the table from notebook section 1 and add the counts table.
   2. EDA: 4–6 of the saved figures. **Under every figure, write one sentence on what it means for the model.**
   3. False-positive vs false-negative consequences: notebook section 9.
   4. Primary metric and why: notebook section 10.
   5. Leakage checks and data-quality handling: from `reports/leakage_report.md`.
   6. **Three findings that constrain modelling**: notebook section 11, filled in with **your real numbers**.

> **Rule from the brief:** do not look at the test machine (id_06) in depth. The notebook only counts it. All deep analysis uses train and validation.

7. Make the sample for submission (done at the end of the project):
   `python -m scripts.make_sample --per-group 20`

---

## Part G: Portfolio tips (for job hunting)
- Keep the README's "Project status" list up to date as you finish each question.
- When the web app is ready (Q4), deploy it free on **Hugging Face Spaces** and put the link at the top of the README.
- After marks are released: make the repo Public, **pin it** on your GitHub profile, and add it to LinkedIn under **Featured** and **Projects**. Write a short post with the demo video: the problem, the user, the result, and one thing you learned.
- Add your real LinkedIn URL in `README.md` (replace `YOUR-LINKEDIN`).
