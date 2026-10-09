# Beginner's Guide: Run the Project in VS Code and Use Git/GitHub

**Project:** Acoustic Fault Detection for Industrial Pumps and Fans (MIMII), ANN coursework
**For:** someone who has never used Python, VS Code or Git before
**Computer:** Windows 10/11

Go through the parts **in order**. Do not skip ahead. Each part ends with a ✅ **Check**. If your screen matches the check, go to the next part. If it doesn't, look at **Part 12 – Troubleshooting**.

---

## Table of contents
1. [The big picture: what each tool does](#part-1--the-big-picture)
2. [Install Python](#part-2--install-python-311)
3. [Install Git](#part-3--install-git)
4. [Install VS Code and its extensions](#part-4--install-vs-code-and-extensions)
5. [Open the project in VS Code](#part-5--open-the-project-in-vs-code)
6. [Using the terminal (the command box)](#part-6--using-the-terminal)
7. [Create the virtual environment and install packages](#part-7--virtual-environment-and-packages)
8. [Test run with fake data](#part-8--test-run-with-fake-data)
9. [Put the project under version control (Git + GitHub)](#part-9--version-control-with-git-and-github)
10. [Download the real dataset and run Question 1](#part-10--real-dataset-and-question-1)
11. [Your daily routine](#part-11--your-daily-routine)
12. [Troubleshooting](#part-12--troubleshooting)
13. [Glossary](#part-13--glossary)

---

## Part 1 – The big picture

| Tool | What it is | Simple comparison |
|---|---|---|
| **Python** | The programming language the project is written in | The engine |
| **VS Code** | The program where you open, read, edit and run the code | The car's dashboard |
| **Terminal** | A text box inside VS Code where you type commands | Talking to the computer directly |
| **Virtual environment (.venv)** | A private copy of Python, just for this project, with its own packages | A separate toolbox per project, so projects don't mix up |
| **Packages** (numpy, pandas, librosa…) | Ready-made code libraries we use | Tools you put in the toolbox |
| **Jupyter notebook (.ipynb)** | A file that mixes code, charts and text in "cells" | A lab notebook that can run code |
| **Git** | Saves snapshots ("commits") of your project on your computer | Unlimited "Save As version 1, 2, 3…" with notes |
| **GitHub** | A website that stores a copy of your Git project online | Google Drive for code, and your public portfolio |

**How the project works, step by step:**
```
download_mimii.py  →  build_metadata.py  →  make_splits.py  →  leakage_checks.py  →  01_eda.ipynb
 (get the audio)      (list every clip)     (train/val/test)    (check for cheating)   (charts for Q1)
```

---

## Part 2 – Install Python 3.11

> Why 3.11 and not the newest? The deep-learning libraries (PyTorch) we use later work most reliably with 3.11.

1. Go to **https://www.python.org/downloads/windows/**
2. Find the latest **Python 3.11.x** release and click **"Windows installer (64-bit)"**.
3. Run the downloaded file.
4. ⚠️ **IMPORTANT:** on the first screen, tick **☑ "Add python.exe to PATH"** at the bottom.
5. Click **"Install Now"** and wait.
6. If you see **"Disable path length limit"**, click it. This avoids errors with long file paths.
7. Click **Close**.

✅ **Check:** press the **Windows key**, type `cmd`, press Enter. In the black window type:
```
py -3.11 --version
```
It should show `Python 3.11.x`. Close the window.

---

## Part 3 – Install Git

1. Go to **https://git-scm.com/download/win** and the download starts.
2. Run the installer. **Keep clicking "Next" with all the default options.** The defaults are fine.
3. Click **Install**, then **Finish**.

✅ **Check:** open `cmd` again and type:
```
git --version
```
It should show something like `git version 2.4x.x.windows.1`.

---

## Part 4 – Install VS Code and extensions

1. Go to **https://code.visualstudio.com** and click **Download for Windows**.
2. Run the installer. On the "Select Additional Tasks" screen, tick:
   - ☑ **Add "Open with Code" action to Windows Explorer file context menu**
   - ☑ **Add "Open with Code" action to Windows Explorer directory context menu**
   - ☑ **Add to PATH**
3. Finish and open VS Code.
4. **Install the extensions.** Click the **Extensions** icon on the left bar (four small squares), or press `Ctrl+Shift+X`. Search for and click **Install** on:
   - **Python** (by Microsoft)
   - **Jupyter** (by Microsoft)
   - *(optional)* **Rainbow CSV**, which makes CSV files easier to read
   - *(optional)* **GitLens**, which shows Git history inside the code

✅ **Check:** the Extensions panel lists Python and Jupyter under "Installed".

### A quick tour of VS Code
```
┌──────┬──────────────────┬─────────────────────────────────────┐
│ Left │  EXPLORER        │                                     │
│ bar  │  (your files)    │      EDITOR (the open file)         │
│      │                  │                                     │
│ 📄   │  ▸ data          │                                     │
│ 🔍   │  ▸ notebooks     │                                     │
│ ⑂    │  ▸ scripts       ├─────────────────────────────────────┤
│ ▶    │  ▸ src           │  TERMINAL  (type commands here)     │
│ ⊞    │  README.md       │  (.venv) PS C:\Projects\mimii...>   │
└──────┴──────────────────┴─────────────────────────────────────┘
```
Left bar icons: 📄 Explorer (files) · 🔍 Search · **⑂ Source Control (Git)** · ▶ Run & Debug · ⊞ Extensions.

---

## Part 5 – Open the project in VS Code

1. Create a folder: **`C:\Projects`**
   ⚠️ **Do not** put the project in *Desktop*, *Documents* or *OneDrive*. OneDrive would try to upload 18 GB of audio, and long paths cause errors.
2. Right-click **`mimii-machine-fault-detection.zip`** → **Extract All…** → choose `C:\Projects` → **Extract**.
   You should now have `C:\Projects\mimii-machine-fault-detection\` with `README.md` directly inside it.
   *(If you get a double folder like `...\mimii-machine-fault-detection\mimii-machine-fault-detection\`, use the inner one.)*
3. In VS Code: **File → Open Folder…** → select `C:\Projects\mimii-machine-fault-detection` → **Select Folder**.
4. If asked *"Do you trust the authors of the files in this folder?"* → **Yes, I trust the authors**.

✅ **Check:** the Explorer on the left shows `data`, `notebooks`, `scripts`, `src`, `tests`, `README.md`, `requirements.txt`.

### What's inside the project
| Folder / file | What it contains |
|---|---|
| `src/config.py` | **All settings in one place**: paths, random seed, audio settings, and the train/val/test split |
| `scripts/download_mimii.py` | Downloads the dataset from Zenodo |
| `src/build_metadata.py` | Makes a table listing every audio clip |
| `src/make_splits.py` | Assigns each clip to train / validation / test |
| `src/leakage_checks.py` | Checks for data leakage and data-quality problems |
| `src/metrics.py` | The evaluation metric used for every model |
| `notebooks/01_eda.ipynb` | The Question 1 analysis and charts |
| `experiments/experiment_log.csv` | Your experiment log (required by the brief) |
| `reports/` | Where charts and reports are saved |
| `tests/make_fake_mimii.py` | Makes a tiny fake dataset for testing |
| `.gitignore` | Tells Git which files **not** to upload (the big audio files) |

---

## Part 6 – Using the terminal

Open the terminal in VS Code: **menu Terminal → New Terminal** (or press ``Ctrl+` ``; the ` key is under Esc).

It looks like this:
```
PS C:\Projects\mimii-machine-fault-detection>
```
- `PS` means it's **PowerShell**, the Windows terminal. That's fine.
- The path shows **which folder** you're in. It **must** be the project folder. If it isn't, type:
  `cd C:\Projects\mimii-machine-fault-detection`

**How to run a command:** type it (or copy-paste it; right-click pastes in the terminal) and press **Enter**. Wait until the prompt `PS C:\...>` appears again before typing the next command.

> In this guide, lines in grey boxes are commands. Type them **without** `PS C:\...>` or `(.venv)` at the front.

Useful keys:
| Key | Does |
|---|---|
| ↑ (up arrow) | Brings back your previous command |
| `Ctrl + C` | **Stops** a running command |
| `cls` | Clears the screen |
| Tab | Auto-completes file/folder names |

---

## Part 7 – Virtual environment and packages

You only do this **once** for the project.

### 7.1 Create the virtual environment
```powershell
py -3.11 -m venv .venv
```
This takes about 20 seconds. A new folder called `.venv` appears in the Explorer.

### 7.2 Activate it
```powershell
.venv\Scripts\activate
```
✅ The prompt now starts with **`(.venv)`**:
```
(.venv) PS C:\Projects\mimii-machine-fault-detection>
```

> ❗ If you get a red error: *"running scripts is disabled on this system"*, run this **once**, type `Y` if asked, then activate again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> .venv\Scripts\activate
> ```

> 🔁 **Every time you open a new terminal, you must activate again** (`.venv\Scripts\activate`). If you don't see `(.venv)`, you're using the wrong Python.

### 7.3 Install the packages
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
This downloads numpy, pandas, librosa and the others, and takes about 3–10 minutes. You'll see a lot of text; that's normal. It ends with `Successfully installed ...`.

### 7.4 Tell VS Code to use this Python
1. Press **`Ctrl+Shift+P`**. A search box opens at the top.
2. Type **`Python: Select Interpreter`** and press Enter.
3. Choose the one that shows **`('.venv': venv)`** and `.\.venv\Scripts\python.exe`.

✅ **Check:**
```powershell
python -c "import librosa, pandas, sklearn; print('All good!')"
```
It prints `All good!`

---

## Part 8 – Test run with fake data

Before downloading 18 GB, test that everything works using a tiny **fake** dataset (304 small clips, about 1 minute).

Make sure you see `(.venv)` and then run these commands **one at a time**:

### 8.1 Make fake data
```powershell
python -m tests.make_fake_mimii
```
Expected output: `Fake MIMII-like data written to data/raw/`

### 8.2 Build the metadata table
```powershell
python -m src.build_metadata
```
Expected output: a table like this:
```
label_name               abnormal  normal  All
machine_type machine_id
fan          id_00              8      30   38
...
Saved 304 rows -> data/processed/metadata.csv
```

### 8.3 Make the train/validation/test split
```powershell
python -m src.make_splits
```
Expected output: a table showing `test`, `train`, `val` rows, and `Saved -> data/processed/splits.csv`

### 8.4 Run the leakage checks
```powershell
python -m src.leakage_checks
```
Expected output: a report ending with `Saved -> reports/leakage_report.md`. It should include **`Result: PASS - every machine is in exactly one split`**.

### 8.5 Run the EDA notebook
1. In the Explorer, open **`notebooks` → `01_eda.ipynb`**.
2. At the top right, click **"Select Kernel"** → **Python Environments** → **`.venv`**.
   *(If VS Code asks to install `ipykernel`, click **Install**.)*
3. Click **"▷▷ Run All"** in the toolbar at the top of the notebook.
4. Scroll down. Every cell should show output (tables, charts, audio players) and **no red errors**.

**Notebook basics:**
| Action | How |
|---|---|
| Run one cell | Click inside it and press `Shift+Enter` |
| Run everything | **Run All** button |
| Start fresh | **Restart** button (then Run All) |
| Cell number `[5]` | The order it was run in |
| `[*]` | Still running, so wait |

✅ **Check:** the folder `reports/figures/` now contains 6 `.png` charts.

### 8.6 Delete the fake data (important!)
The fake data must **not** be mixed with the real data:
```powershell
Remove-Item -Recurse -Force data\raw\6_dB_*, data\processed\*.csv, data\processed\*.npz, reports\figures\*.png, reports\*.md, reports\*.csv, reports\*.json
```
*(It's normal if it says some files were not found.)*

🎉 Your setup works. Now put it under version control **before** changing anything else.

---

## Part 9 – Version control with Git and GitHub

### 9.1 Git in 1 minute
- A **repository (repo)** is a project folder that Git tracks.
- A **commit** is a saved snapshot with a message, for example *"Q1: add EDA charts"*. You can always go back to any commit.
- **Staging (`git add`)** means choosing which changes go into the next commit.
- **Push** uploads your commits to GitHub. **Pull** downloads changes from GitHub.
- **`.gitignore`** lists files Git should never track. Ours excludes the big audio data, `.venv`, and secret keys.

```
 your files ──git add──▶ staging ──git commit──▶ local history ──git push──▶ GitHub
```

### 9.2 Tell Git who you are (once per computer)
```powershell
git config --global user.name "Don Ashan"
git config --global user.email "pasiuvi@gmail.com"
git config --global init.defaultBranch main
```
> Use the **same email as your GitHub account** so the commits show on your GitHub profile (the green squares that recruiters see).

### 9.3 Create the repository on GitHub
1. Log in at **https://github.com** (username `pasiuvi`).
2. Click the **+** (top right) → **New repository**.
3. Fill in:
   - **Repository name:** `mimii-machine-fault-detection`
   - **Description:** `Deep-learning acoustic fault detection for industrial pumps and fans (MIMII) with XAI, web app and GenAI comparison`
   - **Visibility:** 🔒 **Private** *(make it Public after your marks are released, so no one can copy your coursework before the deadline)*
   - ⚠️ **Do NOT tick** "Add a README", ".gitignore" or "license". The project already has them.
4. Click **Create repository**. Leave this page open; you'll need the URL.

### 9.4 Connect your folder and push (first time)
In the VS Code terminal (project folder, `(.venv)` active):
```powershell
git init
git add .
git status
```
⚠️ **Look at the `git status` list before committing.** It must **NOT** contain any `.wav`, `.zip`, or `.venv` files. You should see things like `README.md`, `src/config.py`, `notebooks/01_eda.ipynb`.

Then:
```powershell
git commit -m "Initial project structure: download, metadata, split, leakage checks, EDA notebook"
git branch -M main
git remote add origin https://github.com/pasiuvi/mimii-machine-fault-detection.git
git push -u origin main
```
- The first time, a window pops up asking you to **sign in to GitHub**. Choose **"Sign in with your browser"** and approve.
- Refresh the GitHub page. Your files are now online. 🎉

### 9.5 Doing the same with buttons instead of commands (VS Code Source Control)
You can use the **Source Control** icon (⑂, third icon on the left bar) instead of typing:

1. Click **⑂ Source Control**. Changed files are listed under **Changes**.
   - **M** = modified, **U** = untracked (new), **D** = deleted.
2. Click a file to see **what changed**: red is removed, green is added.
3. Hover over **Changes** and click **+** to stage all changes, or click **+** next to a single file.
4. Type a message in the box at the top, for example `Q1: add class balance chart`.
5. Click **✓ Commit**.
6. Click **Sync Changes** (or **⋯ → Push**) to upload to GitHub.

### 9.6 Good commit messages
| ✅ Good | ❌ Bad |
|---|---|
| `Q1: add spectrum-by-class plot and findings` | `update` |
| `Q3: E007 AdamW lr=3e-4 batch 32 (val TPR 0.61)` | `asdf` |
| `Fix: load mono files in audio_utils` | `final final v2` |

**Commit often**, after each meaningful step. Small commits are easy to understand and undo.

### 9.7 Useful Git commands
| Command | What it does |
|---|---|
| `git status` | Shows what changed since the last commit |
| `git log --oneline` | Lists your commits (press `q` to exit) |
| `git diff` | Shows exact line changes not yet committed |
| `git rev-parse --short HEAD` | Shows the current commit ID, e.g. `a1b2c3d` → put this in your experiment log |
| `git restore <file>` | ⚠️ Throws away **uncommitted** changes to one file |
| `git pull` | Gets changes from GitHub (if you work on two computers) |

### 9.8 Why Git matters for your marks
- The brief says **every result in the report must be traceable to the experiment log**. Write the commit ID in the `git_commit` column of `experiments/experiment_log.csv` for every experiment. Anyone can then see exactly which code produced which result.
- In the live presentation, you can show your commit history as proof that you did the work step by step.

---

## Part 10 – Real dataset and Question 1

### 10.1 Check disk space
You need **about 40 GB free** on drive C: (18 GB download plus the same again while unzipping). Check in **File Explorer → This PC**.

### 10.2 Download (this can take hours, depending on your internet)
```powershell
python -m scripts.download_mimii --list
```
This shows what will be downloaded without downloading it:
`Files: ['6_dB_pump.zip', '6_dB_fan.zip']  (about 17.9 GB to download)`

Download **one machine type at a time**:
```powershell
python -m scripts.download_mimii --machines pump
```
When pump finishes:
```powershell
python -m scripts.download_mimii --machines fan
```
What you'll see for each file:
```
[1/3] Downloading 6_dB_pump.zip (~7.7 GB)
  6_dB_pump.zip:  37%|███████▌        | 2.85G/7.70G [12:31<21:10, 3.8MB/s]
[2/3] Verifying MD5 checksum
  OK - file is identical to the official Zenodo file
[3/3] Extracting
```
- 🔌 **If the internet drops or you close the laptop**, run **the same command again**. It continues from where it stopped.
- 💤 Turn off sleep mode while downloading: **Settings → System → Power → Screen and sleep → Never** (when plugged in).
- After it finishes, the zip is deleted automatically to save space.

### 10.3 Run the Question 1 pipeline on the real data
```powershell
python -m src.build_metadata
python -m src.make_splits
python -m src.leakage_checks
```
- `leakage_checks` takes **about 5–15 minutes** the first time, because it analyses every clip. After that the results are cached and it's fast.
- Read the tables it prints. You'll copy them into your report.

### 10.4 Run the EDA notebook
Open `notebooks/01_eda.ipynb` → kernel **.venv** → **Restart** → **Run All**.
All the charts are saved to `reports/figures/`, ready for your report.

### 10.5 Save your work to GitHub
```powershell
git add data/processed/metadata.csv data/processed/splits.csv notebooks reports
git commit -m "Q1: EDA and leakage checks on real MIMII data (pump+fan, 6 dB)"
git push
```
> Why commit `splits.csv`? It's the record of **exactly** which clips are train, validation and test. That makes your results reproducible, which the brief requires.

### 10.6 Write the Question 1 section of your report
Use the notebook sections in this order. They follow the Q1 marking points:

| Report part | Take it from |
|---|---|
| 1. Dataset description (source, licence, size, target) | Notebook section 1 |
| 2. EDA with charts | Sections 2–7 (choose 4–6 charts; **under each one, write one sentence on what it means for the model**) |
| 3. Costs of false positives / false negatives | Section 9 |
| 4. Primary metric and why | Section 10 |
| 5. Leakage checks and data quality | `reports/leakage_report.md` (section 8) |
| 6. **Three findings that constrain modelling** | Section 11, with **your real numbers** filled in |

> ⚠️ **Test-set rule:** never listen to or analyse the test machine (`id_06`) in depth. The notebook only counts it. This is required by the brief (section 6, "test set isolation").

---

## Part 11 – Your daily routine

Every time you work on the project:

```powershell
# 1. Open VS Code → File → Open Recent → mimii-machine-fault-detection
# 2. Open a terminal (Ctrl + `) and activate the environment
.venv\Scripts\activate

# 3. (Only if you also work on another computer) get the latest version
git pull

# 4. ... do your work ...

# 5. Save a snapshot and upload it
git status
git add .
git commit -m "Q?: what you did"
git push
```

And for every training run (from Q3 onwards), add one row to `experiments/experiment_log.csv`. Failed runs count too: the brief requires at least 4 unsuccessful attempts.

---

## Part 12 – Troubleshooting

| Problem / error message | Fix |
|---|---|
| `'py' is not recognized` / `'python' is not recognized` | Python isn't on PATH. Re-run the Python installer → **Modify** → tick "Add Python to environment variables". Then **restart VS Code**. |
| `running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| No `(.venv)` at the start of the prompt | Run `.venv\Scripts\activate`. |
| `ModuleNotFoundError: No module named 'librosa'` (or another package) | The environment isn't active, or packages aren't installed: activate it, then `pip install -r requirements.txt`. In a notebook: select the **.venv** kernel. |
| `ModuleNotFoundError: No module named 'src'` | You're not in the project folder, or you typed `python src/build_metadata.py`. Use **`python -m src.build_metadata`** from the project folder. |
| `No wav files in data/raw` | The download didn't finish or the data was deleted. Run the download command again. |
| `FileNotFoundError ... metadata.csv` or `splits.csv` | Run the steps in order: `build_metadata` → `make_splits` → `leakage_checks`. |
| Download stuck or very slow | Press `Ctrl+C` and run the same command again; it resumes. Try at night or on a faster connection. |
| `MD5 mismatch` | The file was corrupted during download. The script already deleted it, so run the command again. |
| `WARNING: only XX GB free` | Free up disk space, or download one machine at a time. |
| Notebook: "Select Kernel" doesn't show `.venv` | `Ctrl+Shift+P` → **Developer: Reload Window**, then try again. Or `pip install ipykernel`. |
| Notebook cell runs forever `[*]` | The first EDA run computes features (5–15 min). Otherwise click **Interrupt** and then **Restart**. |
| `git push` → `rejected (fetch first)` | GitHub has changes you don't have: run `git pull`, then `git push`. |
| `git push` → `Authentication failed` | Run `git push` again and choose **Sign in with your browser**. |
| `remote origin already exists` | `git remote set-url origin https://github.com/pasiuvi/mimii-machine-fault-detection.git` |
| `git status` shows `.wav` files | Don't commit! Check that `.gitignore` is in the project root (not inside another folder). |
| You accidentally committed a big file | Don't push. Ask for help (or run `git reset --soft HEAD~1`, then remove the file from staging with `git restore --staged <file>`). |
| `fatal: not a git repository` | You're in the wrong folder: `cd C:\Projects\mimii-machine-fault-detection` |
| Anything else | Copy the **last 10–15 lines** of the red error message and send them to me (Claude). |

---

## Part 13 – Glossary

| Word | Meaning |
|---|---|
| **Clip** | One 10-second audio recording of a machine |
| **Machine ID** (`id_00`…) | One specific physical machine. All its clips stay in the same split. |
| **Train / validation / test split** | Train teaches the model; validation chooses settings and thresholds; test is used **once** at the very end for the final score |
| **Data leakage** | When information the model shouldn't have (for example the same machine appearing in train and test) makes results look better than they really are |
| **Log-mel spectrogram** | A picture of sound (time × frequency × loudness). This is what the CNN "looks at". |
| **EDA** | Exploratory Data Analysis: understanding the data with statistics and charts before modelling |
| **Seed (42)** | A fixed number that makes the random choices repeatable, so results can be reproduced |
| **TPR @ FPR 5 %** | Our main metric: the share of real faults caught when only 5 % of healthy clips raise a false alarm |
| **Virtual environment** | The project's private Python with its own packages (`.venv`) |
| **Commit / push** | Save a snapshot / upload snapshots to GitHub |
| **MD5 checksum** | A fingerprint of a file, used to prove the download is identical to the original |

---

### ✅ Final checklist for this stage
- [ ] Python 3.11, Git and VS Code installed; the Python and Jupyter extensions installed
- [ ] `.venv` created, packages installed, interpreter selected
- [ ] Fake-data test passed, then the fake data deleted
- [ ] Private GitHub repo created and the first commit pushed
- [ ] Real data downloaded (pump + fan, 6 dB)
- [ ] `build_metadata` → `make_splits` → `leakage_checks` run on the real data
- [ ] `01_eda.ipynb` run fully; charts saved in `reports/figures/`
- [ ] Q1 results committed and pushed
- [ ] Q1 report section drafted with the three findings filled in

**Next stage:** Question 2, designing the CNN. Send me your EDA results and the three findings when you're done.
