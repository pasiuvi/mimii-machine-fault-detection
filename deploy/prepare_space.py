"""
Build a clean, small folder for the Hugging Face Space from this project.

    python -m deploy.prepare_space            -> creates ../acoustiguard-space/

It copies ONLY what the web app needs at run time (code, 3 model files, normalisation stats,
reference files, the 8 demo clips and their Q3 rows). No training data, no test-set predictions
beyond the 8 demo rows, nothing secret. Then it checks the copy by starting the app once.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "acoustiguard-space"


def cp(rel: str | Path) -> None:
    src, dst = ROOT / rel, OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    if OUT.exists():
        # keep the .git folder of the Space clone, replace everything else
        for p in OUT.iterdir():
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    OUT.mkdir(exist_ok=True)

    # code
    for f in ["app/__init__.py", "app/main.py", "app/inference.py", "app/xai.py",
              "app/static/index.html", "app/static/styles.css", "app/static/app.js",
              "app/reference/q4_reference.json",
              "src/__init__.py", "src/config.py", "src/model.py",
              "data/processed/norm_stats.npz"]:
        cp(f)
    if (ROOT / "app/reference/q5_xai.json").exists():       # layer chosen in Q5 (otherwise the app uses features.8)
        cp("app/reference/q5_xai.json")

    # models: deployed + the other seeds listed in the reference file
    ref = json.loads((ROOT / "app/reference/q4_reference.json").read_text())
    meta = json.loads((ROOT / "models/final_model.json").read_text())
    cp("models/final_model.pt"); cp("models/final_model.json")
    for sid in ref["seed_runs"]:
        if sid != meta["source_run"] and (ROOT / f"models/{sid}.pt").exists():
            cp(f"models/{sid}.pt")

    # the 8 demo clips (same selection as the app) + only their rows of the Q3 predictions file
    sys.path.insert(0, str(ROOT))
    from app.main import load_examples
    ex = load_examples()
    d = pd.read_csv(ROOT / "reports/q3_test_predictions.csv")
    d = d[d.path.isin([e["path"] for e in ex])]
    (OUT / "reports").mkdir(exist_ok=True)
    d.to_csv(OUT / "reports/q3_test_predictions.csv", index=False)
    for p in d.path:
        cp(p)

    # Docker + Space files
    dep = ROOT / "deploy"
    shutil.copy2(dep / "Dockerfile", OUT / "Dockerfile")
    shutil.copy2(dep / "requirements-deploy.txt", OUT / "requirements-deploy.txt")
    shutil.copy2(dep / "README_SPACE.md", OUT / "README.md")
    shutil.copy2(dep / "dockerignore.txt", OUT / ".dockerignore")
    (OUT / ".gitignore").write_text("__pycache__/\nlogs/\n")
    (OUT / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.npz filter=lfs diff=lfs merge=lfs -text\n"
                                        "*.wav filter=lfs diff=lfs merge=lfs -text\n")

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file() and ".git" not in f.parts) / 1e6
    print(f"Space folder ready: {OUT}  ({size:.0f} MB, {len(d)} demo clips, "
          f"{len(list((OUT / 'models').glob('*.pt')))} model files)")

    # smoke test: start the copied app on CPU and call the health endpoint
    code = ("import os; os.environ['ACOUSTIGUARD_DEVICE']='cpu';"
            "from fastapi.testclient import TestClient; from app.main import app;"
            "c=TestClient(app).__enter__(); h=c.get('/api/health').json(); e=c.get('/api/examples').json();"
            "r=c.post('/api/examples/'+e[0]['id']+'/explain').json();"
            "print('smoke test:', h['status'], h['device'], len(e), 'examples, score', r['score'],"
            " 'explain', r['explanation']['method'], r['explanation']['layer'])")
    subprocess.run([sys.executable, "-c", code], cwd=OUT, check=True)


if __name__ == "__main__":
    main()
