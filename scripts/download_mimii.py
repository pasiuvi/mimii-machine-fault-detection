"""
Download and extract the MIMII dataset from Zenodo (reproducible download procedure).

Dataset : MIMII - Sound Dataset for Malfunctioning Industrial Machine
          Investigation and Inspection (Hitachi, Ltd.)
DOI     : 10.5281/zenodo.3384388
Licence : CC BY-SA 4.0

Usage (run from the project root, with the virtual environment active):
    python -m scripts.download_mimii                       # uses SNR + machines in src/config.py
    python -m scripts.download_mimii --machines pump       # just one machine type
    python -m scripts.download_mimii --snr 6_dB --machines pump fan --keep-zip
    python -m scripts.download_mimii --list                # show files & sizes, download nothing

Features:
  * resumes an interrupted download (important for big files on home internet)
  * verifies every zip with the official MD5 checksum from Zenodo
  * extracts to data/raw/<snr>_<machine>/ so different noise levels never overwrite each other
  * deletes the zip after successful extraction (unless --keep-zip) to save disk space
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

BASE_URL = f"https://zenodo.org/records/{config.ZENODO_RECORD}/files"

# Official file list from https://zenodo.org/records/3384388 (size in GB, md5)
FILES = {
    "-6_dB_fan.zip":    (10.9, "f02ae808a58d84b6815b7ec38ff30879"),
    "-6_dB_pump.zip":   (8.2,  "d20b783a0ff9c93d58f452f98c37b112"),
    "-6_dB_slider.zip": (8.0,  "49913eda7d37f182cbf8ed5c984140e0"),
    "-6_dB_valve.zip":  (8.0,  "fdfaf185fea61b21e11952a070a4ada7"),
    "0_dB_fan.zip":     (10.4, "6354d1cc2165c52168f9ef1bcd9c7c52"),
    "0_dB_pump.zip":    (7.9,  "488748295c3f60b25de07b58fe75b049"),
    "0_dB_slider.zip":  (7.5,  "4d674c21474f0646ecd75546db6c0c4e"),
    "0_dB_valve.zip":   (7.5,  "178478eb0d11c79080a35562bfdeee71"),
    "6_dB_fan.zip":     (10.2, "0890f7d3c2fd8448634e69ff1d66dd47"),
    "6_dB_pump.zip":    (7.7,  "a09ba6060c10fc09cd4c8770213b0b9f"),
    "6_dB_slider.zip":  (7.1,  "838c2b3441858359c4704ef13a1b27ff"),
    "6_dB_valve.zip":   (6.9,  "fe5fb7c337cd701b1d31dc641e621892"),
}

CHUNK = 1024 * 1024  # 1 MB


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in tqdm(iter(lambda: f.read(CHUNK), b""),
                          total=path.stat().st_size // CHUNK + 1,
                          desc=f"  checking MD5 {path.name}", unit="MB", leave=False):
            h.update(block)
    return h.hexdigest()


def download(name: str, dest: Path, retries: int = 10) -> None:
    """Stream a file to disk, resuming from where it stopped if interrupted."""
    url = f"{BASE_URL}/{name}?download=1"
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        done = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                if done and r.status_code != 206:  # server ignored resume -> start over
                    done = 0
                    part.unlink(missing_ok=True)
                total = int(r.headers.get("Content-Length", 0)) + done
                mode = "ab" if done else "wb"
                with open(part, mode) as f, tqdm(total=total, initial=done, unit="B",
                                                 unit_scale=True, desc=f"  {name}") as bar:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        bar.update(len(chunk))
            break
        except (requests.RequestException, ConnectionError) as e:
            wait = min(60, 5 * attempt)
            print(f"  connection problem ({e.__class__.__name__}); retry {attempt}/{retries} in {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Could not download {name} after {retries} attempts.")
    part.rename(dest)


def extract(zip_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = z.infolist()
        for m in tqdm(members, desc=f"  extracting {zip_path.name}", unit="file"):
            z.extract(m, out_dir)
    return sum(1 for _ in out_dir.rglob("*.wav"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snr", default=config.SNR, choices=["-6_dB", "0_dB", "6_dB"])
    p.add_argument("--machines", nargs="+", default=config.MACHINE_TYPES,
                   choices=["fan", "pump", "slider", "valve"])
    p.add_argument("--keep-zip", action="store_true", help="keep the .zip after extracting")
    p.add_argument("--list", action="store_true", help="only list what would be downloaded")
    args = p.parse_args()

    names = [f"{args.snr}_{m}.zip" for m in args.machines]
    total_gb = sum(FILES[n][0] for n in names)
    print(f"Files: {names}  (about {total_gb:.1f} GB to download)")
    if args.list:
        return

    zips_dir = config.DATA_DIR / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(config.DATA_DIR).free / 1e9
    need_gb = total_gb * 2.1  # zip + extracted copy
    if free_gb < need_gb:
        print(f"WARNING: only {free_gb:.0f} GB free; about {need_gb:.0f} GB needed while extracting.")
        print("         Download one machine at a time (e.g. --machines pump), or free some space.")
        if input("Continue anyway? [y/N] ").strip().lower() != "y":
            return

    for name in names:
        size_gb, expected_md5 = FILES[name]
        out_dir = config.RAW_DIR / name.removesuffix(".zip")      # e.g. data/raw/6_dB_pump
        if out_dir.exists() and any(out_dir.rglob("*.wav")):
            print(f"[skip] {name}: already extracted in {out_dir.relative_to(config.ROOT)}")
            continue

        zip_path = zips_dir / name
        print(f"\n[1/3] Downloading {name} (~{size_gb} GB)")
        if not zip_path.exists():
            download(name, zip_path)

        print("[2/3] Verifying MD5 checksum")
        got = md5_of(zip_path)
        if got != expected_md5:
            zip_path.unlink()
            raise SystemExit(f"MD5 mismatch for {name} (got {got}). Deleted - please run again.")
        print("  OK - file is identical to the official Zenodo file")

        print("[3/3] Extracting")
        n = extract(zip_path, out_dir)
        print(f"  {n} wav files in {out_dir.relative_to(config.ROOT)}")
        if not args.keep_zip:
            zip_path.unlink()

    print("\nDone. Next step:  python -m src.build_metadata")


if __name__ == "__main__":
    main()
