"""
Step 3: create THE single train / validation / test split (used by every model).

Rule (src/config.py -> SPLIT_BY_MACHINE_ID): every clip of one physical machine
goes to exactly one split. Output: data/processed/splits.csv  (commit this file
to Git - it is what makes your results reproducible).

Run:  python -m src.make_splits
"""
import pandas as pd

from src import config


def main() -> None:
    df = pd.read_csv(config.METADATA_CSV)
    df = df[df["readable"]].copy()
    df["split"] = df["machine_id"].map(config.SPLIT_BY_MACHINE_ID)

    unknown = df["split"].isna()
    if unknown.any():
        raise SystemExit(f"Machine IDs not assigned in config: {sorted(df.loc[unknown, 'machine_id'].unique())}")

    # Hard guarantee: no machine appears in more than one split
    per_group = df.groupby("group")["split"].nunique()
    assert (per_group == 1).all(), f"Leakage! groups in several splits: {per_group[per_group > 1]}"

    cols = ["path", "split", "group", "machine_type", "machine_id", "label", "label_name", "snr"]
    df[cols].sort_values(["split", "group", "label", "path"]).to_csv(config.SPLITS_CSV, index=False)

    summary = pd.crosstab([df.split, df.machine_type], df.label_name, margins=True)
    summary["abnormal_%"] = (100 * summary["abnormal"] / summary["All"]).round(1)
    print(summary)
    print(f"\nSaved -> {config.SPLITS_CSV.relative_to(config.ROOT)}")
    print("Next step:  python -m src.leakage_checks")


if __name__ == "__main__":
    main()
