"""
Step 0: build a balanced, manageable sample of CICIoT2023.

train.csv is 1.6 GB, so it is streamed in chunks and capped at N rows per class.
test.csv supplies the untouched evaluation set (and the demo traffic in the app).

    python -m ids.prepare_data --train-per-class 6000 --test-per-class 2000
"""

import argparse

import numpy as np
import pandas as pd

from ids.config import PROCESSED_DIR, RAW_DIR, RANDOM_SEED, TEST_SAMPLE, TRAIN_SAMPLE


def sample_per_class(csv_path, per_class: int, chunksize: int = 500_000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    kept: dict[str, list[pd.DataFrame]] = {}
    counts: dict[str, int] = {}
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        for label, grp in chunk.groupby("label"):
            have = counts.get(label, 0)
            if have >= per_class:
                continue
            # Take a random part of every chunk so the sample is spread over the file.
            take = min(len(grp), per_class - have, max(1, per_class // 4))
            idx = rng.choice(len(grp), size=take, replace=False)
            kept.setdefault(label, []).append(grp.iloc[idx])
            counts[label] = have + take
        print(f"  chunk {i + 1}: {len(counts)} classes, min={min(counts.values())}")
        if len(counts) >= 34 and min(counts.values()) >= per_class:
            break
    df = pd.concat([pd.concat(v) for v in kept.values()], ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-class", type=int, default=6000)
    ap.add_argument("--test-per-class", type=int, default=2000)
    args = ap.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Sampling train.csv ...")
    tr = sample_per_class(RAW_DIR / "train.csv", args.train_per_class)
    tr.to_parquet(TRAIN_SAMPLE, index=False)
    print(f"train sample: {tr.shape}")

    print("Sampling test.csv ...")
    te = sample_per_class(RAW_DIR / "test.csv", args.test_per_class)
    te.to_parquet(TEST_SAMPLE, index=False)
    print(f"test sample: {te.shape}")
    print(te["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
