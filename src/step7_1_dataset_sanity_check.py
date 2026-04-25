import os
import numpy as np
import pandas as pd

for csv_path in ["data/train.csv", "data/val.csv", "data/test.csv"]:
    df = pd.read_csv(csv_path)

    print("\n==============================")
    print(csv_path)
    print("==============================")
    print(df["label"].value_counts())

    for label in [0, 1]:
        sub = df[df["label"] == label].head(5)
        print(f"\nLabel {label} sample paths:")
        for p in sub["patch_path"]:
            print(p, "exists:", os.path.exists(p))

    means = []
    for _, row in df.iterrows():
        patch = np.load(row["patch_path"]).astype("float32")
        means.append([
            row["label"],
            patch[:3].mean(),   # RGB mean
            patch[3].mean(),    # NDVI mean
            patch[4].mean(),    # NDWI mean
            patch[5].mean()     # NDBI mean
        ])

    stats = pd.DataFrame(means, columns=["label", "rgb_mean", "ndvi_mean", "ndwi_mean", "ndbi_mean"])
    print("\nMean stats by label:")
    print(stats.groupby("label").mean())