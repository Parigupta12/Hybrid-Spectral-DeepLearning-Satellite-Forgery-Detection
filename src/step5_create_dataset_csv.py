import os
import pandas as pd

real_folder = "data/patches/real"
fake_folder = "data/patches/fake"
output_csv = "data/dataset.csv"

rows = []

for file in os.listdir(real_folder):
    if file.endswith(".npy"):
        rows.append({
            "patch_path": os.path.join(real_folder, file),
            "label": 0
        })

for file in os.listdir(fake_folder):
    if file.endswith(".npy"):
        rows.append({
            "patch_path": os.path.join(fake_folder, file),
            "label": 1
        })

df = pd.DataFrame(rows)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(output_csv, index=False)

print("✅ Step 5 complete.")
print("Dataset CSV saved at:", output_csv)
print("Total samples:", len(df))
print("Class distribution:")
print(df["label"].value_counts())