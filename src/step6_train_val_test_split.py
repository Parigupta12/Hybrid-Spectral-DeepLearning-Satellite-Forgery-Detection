import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_csv = "data/dataset.csv"

train_csv = "data/train.csv"
val_csv = "data/val.csv"
test_csv = "data/test.csv"

# Load dataset
df = pd.read_csv(input_csv)

# Check required columns
required_columns = {"patch_path", "label"}
if not required_columns.issubset(df.columns):
    raise ValueError("dataset.csv must contain patch_path and label columns")

# First split: 70% train, 30% temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

# Second split: temp into 15% val, 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

# Save files
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

print("✅ Step 6 complete.")
print("Files saved:")
print("Train:", train_csv)
print("Validation:", val_csv)
print("Test:", test_csv)

print("\nDataset sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nClass distribution:")
print("\nTrain:")
print(train_df["label"].value_counts())

print("\nValidation:")
print(val_df["label"].value_counts())

print("\nTest:")
print(test_df["label"].value_counts())