import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

val_csv = "data/val.csv"
test_csv = "data/test.csv"

output_dir = "outputs/spectral_branch"
os.makedirs(output_dir, exist_ok=True)


def compute_spectral_anomaly_score(patch):
    """
    Final spectral anomaly score.
    Patch shape: (6, 64, 64)
    Channels: [R, G, B, NDVI, NDWI, NDBI]
    """

    R = patch[0]
    G = patch[1]
    B = patch[2]
    NDVI = patch[3]
    NDWI = patch[4]
    NDBI = patch[5]

    brightness = (R + G + B) / 3.0

    # 1. Spectral contradiction rules
    vegetation_contradiction = ((G > R) & (G > B) & (NDVI < 0.25)).astype("float32")
    water_contradiction = ((B > R) & (NDWI < 0.05)).astype("float32")
    builtup_contradiction = ((brightness > 0.08) & (NDBI < -0.05)).astype("float32")

    rule_score = (
        vegetation_contradiction.mean()
        + water_contradiction.mean()
        + builtup_contradiction.mean()
    ) / 3.0

    # 2. Extreme spectral values
    ndvi_extreme = np.mean((NDVI < -0.2) | (NDVI > 0.85))
    ndwi_extreme = np.mean((NDWI < -0.75) | (NDWI > 0.45))
    ndbi_extreme = np.mean((NDBI < -0.50) | (NDBI > 0.50))

    extreme_score = (ndvi_extreme + ndwi_extreme + ndbi_extreme) / 3.0

    # 3. Local spectral variation
    spectral_stack = np.stack([NDVI, NDWI, NDBI], axis=0)
    local_std = np.std(spectral_stack, axis=(1, 2)).mean()
    local_score = min(local_std / 0.35, 1.0)

    # Final combined score
    final_score = (
        0.50 * rule_score
        + 0.30 * extreme_score
        + 0.20 * local_score
    )

    return float(final_score)


def get_scores(csv_path):
    df = pd.read_csv(csv_path)

    scores = []
    labels = []
    paths = []

    for _, row in df.iterrows():
        patch = np.load(row["patch_path"]).astype("float32")
        score = compute_spectral_anomaly_score(patch)

        scores.append(score)
        labels.append(int(row["label"]))
        paths.append(row["patch_path"])

    return df, np.array(scores), np.array(labels), paths


def find_best_threshold(scores, labels):
    best_threshold = None
    best_f1 = -1
    best_acc = -1

    thresholds = np.linspace(scores.min(), scores.max(), 300)

    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)

        f1 = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)

        if (f1 > best_f1) or (f1 == best_f1 and acc > best_acc):
            best_f1 = f1
            best_acc = acc
            best_threshold = threshold

    return best_threshold, best_f1, best_acc


print("📥 Loading validation data...")
val_df, val_scores, val_labels, _ = get_scores(val_csv)

best_threshold, best_val_f1, best_val_acc = find_best_threshold(
    val_scores,
    val_labels
)

val_preds = (val_scores >= best_threshold).astype(int)
val_auc = roc_auc_score(val_labels, val_scores)

print("\n✅ Validation threshold selected")
print("Threshold:", best_threshold)
print("Validation Accuracy:", round(best_val_acc, 4))
print("Validation F1:", round(best_val_f1, 4))
print("Validation ROC-AUC:", round(val_auc, 4))

print("\n📥 Loading test data...")
test_df, test_scores, test_labels, test_paths = get_scores(test_csv)

test_preds = (test_scores >= best_threshold).astype(int)

report = classification_report(test_labels, test_preds, digits=4, zero_division=0)
cm = confusion_matrix(test_labels, test_preds)

test_acc = accuracy_score(test_labels, test_preds)
test_precision = precision_score(test_labels, test_preds, zero_division=0)
test_recall = recall_score(test_labels, test_preds, zero_division=0)
test_f1 = f1_score(test_labels, test_preds, zero_division=0)
test_auc = roc_auc_score(test_labels, test_scores)

print("\n✅ Final Clean Step 8 Complete")
print("\nTest score range:")
print("Min:", test_scores.min())
print("Max:", test_scores.max())
print("Mean:", test_scores.mean())

print("\nClassification Report:")
print(report)

print("Confusion Matrix:")
print(cm)

print("Accuracy:", round(test_acc, 4))
print("Precision:", round(test_precision, 4))
print("Recall:", round(test_recall, 4))
print("F1-score:", round(test_f1, 4))
print("ROC-AUC:", round(test_auc, 4))


results_df = pd.DataFrame({
    "patch_path": test_paths,
    "label": test_labels,
    "spectral_score": test_scores,
    "prediction": test_preds,
    "threshold": best_threshold
})

results_df.to_csv(
    os.path.join(output_dir, "spectral_scores.csv"),
    index=False
)

with open(os.path.join(output_dir, "spectral_branch_results.txt"), "w") as f:
    f.write("Final Clean Spectral Anomaly Branch Results\n")
    f.write("===========================================\n\n")

    f.write("Validation-based threshold selection:\n")
    f.write(f"Threshold: {best_threshold}\n")
    f.write(f"Validation Accuracy: {round(best_val_acc, 4)}\n")
    f.write(f"Validation F1: {round(best_val_f1, 4)}\n")
    f.write(f"Validation ROC-AUC: {round(val_auc, 4)}\n\n")

    f.write("Final test results:\n")
    f.write(f"Accuracy: {round(test_acc, 4)}\n")
    f.write(f"Precision: {round(test_precision, 4)}\n")
    f.write(f"Recall: {round(test_recall, 4)}\n")
    f.write(f"F1-score: {round(test_f1, 4)}\n")
    f.write(f"ROC-AUC: {round(test_auc, 4)}\n\n")

    f.write("Classification Report:\n")
    f.write(report)

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("\nSaved files:")
print(os.path.join(output_dir, "spectral_scores.csv"))
print(os.path.join(output_dir, "spectral_branch_results.txt"))