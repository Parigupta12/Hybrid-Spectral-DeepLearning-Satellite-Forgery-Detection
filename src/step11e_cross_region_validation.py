import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ==============================
# Paths
# ==============================
train_csv = "data/train.csv"
val_csv = "data/val.csv"
roi2_csv = "data_roi2/roi2_test.csv"

cnn_model_path = "outputs/cnn_baseline/cnn_baseline_model.keras"
output_dir = "outputs/cross_region"
os.makedirs(output_dir, exist_ok=True)


# ==============================
# Spectral feature extractor
# MUST match Step 9
# ==============================
def extract_spectral_features(patch):
    """
    patch shape: (6, 64, 64)
    channels: [R, G, B, NDVI, NDWI, NDBI]
    """

    R, G, B = patch[0], patch[1], patch[2]
    NDVI, NDWI, NDBI = patch[3], patch[4], patch[5]

    brightness = (R + G + B) / 3.0

    veg_contradiction = np.mean((G > R) & (G > B) & (NDVI < 0.25))
    water_contradiction = np.mean((B > R) & (NDWI < 0.05))
    builtup_contradiction = np.mean((brightness > 0.08) & (NDBI < -0.05))

    ndvi_extreme = np.mean((NDVI < -0.2) | (NDVI > 0.85))
    ndwi_extreme = np.mean((NDWI < -0.75) | (NDWI > 0.45))
    ndbi_extreme = np.mean((NDBI < -0.50) | (NDBI > 0.50))

    ndvi_mean = NDVI.mean()
    ndvi_std = NDVI.std()
    ndwi_mean = NDWI.mean()
    ndwi_std = NDWI.std()
    ndbi_mean = NDBI.mean()
    ndbi_std = NDBI.std()

    spectral_stack = np.stack([NDVI, NDWI, NDBI], axis=0)
    local_std = np.std(spectral_stack, axis=(1, 2)).mean()

    spectral_score = (
        0.50 * ((veg_contradiction + water_contradiction + builtup_contradiction) / 3.0)
        + 0.30 * ((ndvi_extreme + ndwi_extreme + ndbi_extreme) / 3.0)
        + 0.20 * min(local_std / 0.35, 1.0)
    )

    return np.array([
        spectral_score,
        veg_contradiction,
        water_contradiction,
        builtup_contradiction,
        ndvi_extreme,
        ndwi_extreme,
        ndbi_extreme,
        ndvi_mean,
        ndvi_std,
        ndwi_mean,
        ndwi_std,
        ndbi_mean,
        ndbi_std,
        local_std
    ], dtype="float32")


# ==============================
# CNN feature extractor
# ==============================
def build_cnn_feature_extractor():
    if not os.path.exists(cnn_model_path):
        raise FileNotFoundError("CNN baseline model not found. Run Step 7 first.")

    cnn_model = models.load_model(cnn_model_path)

    dummy_input = np.zeros((1, 64, 64, 6), dtype="float32")
    _ = cnn_model(dummy_input)

    feature_layer_output = cnn_model.layers[-3].output

    feature_extractor = models.Model(
        inputs=cnn_model.inputs,
        outputs=feature_layer_output
    )

    return feature_extractor


# ==============================
# Load fused features
# ==============================
def load_hybrid_features(csv_path, feature_extractor):
    df = pd.read_csv(csv_path)

    fused_features = []
    labels = []
    paths = []

    for _, row in df.iterrows():
        patch = np.load(row["patch_path"]).astype("float32")

        patch_tf = np.transpose(patch, (1, 2, 0))
        patch_tf = np.expand_dims(patch_tf, axis=0)

        cnn_features = feature_extractor.predict(patch_tf, verbose=0).ravel()
        spectral_features = extract_spectral_features(patch)

        fused = np.concatenate([cnn_features, spectral_features])

        fused_features.append(fused)
        labels.append(int(row["label"]))
        paths.append(row["patch_path"])

    return np.array(fused_features), np.array(labels), paths


# ==============================
# Threshold selection on Agra validation only
# ==============================
def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_macro_f1 = -1

    for threshold in np.linspace(0.10, 0.90, 200):
        preds = (y_prob >= threshold).astype(int)
        macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold

    return best_threshold, best_macro_f1


# ==============================
# Main
# ==============================
print("📥 Loading CNN feature extractor...")
feature_extractor = build_cnn_feature_extractor()

print("\n📥 Loading Agra training and validation features...")
X_train, y_train, _ = load_hybrid_features(train_csv, feature_extractor)
X_val, y_val, _ = load_hybrid_features(val_csv, feature_extractor)

print("Agra train shape:", X_train.shape)
print("Agra val shape:", X_val.shape)

print("\n📥 Loading Jaipur ROI-2 external test features...")
X_roi2, y_roi2, roi2_paths = load_hybrid_features(roi2_csv, feature_extractor)

print("Jaipur ROI-2 shape:", X_roi2.shape)
print("Jaipur labels:", np.bincount(y_roi2))

print("\n⚙️ Scaling features using Agra training data only...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_roi2_scaled = scaler.transform(X_roi2)

print("\n🚀 Training fusion classifier on Agra training data only...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train_scaled, y_train)

print("\n🔍 Selecting threshold using Agra validation set only...")
val_prob = rf.predict_proba(X_val_scaled)[:, 1]
best_threshold, best_val_macro_f1 = find_best_threshold(y_val, val_prob)

print("Best Agra validation threshold:", round(best_threshold, 4))
print("Agra validation Macro-F1:", round(best_val_macro_f1, 4))

print("\n🧪 Running external cross-region validation on Jaipur ROI-2...")
roi2_prob = rf.predict_proba(X_roi2_scaled)[:, 1]
roi2_pred = (roi2_prob >= best_threshold).astype(int)

report = classification_report(y_roi2, roi2_pred, digits=4, zero_division=0)
cm = confusion_matrix(y_roi2, roi2_pred)

acc = accuracy_score(y_roi2, roi2_pred)
precision = precision_score(y_roi2, roi2_pred, zero_division=0)
recall = recall_score(y_roi2, roi2_pred, zero_division=0)
f1 = f1_score(y_roi2, roi2_pred, zero_division=0)
auc = roc_auc_score(y_roi2, roi2_prob)

print("\n✅ Step 11E Complete: Cross-region validation done")

print("\nClassification Report:")
print(report)

print("Confusion Matrix:")
print(cm)

print("Accuracy:", round(acc, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("ROC-AUC:", round(auc, 4))

results_df = pd.DataFrame({
    "patch_path": roi2_paths,
    "label": y_roi2,
    "probability": roi2_prob,
    "prediction": roi2_pred,
    "threshold": best_threshold
})

results_df.to_csv(
    os.path.join(output_dir, "roi2_jaipur_predictions.csv"),
    index=False
)

with open(os.path.join(output_dir, "cross_region_results.txt"), "w") as f:
    f.write("External Cross-Region Validation Results\n")
    f.write("========================================\n\n")
    f.write("Training region: Agra\n")
    f.write("External testing region: Jaipur ROI-2\n\n")
    f.write(f"Agra validation threshold: {best_threshold}\n")
    f.write(f"Agra validation Macro-F1: {round(best_val_macro_f1, 4)}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nAccuracy: ")
    f.write(str(round(acc, 4)))
    f.write("\nPrecision: ")
    f.write(str(round(precision, 4)))
    f.write("\nRecall: ")
    f.write(str(round(recall, 4)))
    f.write("\nF1-score: ")
    f.write(str(round(f1, 4)))
    f.write("\nROC-AUC: ")
    f.write(str(round(auc, 4)))

print("\nSaved in:", output_dir)