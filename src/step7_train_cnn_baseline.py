import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

train_csv = "data/train.csv"
val_csv = "data/val.csv"
test_csv = "data/test.csv"

output_dir = "outputs/cnn_baseline"
os.makedirs(output_dir, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 25
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    X = []
    y = []

    for _, row in df.iterrows():
        patch = np.load(row["patch_path"]).astype("float32")  # (6, 64, 64)

        # Convert to TensorFlow format: (64, 64, 6)
        patch = np.transpose(patch, (1, 2, 0))

        X.append(patch)
        y.append(row["label"])

    return np.array(X, dtype="float32"), np.array(y, dtype="int32")


print("📥 Loading data...")
X_train, y_train = load_data(train_csv)
X_val, y_val = load_data(val_csv)
X_test, y_test = load_data(test_csv)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)


model = models.Sequential([
    layers.Input(shape=(64, 64, 6)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    os.path.join(output_dir, "cnn_baseline_model.keras"),
    monitor="val_loss",
    save_best_only=True
)

print("\n🚀 Training CNN baseline...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\n🧪 Evaluating on test set...")

y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nClassification Report:")
print(report)

print("Confusion Matrix:")
print(cm)

print("ROC-AUC:", round(auc, 4))

with open(os.path.join(output_dir, "cnn_baseline_results.txt"), "w") as f:
    f.write("CNN-Only Baseline Results\n")
    f.write("=========================\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nROC-AUC: ")
    f.write(str(round(auc, 4)))

print("\n✅ Step 7 complete.")
print("Results saved in:", output_dir)