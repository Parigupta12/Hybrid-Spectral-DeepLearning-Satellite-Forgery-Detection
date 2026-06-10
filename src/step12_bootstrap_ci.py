import numpy as np
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
import pandas as pd
import os

# ============================================

# ============================================

jaipur_csv = "outputs/hybrid_feature_fusion/cross_region_predictions.csv"

if os.path.exists(jaipur_csv):
    # Real predictions file se load karo
    df = pd.read_csv(jaipur_csv)
    y_true = df["label"].values
    y_pred = df["prediction"].values
    print("✅ Jaipur predictions file se load ki!")

else:
    # Fallback — confusion matrix manually
    print("⚠️ File nahi mili, confusion matrix se use kar raha hoon...")
    y_true = np.array([0]*50 + [1]*50)
    y_pred = np.array([0]*42 + [1]*8 +
                      [0]*27 + [1]*23)

# ============================================
# Bootstrap — 1000 iterations, seed=42
# ============================================
np.random.seed(42)

n_iter = 1000
metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": []}

for _ in range(n_iter):
    idx = resample(np.arange(len(y_true)), replace=True, n_samples=len(y_true))
    yt = y_true[idx]
    yp = y_pred[idx]

    metrics["Accuracy"].append(accuracy_score(yt, yp))
    metrics["Precision"].append(precision_score(yt, yp, zero_division=0))
    metrics["Recall"].append(recall_score(yt, yp, zero_division=0))
    metrics["F1-Score"].append(f1_score(yt, yp, zero_division=0))

# ============================================
# Results Print + Save
# ============================================
print("\n" + "="*55)
print("  95% Bootstrap Confidence Intervals")
print("  (n=1000 iterations, fixed seed=42)")
print("="*55)

results = {}
for name, scores in metrics.items():
    lo  = np.percentile(scores, 2.5)
    hi  = np.percentile(scores, 97.5)
    mn  = np.mean(scores)
    results[name] = (mn, lo, hi)
    print(f"{name:12s}: {mn*100:.2f}%  [{lo*100:.2f}% – {hi*100:.2f}%]")

print("="*55)
print("\n📋 Manuscript mein copy karo (Section 1.5.5):\n")
for name, (mn, lo, hi) in results.items():
    print(f"{name}: {mn*100:.2f}% (95% CI: {lo*100:.2f}%–{hi*100:.2f}%)")

# Save to file
os.makedirs("outputs/bootstrap", exist_ok=True)
with open("outputs/bootstrap/bootstrap_ci_results.txt", "w") as f:
    f.write("95% Bootstrap Confidence Intervals\n")
    f.write("n=1000 iterations, seed=42\n")
    f.write("="*40 + "\n")
    for name, (mn, lo, hi) in results.items():
        f.write(f"{name}: {mn*100:.2f}% [{lo*100:.2f}% - {hi*100:.2f}%]\n")
    f.write("\nManuscript text:\n")
    for name, (mn, lo, hi) in results.items():
        f.write(f"{name}: {mn*100:.2f}% (95% CI: {lo*100:.2f}%–{hi*100:.2f}%)\n")

print("\n✅ Results saved: outputs/bootstrap/bootstrap_ci_results.txt")