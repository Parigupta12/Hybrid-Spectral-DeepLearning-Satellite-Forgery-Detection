import os
import numpy as np
import matplotlib.pyplot as plt

output_dir = "outputs/figures"
os.makedirs(output_dir, exist_ok=True)

# Cross-region Jaipur result
cm = np.array([
    [42, 8],
    [27, 23]
])

labels = ["Real", "Manipulated"]

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm, cmap="Blues")

ax.set_title(
    "Cross-Region Validation Confusion Matrix",
    fontsize=12,
    fontweight="bold",
    pad=10
)

ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))

ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)

# Add numbers
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = cm[i, j]
        ax.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            color="white" if value > cm.max() / 2 else "black",
            fontsize=14,
            fontweight="bold"
        )

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel("Number of Samples", rotation=270, labelpad=15)

plt.tight_layout()

output_path = os.path.join(output_dir, "figure6_cross_region_confusion_matrix.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print("✅ Figure 6 saved at:", output_path)