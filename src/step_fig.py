import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Use clean publication-style background
plt.style.use('default')

# Confusion matrices
matrices = {
    "CNN-only Model": np.array([[37, 1],
                                [27, 10]]),

    "Spectral-only Model": np.array([[14, 24],
                                     [4, 33]]),

    "Hybrid Fusion Model": np.array([[31, 7],
                                     [12, 25]])
}

# Better scientific labels
labels = ["Real", "Manipulated"]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

# Pure white background
fig.patch.set_facecolor('white')

# Plot matrices
for i, (ax, (title, cm)) in enumerate(zip(axes, matrices.items())):

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d"
    )

    # Titles
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Remove repeated labels
    if i == 0:
        ax.set_ylabel("True Label", fontsize=11)
    else:
        ax.set_ylabel("")

    ax.set_xlabel("Predicted Label", fontsize=11)

    # Equal square cells
    ax.set_aspect('equal')

    # Make numbers bold and larger
    for text in disp.text_.ravel():
        text.set_fontsize(13)
        text.set_fontweight("bold")

# Main title
plt.suptitle(
    "Internal Evaluation Confusion Matrices",
    fontsize=16,
    fontweight="bold"
)

# Adjust spacing
plt.tight_layout()

# Save high-quality figure
plt.savefig(
    "internal_model_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight",
    facecolor='white'
)

plt.show()