import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# Setup
# ==============================
output_dir = "outputs/comparison"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# Final Results
# ==============================
results = [
    {
        "Model": "CNN Only",
        "Accuracy": 0.6267,
        "Precision": 0.9091,
        "Recall": 0.2703,
        "F1-score": 0.4167,
        "ROC-AUC": 0.6216
    },
    {
        "Model": "Spectral Only",
        "Accuracy": 0.6267,
        "Precision": 0.5789,
        "Recall": 0.8919,
        "F1-score": 0.7021,
        "ROC-AUC": 0.7262
    },
    {
        "Model": "Hybrid Feature Fusion",
        "Accuracy": 0.7467,
        "Precision": 0.7812,
        "Recall": 0.6757,
        "F1-score": 0.7246,
        "ROC-AUC": 0.7319
    }
]

df = pd.DataFrame(results)

csv_path = os.path.join(output_dir, "model_comparison_results.csv")
df.to_csv(csv_path, index=False)

print("✅ Step 10 complete.")
print("\nComparative Results:")
print(df)
print("\nSaved CSV:", csv_path)

# ==============================
# Research-ready style
# ==============================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 300
})

colors_2 = ["#4C72B0", "#DD8452"]                 # muted blue, muted orange
colors_3 = ["#4C72B0", "#55A868", "#C44E52"]      # muted blue, green, red


def grouped_bar_plot(metrics, title, filename):
    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.34 if len(metrics) == 2 else 0.24

    fig, ax = plt.subplots(figsize=(9, 5.4))

    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * width
        color = colors_2[i] if len(metrics) == 2 else colors_3[i]

        bars = ax.bar(
            x + offset,
            df[metric] * 100,  # Convert to percentage
            width,
            label=metric,
            color=color,
            edgecolor="black",
            linewidth=0.4
        )

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.8,  # Adjusted offset for percentage scale
                f"{height:.1f}%",  # Display as percentage
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold"
            )

    ax.set_ylim(0, 105)  # 0-100% with some padding
    ax.set_ylabel("Score (%)")  # Updated label
    ax.set_title(title, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=8, ha="center")

    # Y-axis ticks in percentage
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=True,
        loc="upper right",
        borderpad=0.6
    )

    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)


# ==============================
# Figure 1: Accuracy + ROC-AUC
# ==============================
grouped_bar_plot(
    metrics=["Accuracy", "ROC-AUC"],
    title="Overall Performance Comparison Across Models",
    filename="figure1_accuracy_roc_auc_research_ready.png"
)

# ==============================
# Figure 2: Precision + Recall + F1-score
# ==============================
grouped_bar_plot(
    metrics=["Precision", "Recall", "F1-score"],
    title="Precision–Recall Trade-off Across Models",
    filename="figure2_precision_recall_f1_research_ready.png"
)

print("\n✅ Research-ready comparison figures saved.")
