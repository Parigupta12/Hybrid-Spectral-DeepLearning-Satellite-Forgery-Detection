import os
import numpy as np
import matplotlib.pyplot as plt

real_folder = "data/patches/real"
fake_folder = "data/patches/fake"
output_dir = "outputs/figures"

os.makedirs(output_dir, exist_ok=True)

# Final locked patch for Figure 3
PATCH_INDEX = 150

real_files = sorted([f for f in os.listdir(real_folder) if f.endswith(".npy")])
fake_files = sorted([f for f in os.listdir(fake_folder) if f.endswith(".npy")])

real_path = os.path.join(real_folder, real_files[PATCH_INDEX])
fake_path = os.path.join(fake_folder, fake_files[PATCH_INDEX])

print("Using real patch:", real_path)
print("Using fake patch:", fake_path)

real_patch = np.load(real_path).astype("float32")
fake_patch = np.load(fake_path).astype("float32")


def make_rgb(patch):
    rgb = np.stack([patch[0], patch[1], patch[2]], axis=-1)

    p2 = np.percentile(rgb, 2)
    p98 = np.percentile(rgb, 98)

    rgb = (rgb - p2) / (p98 - p2 + 1e-8)
    rgb = np.clip(rgb, 0, 1)

    return rgb


real_rgb = make_rgb(real_patch)
fake_rgb = make_rgb(fake_patch)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# RGB
axes[0, 0].imshow(real_rgb)
axes[1, 0].imshow(fake_rgb)

axes[0, 0].set_title("RGB", fontweight="bold")
axes[1, 0].set_title("RGB", fontweight="bold")

axes[0, 0].axis("off")
axes[1, 0].axis("off")

configs = [
    ("NDVI", 3, "RdYlGn"),
    ("NDWI", 4, "Blues"),
    ("NDBI", 5, "YlOrRd")
]

colorbars = []

for i, (name, ch, cmap) in enumerate(configs, start=1):

    im_real = axes[0, i].imshow(
        real_patch[ch],
        cmap=cmap,
        vmin=-1,
        vmax=1
    )

    axes[1, i].imshow(
        fake_patch[ch],
        cmap=cmap,
        vmin=-1,
        vmax=1
    )

    axes[0, i].set_title(name, fontweight="bold")
    axes[1, i].set_title(name, fontweight="bold")

    axes[0, i].axis("off")
    axes[1, i].axis("off")

    colorbars.append(im_real)

# Row labels
fig.text(
    0.04,
    0.72,
    "Real",
    rotation=90,
    fontsize=13,
    fontweight="bold",
    va="center"
)

fig.text(
    0.04,
    0.28,
    "Manipulated",
    rotation=90,
    fontsize=13,
    fontweight="bold",
    va="center"
)

# Horizontal colorbars
cbar_ax1 = fig.add_axes([0.25, 0.055, 0.15, 0.02])
cbar_ax2 = fig.add_axes([0.47, 0.055, 0.15, 0.02])
cbar_ax3 = fig.add_axes([0.69, 0.055, 0.15, 0.02])

cb1 = plt.colorbar(colorbars[0], cax=cbar_ax1, orientation="horizontal")
cb2 = plt.colorbar(colorbars[1], cax=cbar_ax2, orientation="horizontal")
cb3 = plt.colorbar(colorbars[2], cax=cbar_ax3, orientation="horizontal")

cb1.set_label("NDVI value", fontsize=9)
cb2.set_label("NDWI value", fontsize=9)
cb3.set_label("NDBI value", fontsize=9)

cb1.ax.tick_params(labelsize=8)
cb2.ax.tick_params(labelsize=8)
cb3.ax.tick_params(labelsize=8)

plt.suptitle(
    "Spectral Comparison Between Real and Manipulated Satellite Patches",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout(rect=[0.05, 0.11, 1, 0.92])

save_path = os.path.join(
    output_dir,
    "figure3_spectral_comparison_final.png"
)

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print("\n✅ Final Figure 3 saved at:", save_path)