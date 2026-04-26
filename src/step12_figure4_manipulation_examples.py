import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -------------------------
# Paths
# -------------------------
REAL_FOLDER = "data/patches/real"
OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Final patch choice
# -------------------------
PATCH_INDEX = 120

real_files = sorted([f for f in os.listdir(REAL_FOLDER) if f.endswith(".npy")])
patch_path = os.path.join(REAL_FOLDER, real_files[PATCH_INDEX])

print("Using patch:", patch_path)

patch = np.load(patch_path).astype(np.float32)

# -------------------------
# Helper: RGB conversion
# -------------------------
def make_rgb(patch):
    rgb = np.stack([patch[0], patch[1], patch[2]], axis=-1)

    p2 = np.percentile(rgb, 2)
    p98 = np.percentile(rgb, 98)

    rgb = (rgb - p2) / (p98 - p2 + 1e-8)
    rgb = np.clip(rgb, 0, 1)

    return rgb

# -------------------------
# Common manipulation region
# -------------------------
x1, x2 = 20, 42
y1, y2 = 20, 42
box_width = y2 - y1
box_height = x2 - x1

# -------------------------
# Original
# -------------------------
original = patch.copy()

# -------------------------
# 1. Copy-paste manipulation
# -------------------------
copy_patch = patch.copy()
copy_patch[:, x1:x2, y1:y2] = copy_patch[:, 5:27, 5:27]

# -------------------------
# 2. Localized noise injection
# -------------------------
noise_patch = patch.copy()
region = noise_patch[:3, x1:x2, y1:y2]

gaussian_noise = np.random.normal(
    loc=0,
    scale=0.012,
    size=region.shape
)

region = (region + gaussian_noise) * 0.94
noise_patch[:3, x1:x2, y1:y2] = region
noise_patch[:3] = np.clip(noise_patch[:3], 0, 1)

# -------------------------
# 3. Radiometric distortion
# -------------------------
radio_patch = patch.copy()
radio_patch[:3, x1:x2, y1:y2] *= 1.35
radio_patch[:3] = np.clip(radio_patch[:3], 0, 1)

# -------------------------
# 4. Spectral inconsistency
# RGB unchanged, only spectral channels changed
# -------------------------
spectral_patch = patch.copy()
spectral_patch[3, x1:x2, y1:y2] *= 0.50  # NDVI
spectral_patch[4, x1:x2, y1:y2] *= 1.35  # NDWI
spectral_patch[5, x1:x2, y1:y2] *= 1.45  # NDBI
spectral_patch[3:] = np.clip(spectral_patch[3:], -1, 1)

# -------------------------
# Convert images to RGB
# -------------------------
images = [
    make_rgb(original),
    make_rgb(copy_patch),
    make_rgb(noise_patch),
    make_rgb(radio_patch),
    make_rgb(spectral_patch),
]

titles = [
    "(a) Original",
    "(b) Copy-Paste",
    "(c) Localized Noise",
    "(d) Radiometric Distortion",
    "(e) Spectral Inconsistency",
]

# -------------------------
# Final centered layout
# -------------------------
fig = plt.figure(figsize=(14,8))

positions = [
    [0.05, 0.52, 0.26, 0.35],  # Original
    [0.37, 0.52, 0.26, 0.35],  # Copy-paste
    [0.69, 0.52, 0.26, 0.35],  # Noise

    [0.20, 0.08, 0.26, 0.35],  # Radiometric
    [0.54, 0.08, 0.26, 0.35],  # Spectral
]

for i, pos in enumerate(positions):
    ax = fig.add_axes(pos)

    ax.imshow(images[i])
    ax.set_title(
        titles[i],
        fontsize=12,
        fontweight="bold"
    )
    ax.axis("off")

    if i > 0:
        rect = Rectangle(
            (y1, x1),
            box_width,
            box_height,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

plt.suptitle(
    "Synthetic Manipulation Examples Used for Fake Patch Generation",
    fontsize=16,
    fontweight="bold",
    y=0.96
)

save_path = os.path.join(
    OUTPUT_DIR,
    "figure4_manipulation_examples_final.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✅ Final Figure 4 saved:", save_path)