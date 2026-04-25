import os
import rasterio
import numpy as np

input_folder = "data_roi2/processed"
output_folder = "data_roi2/patches/real"

os.makedirs(output_folder, exist_ok=True)

# Clear old real patches
for f in os.listdir(output_folder):
    if f.endswith(".npy"):
        os.remove(os.path.join(output_folder, f))

PATCH_SIZE = 64
STRIDE = 32

patch_count = 0

for file in os.listdir(input_folder):
    if file.endswith(".tif") or file.endswith(".tiff"):
        path = os.path.join(input_folder, file)

        with rasterio.open(path) as src:
            img = src.read().astype("float32")

            print("\n📂 Processing:", file)
            print("Image shape:", img.shape)

            bands, height, width = img.shape

            if bands != 6:
                raise ValueError("❌ Processed ROI-2 image must have 6 channels")

            for y in range(0, height - PATCH_SIZE + 1, STRIDE):
                for x in range(0, width - PATCH_SIZE + 1, STRIDE):
                    patch = img[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE]

                    if patch.shape != (6, PATCH_SIZE, PATCH_SIZE):
                        continue

                    if np.isnan(patch).any():
                        continue

                    if np.mean(patch[:3]) < 0.01:
                        continue

                    save_name = f"jaipur_roi2_patch_{patch_count}.npy"
                    save_path = os.path.join(output_folder, save_name)

                    np.save(save_path, patch.astype("float32"))
                    patch_count += 1

print("\n✅ Revised Step 11B complete.")
print("Patch size:", PATCH_SIZE)
print("Stride:", STRIDE)
print("Total Jaipur ROI-2 real patches created:", patch_count)
print("Saved in:", output_folder)