import rasterio
import numpy as np
import os

input_folder = "data/processed"
output_folder = "data/patches/real"

os.makedirs(output_folder, exist_ok=True)

PATCH_SIZE = 64
patch_count = 0

for file in os.listdir(input_folder):
    if file.endswith(".tif") or file.endswith(".tiff"):
        path = os.path.join(input_folder, file)

        with rasterio.open(path) as src:
            img = src.read().astype("float32")  # shape: (6, H, W)
            bands, height, width = img.shape

            print(f"\n📂 Processing: {file}")
            print("Image shape:", img.shape)

            for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
                for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):
                    patch = img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                    # Skip invalid patches
                    if np.isnan(patch).any():
                        continue

                    # Skip empty/near-empty patches
                    if np.mean(patch[:3]) < 0.01:
                        continue

                    save_name = file.replace("_processed.tif", f"_patch_{patch_count}.npy")
                    save_path = os.path.join(output_folder, save_name)

                    np.save(save_path, patch)
                    patch_count += 1

            print("✅ Patches created so far:", patch_count)

print("\n✅ Step 3 complete.")
print("Total real patches created:", patch_count)