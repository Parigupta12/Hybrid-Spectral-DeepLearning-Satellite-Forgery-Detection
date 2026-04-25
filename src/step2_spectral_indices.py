import rasterio
import numpy as np
import os

input_folder = "data/raw"
output_folder = "data/processed"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".tif") or file.endswith(".tiff"):
        path = os.path.join(input_folder, file)

        with rasterio.open(path) as src:
            print("\n📂 Processing:", file)

            B2 = src.read(1).astype("float32")   # Blue
            B3 = src.read(2).astype("float32")   # Green
            B4 = src.read(3).astype("float32")   # Red
            B8 = src.read(4).astype("float32")   # NIR
            B11 = src.read(5).astype("float32")  # SWIR

            eps = 1e-10

            NDVI = (B8 - B4) / (B8 + B4 + eps)
            NDWI = (B3 - B8) / (B3 + B8 + eps)
            NDBI = (B11 - B8) / (B11 + B8 + eps)

            stacked = np.stack([B4, B3, B2, NDVI, NDWI, NDBI], axis=0)

            print("✅ 6-channel shape:", stacked.shape)
            print("NDVI range:", NDVI.min(), "to", NDVI.max())
            print("NDWI range:", NDWI.min(), "to", NDWI.max())
            print("NDBI range:", NDBI.min(), "to", NDBI.max())

            profile = src.profile.copy()
            profile.update(count=6, dtype="float32")

            output_path = os.path.join(
                output_folder,
                file.replace(".tif", "_processed.tif").replace(".tiff", "_processed.tif")
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(stacked.astype("float32"))

            print("💾 Saved:", output_path)

print("\n✅ Step 2 complete: all processed files saved.")