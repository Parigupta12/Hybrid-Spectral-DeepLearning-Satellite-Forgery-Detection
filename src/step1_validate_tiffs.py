import os
import rasterio
import numpy as np

folder_path = "data/raw"

expected_bands = {
    1: "B2 - Blue",
    2: "B3 - Green",
    3: "B4 - Red",
    4: "B8 - NIR",
    5: "B11 - SWIR"
}

print("STEP 1: DATA UNDERSTANDING + VALIDATION")
print("=" * 60)

for file in sorted(os.listdir(folder_path)):
    if file.endswith(".tif") or file.endswith(".tiff"):
        path = os.path.join(folder_path, file)

        print("\n📂 File:", file)

        with rasterio.open(path) as src:
            print("Bands:", src.count)
            print("CRS:", src.crs)
            print("Shape:", (src.height, src.width))
            print("Data type:", src.dtypes)
            print("Bounds:", src.bounds)

            if src.count != 5:
                print("❌ ERROR: This file does not have 5 bands.")
            else:
                print("✅ Band count correct: 5 bands present")

            print("\nBand order check:")
            for i in range(1, src.count + 1):
                band = src.read(i)
                band_min = np.nanmin(band)
                band_max = np.nanmax(band)
                band_mean = np.nanmean(band)

                band_name = expected_bands.get(i, "Unknown Band")

                print(
                    f"Band {i} ({band_name}) → "
                    f"Min: {band_min:.4f}, "
                    f"Max: {band_max:.4f}, "
                    f"Mean: {band_mean:.4f}"
                )

            print("-" * 60)