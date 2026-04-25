import os
import numpy as np
import random
import shutil

real_folder = "data_roi2/patches/real"
fake_folder = "data_roi2/patches/fake"

os.makedirs(fake_folder, exist_ok=True)

# clear old fake patches if rerun
for f in os.listdir(fake_folder):
    os.remove(os.path.join(fake_folder, f))

real_files = [f for f in os.listdir(real_folder) if f.endswith(".npy")]

print("Total real Jaipur patches:", len(real_files))

fake_count = 0

for file in real_files:
    path = os.path.join(real_folder, file)
    patch = np.load(path)

    manipulated = patch.copy()

    manipulation_type = random.choice([
        "noise",
        "brightness",
        "spectral_shift"
    ])

    if manipulation_type == "noise":
        noise = np.random.normal(0, 0.03, manipulated.shape)
        manipulated += noise

    elif manipulation_type == "brightness":
        manipulated[:3] *= random.uniform(1.1, 1.3)

    elif manipulation_type == "spectral_shift":
        manipulated[3] *= random.uniform(0.7, 1.3)   # NDVI
        manipulated[4] *= random.uniform(0.7, 1.3)   # NDWI
        manipulated[5] *= random.uniform(0.7, 1.3)   # NDBI

    manipulated = np.clip(manipulated, -1, 1)

    save_name = file.replace(".npy", "_fake.npy")
    save_path = os.path.join(fake_folder, save_name)

    np.save(save_path, manipulated.astype("float32"))
    fake_count += 1

print("\n✅ Step 11C complete.")
print("Fake Jaipur patches created:", fake_count)
print("Saved in:", fake_folder)