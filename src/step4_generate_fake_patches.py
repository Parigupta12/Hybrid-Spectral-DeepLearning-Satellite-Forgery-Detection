import os
import random
import csv
import numpy as np

real_folder = "data/patches/real"
fake_folder = "data/patches/fake"
log_path = "data/manipulation_log.csv"

os.makedirs(fake_folder, exist_ok=True)

random.seed(42)
np.random.seed(42)

PATCH_SIZE = 64


def copy_paste_manipulation(patch):
    fake = patch.copy()

    size = random.choice([24, 28, 32])

    x1 = random.randint(0, PATCH_SIZE - size)
    y1 = random.randint(0, PATCH_SIZE - size)
    x2 = random.randint(0, PATCH_SIZE - size)
    y2 = random.randint(0, PATCH_SIZE - size)

    source = fake[:, y1:y1+size, x1:x1+size].copy()
    fake[:, y2:y2+size, x2:x2+size] = source

    return fake, "context_aware_copy_paste"


def spectral_inconsistency_manipulation(patch):
    fake = patch.copy()

    size = random.choice([24, 28, 32])
    x = random.randint(0, PATCH_SIZE - size)
    y = random.randint(0, PATCH_SIZE - size)

    # Keep RGB unchanged, strongly disturb spectral indices
    fake[3, y:y+size, x:x+size] = random.uniform(-0.4, 0.1)   # NDVI abnormal
    fake[4, y:y+size, x:x+size] = random.uniform(0.3, 0.8)    # NDWI abnormal
    fake[5, y:y+size, x:x+size] = random.uniform(0.2, 0.7)    # NDBI abnormal

    fake[3:] = np.clip(fake[3:], -1, 1)

    return fake, "spectral_inconsistency"


def radiometric_distortion(patch):
    fake = patch.copy()

    size = random.choice([24, 28, 32])
    x = random.randint(0, PATCH_SIZE - size)
    y = random.randint(0, PATCH_SIZE - size)

    factor = random.choice([0.55, 0.65, 1.35, 1.45])
    fake[:3, y:y+size, x:x+size] *= factor
    fake[:3] = np.clip(fake[:3], 0, 1)

    return fake, "radiometric_distortion"


def noise_injection(patch):
    fake = patch.copy()

    size = random.choice([24, 28, 32])
    x = random.randint(0, PATCH_SIZE - size)
    y = random.randint(0, PATCH_SIZE - size)

    noise = np.random.normal(0, 0.06, fake[:3, y:y+size, x:x+size].shape)
    fake[:3, y:y+size, x:x+size] += noise
    fake[:3] = np.clip(fake[:3], 0, 1)

    return fake, "noise_injection"


manipulation_functions = [
    copy_paste_manipulation,
    spectral_inconsistency_manipulation,
    radiometric_distortion,
    noise_injection
]

real_files = sorted([f for f in os.listdir(real_folder) if f.endswith(".npy")])

fake_count = 0
log_rows = []

for file in real_files:
    path = os.path.join(real_folder, file)
    patch = np.load(path).astype("float32")

    manipulation = random.choice(manipulation_functions)
    fake_patch, manipulation_type = manipulation(patch)

    save_name = file.replace(".npy", "_fake.npy")
    save_path = os.path.join(fake_folder, save_name)

    np.save(save_path, fake_patch.astype("float32"))

    log_rows.append([file, save_name, manipulation_type])
    fake_count += 1

with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["real_patch", "fake_patch", "manipulation_type"])
    writer.writerows(log_rows)

print("✅ Revised Step 4 complete.")
print("Real patches used:", len(real_files))
print("Fake patches created:", fake_count)
print("Saved in:", fake_folder)
print("Manipulation log saved at:", log_path)

print("\nManipulation distribution:")
types = [row[2] for row in log_rows]
for t in sorted(set(types)):
    print(t, ":", types.count(t))