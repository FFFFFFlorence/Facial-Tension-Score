"""
lfw_gallery_builder.py
--------------------------
Auto-samples identities from LFW (Labeled Faces in the Wild) to build a
test gallery, using scikit-learn's built-in LFW downloader/loader (no
manual dataset download needed - it fetches and caches automatically on
first run, given internet access on your machine).

Produces two folders:
  known_faces/<person_name>/*.jpg   - "enrolled" identities (gallery)
  test_faces/<person_name>/*.jpg    - held-out photos of the SAME people,
                                       used to measure recognition accuracy
                                       (never shown during enrollment)

Only people with enough photos in LFW are selected (min_faces_per_person),
so each identity has both gallery and held-out test images available.

Usage:
    python lfw_gallery_builder.py
"""

import os
import numpy as np
from sklearn.datasets import fetch_lfw_people
from PIL import Image

# ---------------------------------------------------------------------------
OUTPUT_ROOT = r"C:\FaceRecognition"
KNOWN_FACES_DIR = os.path.join(OUTPUT_ROOT, "known_faces")
TEST_FACES_DIR = os.path.join(OUTPUT_ROOT, "test_faces")

MIN_FACES_PER_PERSON = 20   # only pick identities with at least this many LFW photos
N_IDENTITIES = 10           # how many different people to sample for the test gallery
N_GALLERY_IMAGES_PER_PERSON = 3   # how many photos go into the "known" gallery per person
N_TEST_IMAGES_PER_PERSON = 3      # how many DIFFERENT held-out photos go into the test set


def main():
    print("Downloading/loading LFW dataset via scikit-learn (cached after first run)...")
    lfw = fetch_lfw_people(min_faces_per_person=MIN_FACES_PER_PERSON, color=True, resize=1.0)

    images = lfw.images          # shape: (n_samples, h, w, 3), float in [0, 255]... actually [0,1] scaled sometimes
    target = lfw.target          # integer label per image
    target_names = lfw.target_names  # label index -> person name

    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(TEST_FACES_DIR, exist_ok=True)

    unique_labels = np.unique(target)
    rng = np.random.default_rng(seed=42)
    chosen_labels = rng.choice(unique_labels, size=min(N_IDENTITIES, len(unique_labels)), replace=False)

    print(f"Selected {len(chosen_labels)} identities out of {len(unique_labels)} eligible "
          f"(each with >= {MIN_FACES_PER_PERSON} photos in LFW).")

    for label in chosen_labels:
        person_name = target_names[label].replace(" ", "_")
        person_indices = np.where(target == label)[0]
        rng.shuffle(person_indices)

        needed = N_GALLERY_IMAGES_PER_PERSON + N_TEST_IMAGES_PER_PERSON
        if len(person_indices) < needed:
            print(f"  Skipping {person_name}: only {len(person_indices)} photos, need {needed}.")
            continue

        gallery_idx = person_indices[:N_GALLERY_IMAGES_PER_PERSON]
        test_idx = person_indices[N_GALLERY_IMAGES_PER_PERSON:needed]

        person_known_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        person_test_dir = os.path.join(TEST_FACES_DIR, person_name)
        os.makedirs(person_known_dir, exist_ok=True)
        os.makedirs(person_test_dir, exist_ok=True)

        for i, idx in enumerate(gallery_idx):
            _save_image(images[idx], os.path.join(person_known_dir, f"gallery_{i}.jpg"))
        for i, idx in enumerate(test_idx):
            _save_image(images[idx], os.path.join(person_test_dir, f"test_{i}.jpg"))

        print(f"  {person_name}: {len(gallery_idx)} gallery photos, {len(test_idx)} held-out test photos")

    print(f"\nDone. Gallery images: {KNOWN_FACES_DIR}")
    print(f"Held-out test images: {TEST_FACES_DIR}")
    print("\nNext step: run face_recognition_engine.py to enroll the gallery and test recognition accuracy.")


def _save_image(img_array, path):
    # LFW images from sklearn come back as float arrays (0-255 range already for color=True)
    arr = np.clip(img_array, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


if __name__ == "__main__":
    main()
