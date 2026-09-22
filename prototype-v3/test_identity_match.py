"""
test_identity_match.py
-----------------------
Batch test harness for the ArcFace lookup, built around LFW-style data:

KNOWN_FACES_DIR - the "database" - one subfolder per person, each
    holding that person's reference photo(s).
TEST_FACES_DIR  - a held-out set in the SAME one-subfolder-per-person
    layout, used as queries: for each photo in here, we check whether
    the pipeline matches it back to the correct person in
    KNOWN_FACES_DIR.

The "expected" identity for a test photo comes from its own parent
folder name (e.g. test_faces/Colin_Powell/xyz.jpg -> "Colin_Powell").
The "predicted" identity comes from the parent folder of whichever
known_faces photo it matched to. Comparing the two, across every test
photo, gives an accuracy number instead of one-off spot checks.

Usage:
    python test_identity_match.py
        Batch mode - runs every photo under TEST_FACES_DIR and prints a
        per-photo verdict plus a summary (correct / wrong / no-match).

    python test_identity_match.py path\\to\\one_photo.jpg
        Single-photo mode - runs just that one photo against
        KNOWN_FACES_DIR and prints the result. Useful for digging into
        one case the batch run flagged as wrong.
"""
import os
import sys
from pathlib import Path

# Must match FACE_DATABASE_DIR in prototype_facial_recognition.py / enroll_test_subject.py
KNOWN_FACES_DIR = r"C:\FaceRecognition\known_faces"
TEST_FACES_DIR = r"C:\FaceRecognition\test_faces"

ARCFACE_MODEL_NAME = "ArcFace"
ARCFACE_DETECTOR_BACKEND = "retinaface"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def person_name_from_path(photo_path, root_dir):
    """LFW-style layout: root_dir/Person_Name/some_photo.jpg -> 'Person_Name'."""
    try:
        rel = os.path.relpath(photo_path, root_dir)
        first_part = rel.split(os.sep)[0]
        if first_part != os.pardir and not os.path.isabs(first_part):
            return first_part
    except ValueError:
        pass
    return os.path.basename(os.path.dirname(photo_path))  # fallback: photo not actually under root_dir


def find_match(photo_path, db_path=KNOWN_FACES_DIR):
    """Runs one lookup. Returns (matched_photo_path, distance) or (None, distance-of-closest-miss)."""
    from deepface import DeepFace  # lazy import - keeps startup fast, esp. for --help-style single calls

    try:
        results = DeepFace.find(
            img_path=photo_path,
            db_path=db_path,
            model_name=ARCFACE_MODEL_NAME,
            detector_backend=ARCFACE_DETECTOR_BACKEND,
            enforce_detection=False,
            silent=True,
        )
    except Exception as e:
        print(f"    Lookup failed on {photo_path}: {e}")
        return None, None

    if not results or len(results[0]) == 0:
        return None, None  # nothing in the database looked close at all

    best = results[0].iloc[0]
    if best["distance"] > best["threshold"]:
        return None, best["distance"]  # closest candidate still isn't within the match threshold

    return best["identity"], best["distance"]


def list_images(root_dir):
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


def run_batch():
    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"Known-faces folder not found: {KNOWN_FACES_DIR}")
        return
    if not os.path.isdir(TEST_FACES_DIR):
        print(f"Test-faces folder not found: {TEST_FACES_DIR}")
        return

    test_images = list_images(TEST_FACES_DIR)
    if not test_images:
        print(f"No image files found under {TEST_FACES_DIR}")
        return

    print(f"Found {len(test_images)} test photo(s) across "
          f"{len(set(person_name_from_path(p, TEST_FACES_DIR) for p in test_images))} people.")
    print(f"Running lookups against {KNOWN_FACES_DIR} ...\n")

    correct, wrong, no_match = 0, 0, 0

    for photo_path in test_images:
        expected = person_name_from_path(photo_path, TEST_FACES_DIR)
        matched_path, distance = find_match(photo_path)
        rel = os.path.relpath(photo_path, TEST_FACES_DIR)

        if matched_path is None:
            no_match += 1
            dist_note = f" (closest dist={distance:.4f})" if distance is not None else ""
            verdict = f"NO MATCH{dist_note}"
        else:
            predicted = person_name_from_path(matched_path, KNOWN_FACES_DIR)
            if predicted == expected:
                correct += 1
                verdict = f"CORRECT  dist={distance:.4f}"
            else:
                wrong += 1
                verdict = f"WRONG -> predicted '{predicted}'  dist={distance:.4f}"

        print(f"[{expected:20s}] {rel:45s} {verdict}")

    total = len(test_images)
    print("\n--- Summary ---")
    print(f"Total test photos : {total}")
    print(f"Correct           : {correct} ({100 * correct / total:.1f}%)")
    print(f"Wrong person      : {wrong} ({100 * wrong / total:.1f}%)")
    print(f"No match at all   : {no_match} ({100 * no_match / total:.1f}%)")


def run_single(photo_path):
    if not os.path.exists(photo_path):
        print(f"Photo not found: {photo_path}")
        return
    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"Known-faces folder not found: {KNOWN_FACES_DIR}")
        return

    matched_path, distance = find_match(photo_path)
    if matched_path is None:
        dist_note = f" (closest dist={distance:.4f})" if distance is not None else ""
        print(f"Result: NO MATCH{dist_note}")
    else:
        predicted = person_name_from_path(matched_path, KNOWN_FACES_DIR)
        print(f"Result: MATCH -> {predicted}")
        print(f"  matched photo : {matched_path}")
        print(f"  distance      : {distance:.4f}")


def main():
    if len(sys.argv) > 1:
        run_single(sys.argv[1])
    else:
        run_batch()


if __name__ == "__main__":
    main()
