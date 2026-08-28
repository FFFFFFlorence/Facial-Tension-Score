"""
face_recognition_engine.py
------------------------------
Standalone face recognition prototype - detection, embedding extraction,
enrollment, and matching. Deliberately kept separate from the tension-
monitoring codebase, as requested.

IMPORTANT SCOPE NOTE: this is a prototype/test harness for evaluating
recognition ACCURACY against public data (e.g. LFW). It is NOT connected
to any government or immigration database. Before pointing this at any
real system, the legal authorization / compliance questions raised
earlier in this project (PDP Law, scope of use, etc.) need to be resolved
- this script does not make that determination for you.

-------------------------------------------------------------------------
ARCHITECTURE - designed so YOU can swap in a real database later
-------------------------------------------------------------------------
All "where are known faces stored, and how do we look one up" logic goes
through the FaceDatabaseBackend interface below. This prototype ships with
one concrete implementation - LocalPickleDatabase - which just stores
embeddings in a local pickle file. It exists purely so you can test the
recognition ALGORITHM without needing real database access yet.

To connect this to your actual database once you have access, you write a
NEW class that inherits from FaceDatabaseBackend and implements its three
methods (enroll, find_match, all_identities) using your real database's
API/SQL/etc. instead of the pickle file. A stub template for this,
RealDatabaseBackend, is included below with TODOs marking exactly what to
fill in. Nothing else in this file needs to change - recognize_image() and
the evaluation harness work against the abstract interface, not against
LocalPickleDatabase specifically.
-------------------------------------------------------------------------
"""

import os
import glob
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import face_recognition  # pip install face_recognition (wraps dlib)

# ---------------------------------------------------------------------------
KNOWN_FACES_DIR = r"C:\FaceRecognition\known_faces"
TEST_FACES_DIR = r"C:\FaceRecognition\test_faces"
DATABASE_PICKLE_PATH = r"C:\FaceRecognition\face_gallery.pkl"

# Euclidean distance threshold below which two face embeddings are
# considered a match. Lower = stricter (fewer false matches, more missed
# matches). face_recognition's own docs suggest ~0.6 as a reasonable
# starting point for their embedding model - not independently validated
# here, same "reasonable default, not a proven threshold" caveat that's
# applied to every other heuristic threshold in this project.
MATCH_DISTANCE_THRESHOLD = 0.6


# ===========================================================================
# ABSTRACT DATABASE INTERFACE - implement this against your real database
# ===========================================================================
class FaceDatabaseBackend(ABC):
    """
    Anything that can store a face embedding under an identity, and later
    be searched for the closest match to a new embedding. Implement this
    interface against your real database when you have access - the rest
    of this file (recognize_image, evaluation harness) only depends on
    these three methods, never on how they're implemented underneath.
    """

    @abstractmethod
    def enroll(self, identity_id, embedding, metadata=None):
        """Store one face embedding under an identity. `identity_id` should
        be whatever your real system uses to key a person (e.g. a national
        ID number, case number, or database primary key) - NOT necessarily
        a human name. `metadata` is an optional dict for anything else you
        want retrievable alongside a match (e.g. full record, case notes)."""
        raise NotImplementedError

    @abstractmethod
    def find_match(self, embedding, threshold=MATCH_DISTANCE_THRESHOLD):
        """Given a query embedding, return (identity_id, distance, metadata)
        for the closest enrolled face within `threshold`, or None if
        nothing is close enough to count as a match."""
        raise NotImplementedError

    @abstractmethod
    def all_identities(self):
        """Return a list of all enrolled identity_ids - mostly useful for
        debugging/evaluation, not required for live recognition itself."""
        raise NotImplementedError


class LocalPickleDatabase(FaceDatabaseBackend):
    """
    PROTOTYPE backend - stores embeddings in a local pickle file. This is
    what lets you test the recognition algorithm's accuracy right now,
    without needing real database access. Not intended for any real
    deployment - no encryption, no access control, no audit logging, all
    of which a real system handling identity data would need.
    """

    def __init__(self, pickle_path=DATABASE_PICKLE_PATH):
        self.pickle_path = pickle_path
        self.records = []  # list of dicts: {identity_id, embedding, metadata}
        self._load()

    def _load(self):
        if os.path.exists(self.pickle_path):
            with open(self.pickle_path, "rb") as f:
                self.records = pickle.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.pickle_path), exist_ok=True)
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.records, f)

    def enroll(self, identity_id, embedding, metadata=None):
        self.records.append({
            "identity_id": identity_id,
            "embedding": embedding,
            "metadata": metadata or {},
            "enrolled_at": datetime.now().isoformat(),
        })
        self._save()

    def find_match(self, embedding, threshold=MATCH_DISTANCE_THRESHOLD):
        if not self.records:
            return None

        distances = [
            np.linalg.norm(embedding - rec["embedding"])
            for rec in self.records
        ]
        best_idx = int(np.argmin(distances))
        best_distance = distances[best_idx]

        if best_distance > threshold:
            return None

        best = self.records[best_idx]
        return best["identity_id"], best_distance, best["metadata"]

    def all_identities(self):
        return sorted(set(rec["identity_id"] for rec in self.records))


# ===========================================================================
# TEMPLATE STUB - fill this in yourself once you have real database access.
# Nothing in this file calls this class - it's here as a starting point.
# ===========================================================================
class RealDatabaseBackend(FaceDatabaseBackend):
    """
    TEMPLATE ONLY - not implemented, not used anywhere in this file.

    Replace the method bodies below with real calls to your actual
    database/API. The rest of this codebase (recognize_image, evaluation
    harness) will work unchanged once you do, since they only call the
    three abstract methods defined above.

    Things you'll likely need to figure out on your end:
    - How embeddings get stored (a BLOB/vector column? a separate vector
      search service? exported to a file the DB references?)
    - Whether matching happens in Python (like LocalPickleDatabase does -
      pull all embeddings, compare in numpy) or is pushed down into the
      database/search engine itself (faster at scale, but needs the DB
      to support vector similarity search)
    - Authentication/connection details for your database
    - Audit logging - a real system doing identity lookups should almost
      certainly log who queried what and when
    """

    def __init__(self, connection_config):
        # TODO: open your real database connection here
        # self.conn = your_db_library.connect(**connection_config)
        raise NotImplementedError("Fill in your real database connection here.")

    def enroll(self, identity_id, embedding, metadata=None):
        # TODO: INSERT the embedding + identity_id + metadata into your DB
        raise NotImplementedError

    def find_match(self, embedding, threshold=MATCH_DISTANCE_THRESHOLD):
        # TODO: query your DB for the closest embedding, e.g.:
        #   - pull candidate embeddings and compare in Python (simple, slow at scale), or
        #   - use a vector similarity search feature if your DB/search engine has one
        raise NotImplementedError

    def all_identities(self):
        # TODO: return all known identity_ids from your DB
        raise NotImplementedError


# ===========================================================================
# CORE RECOGNITION LOGIC - works against ANY FaceDatabaseBackend
# ===========================================================================
def get_face_embedding(image_path):
    """
    Detects the largest/first face in an image and returns its 128-d
    embedding, or None if no face was detected.
    """
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)
    if not locations:
        return None
    encodings = face_recognition.face_encodings(image, known_face_locations=locations)
    return encodings[0] if encodings else None


def enroll_gallery_from_folder(folder_path, db):
    """
    Walks a folder structured as folder_path/<identity_name>/*.jpg and
    enrolls every detected face into the given database backend. In a real
    deployment, identity_name would be replaced by whatever ID your
    database actually uses to key a person.
    """
    enrolled_count = 0
    skipped_count = 0

    for identity_dir in sorted(glob.glob(os.path.join(folder_path, "*"))):
        if not os.path.isdir(identity_dir):
            continue
        identity_id = os.path.basename(identity_dir)

        for image_path in glob.glob(os.path.join(identity_dir, "*.jpg")) + \
                           glob.glob(os.path.join(identity_dir, "*.png")):
            embedding = get_face_embedding(image_path)
            if embedding is None:
                print(f"  No face detected in {image_path}, skipping.")
                skipped_count += 1
                continue
            db.enroll(identity_id, embedding, metadata={"source_image": image_path})
            enrolled_count += 1

    print(f"Enrollment complete: {enrolled_count} face(s) enrolled, {skipped_count} skipped (no face detected).")


def recognize_image(image_path, db, threshold=MATCH_DISTANCE_THRESHOLD):
    """
    Detects a face in image_path and looks it up against the database.
    Returns (identity_id, distance, metadata) on a match, or None.
    """
    embedding = get_face_embedding(image_path)
    if embedding is None:
        print(f"No face detected in {image_path}.")
        return None
    return db.find_match(embedding, threshold=threshold)


# ===========================================================================
# EVALUATION HARNESS - measures accuracy against the held-out LFW test set
# built by lfw_gallery_builder.py
# ===========================================================================
def evaluate_against_test_set(db, test_folder=TEST_FACES_DIR, threshold=MATCH_DISTANCE_THRESHOLD):
    """
    For every held-out test image, checks whether the database correctly
    identifies which person it is - the same kind of held-out evaluation
    approach used for the tension model's cross-validation, applied here
    to recognition accuracy instead.
    """
    total = 0
    correct = 0
    false_matches = 0
    missed_matches = 0
    no_face_detected = 0

    results_log = []

    for identity_dir in sorted(glob.glob(os.path.join(test_folder, "*"))):
        if not os.path.isdir(identity_dir):
            continue
        true_identity = os.path.basename(identity_dir)

        for image_path in glob.glob(os.path.join(identity_dir, "*.jpg")) + \
                           glob.glob(os.path.join(identity_dir, "*.png")):
            total += 1
            embedding = get_face_embedding(image_path)
            if embedding is None:
                no_face_detected += 1
                continue

            match = db.find_match(embedding, threshold=threshold)

            if match is None:
                missed_matches += 1
                predicted = "NO MATCH"
                is_correct = False
            else:
                predicted_identity, distance, _ = match
                predicted = f"{predicted_identity} (dist={distance:.3f})"
                is_correct = predicted_identity == true_identity
                if is_correct:
                    correct += 1
                else:
                    false_matches += 1

            results_log.append({
                "image": image_path,
                "true_identity": true_identity,
                "predicted": predicted,
                "correct": is_correct,
            })

    print(f"\n{'='*60}\nEVALUATION RESULTS (threshold={threshold})\n{'='*60}")
    print(f"Total test images: {total}")
    print(f"No face detected:  {no_face_detected}")
    print(f"Correct matches:   {correct}")
    print(f"False matches (wrong identity): {false_matches}")
    print(f"Missed matches (should've matched, didn't): {missed_matches}")
    if total - no_face_detected > 0:
        accuracy = correct / (total - no_face_detected)
        print(f"\nAccuracy (of images with a detected face): {accuracy:.1%}")

    print(f"\n{'-'*60}\nPer-image results:\n{'-'*60}")
    for r in results_log:
        status = "OK " if r["correct"] else "ERR"
        print(f"[{status}] {os.path.basename(r['image']):20s} true={r['true_identity']:15s} predicted={r['predicted']}")

    return results_log


# ===========================================================================
if __name__ == "__main__":
    print("Initializing local prototype database (swap for RealDatabaseBackend later)...")
    db = LocalPickleDatabase()

    if not db.all_identities():
        print(f"\nEnrolling gallery from {KNOWN_FACES_DIR} ...")
        enroll_gallery_from_folder(KNOWN_FACES_DIR, db)
    else:
        print(f"Database already has {len(db.all_identities())} enrolled identities - skipping re-enrollment.")
        print("(Delete face_gallery.pkl to start fresh.)")

    print(f"\nEnrolled identities: {db.all_identities()}")

    evaluate_against_test_set(db, TEST_FACES_DIR)
