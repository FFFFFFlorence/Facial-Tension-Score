"""
enroll_test_subject.py
-----------------------
Quick utility to build TEST entries for the facial-recognition database
used by find_target_in_database() in the main tension-monitor script.
Run this separately - once per person you want in the test database
(yourself, a consenting colleague). NOT part of the main app, and not
meant to be how the real production database gets built - it exists so
you have real, working test data to validate the ArcFace lookup against,
without anyone needing to hand over real strangers' photos/info.

Usage:
    python enroll_test_subject.py

Controls (in the webcam window):
    SPACE - capture the current frame
    ESC   - cancel without saving
"""
import os
import json
import cv2

# Must match FACE_DATABASE_DIR in the main script
FACE_DATABASE_DIR = r"C:\FaceDatabase\known_faces"


def main():
    os.makedirs(FACE_DATABASE_DIR, exist_ok=True)

    print("=== Enroll a test subject into the face database ===")
    print("(Use a real, consenting person - yourself or a colleague. This")
    print(" is test data for validating the recognition pipeline, not a")
    print(" real production enrollment flow.)\n")

    nama = input("Nama: ").strip()
    if not nama:
        print("Nama tidak boleh kosong. Dibatalkan.")
        return

    fields = {"Nama": nama}
    for label in ["NIK", "Tempat/Tgl Lahir", "Kewarganegaraan", "Alamat"]:
        value = input(f"{label} (kosongkan jika tidak ada): ").strip()
        if value:
            fields[label] = value

    print("\nBisa tambah field lain di luar daftar di atas (contoh: 'Nomor Paspor').")
    while True:
        extra_key = input("Nama field tambahan (kosongkan untuk selesai): ").strip()
        if not extra_key:
            break
        extra_value = input(f"  Nilai untuk '{extra_key}': ").strip()
        fields[extra_key] = extra_value

    safe_name = "".join(c for c in nama if c.isalnum() or c in (" ", "_")).strip().replace(" ", "_")
    if not safe_name:
        safe_name = "subject"
    photo_path = os.path.join(FACE_DATABASE_DIR, f"{safe_name}.jpg")
    info_path = os.path.join(FACE_DATABASE_DIR, f"{safe_name}.json")

    if os.path.exists(photo_path):
        overwrite = input(f"\n'{safe_name}' sudah ada di database. Timpa? (y/n): ").strip().lower()
        if overwrite != "y":
            print("Dibatalkan.")
            return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak bisa membuka webcam.")
        return

    print("\nArahkan wajah ke kamera.")
    print("Tekan SPACE untuk mengambil foto, ESC untuk batal.\n")

    captured = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca dari webcam.")
                break
            cv2.imshow("Enroll - SPACE untuk ambil foto, ESC untuk batal", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                captured = frame
                break
            elif key == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured is None:
        print("Dibatalkan - tidak ada foto diambil.")
        return

    cv2.imwrite(photo_path, captured)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(fields, f, ensure_ascii=False, indent=2)

    print(f"\nTersimpan:")
    print(f"  Foto : {photo_path}")
    print(f"  Info : {info_path}")
    print(f"\nField yang disimpan: {list(fields.keys())}")


if __name__ == "__main__":
    main()
