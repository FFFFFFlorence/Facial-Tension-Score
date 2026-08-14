"""
au_feature_selection.py
--------------------------
Data-driven replacement for the hand-picked "4 AUs + gaze" tension formula.

Runs OpenFace across a folder of emotion-labeled videos (RAVDESS video files,
filename-encoded per the standard RAVDESS convention), extracts ALL Action
Units (not just the 4 originally chosen), and statistically ranks which AUs
actually differ most between "tense-associated" emotions (fearful, angry,
disgust) and a neutral/calm baseline.

IMPORTANT CAVEATS:
- RAVDESS emotions are ACTED by professional actors, not genuine spontaneous
  nervousness. This tells you which AUs move together with *portrayed*
  negative-high-arousal emotion, which is a reasonable proxy but not proof
  that the same AUs will track real interview nervousness.
- This is a first-pass, exploratory analysis (effect sizes / mean
  differences), not a trained classifier. Treat the output as a better-
  informed starting point for TENSION_WEIGHTS, not a validated final answer.

Usage:
    1. Download and extract RAVDESS video files (Video_Speech_Actor_01.zip
       through Video_Speech_Actor_24.zip) from https://zenodo.org/record/1188976
    2. Put all the extracted Actor_XX folders under one parent folder.
    3. Set VIDEO_ROOT below to that parent folder.
    4. Run this script. It will process every video through OpenFace,
       aggregate AU means per file, and print/save a ranked AU comparison.

This can take a while depending on how many videos you point it at - start
with a handful of actors (e.g. Actor_01 through Actor_04) rather than all 24
to get a quick first read before committing to the full dataset.
"""

import os
import subprocess
import glob
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
OPENFACE_EXE = r"C:\OpenFace\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
VIDEO_ROOT = r"C:\path\to\RAVDESS_video\Actor_folders"   # <-- change this: parent folder containing Actor_01, Actor_02, ...
OUTPUT_DIR = r"C:\OpenFace\au_analysis_output"
RESULTS_CSV = r"C:\OpenFace\au_analysis_output\au_feature_ranking.csv"

# RAVDESS emotion code -> label (3rd number in the filename)
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Emotions we're treating as "tense-associated" (high-arousal, negative) vs
# the baseline/comparison group. This grouping choice is itself a judgment
# call worth reconsidering depending on what you actually mean by "tension" -
# e.g. you might argue "surprised" shouldn't count, or that "sad" should.
TENSE_GROUP = {"angry", "fearful", "disgust"}
BASELINE_GROUP = {"neutral", "calm"}

# All AUs OpenFace reports as intensity ("_r") columns
ALL_AU_COLUMNS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
    "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r",
    "AU25_r", "AU26_r", "AU45_r",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_ravdess_filename(filename):
    """Returns (emotion_label, actor_id) or (None, None) if it doesn't match
    the expected RAVDESS naming pattern."""
    parts = filename.replace(".mp4", "").split("-")
    if len(parts) != 7:
        return None, None
    emotion_code = parts[2]
    actor_id = parts[6]
    return EMOTION_MAP.get(emotion_code), actor_id


def find_ravdess_videos(root_dir):
    videos = []
    for path in glob.glob(os.path.join(root_dir, "**", "*.mp4"), recursive=True):
        filename = os.path.basename(path)
        emotion, actor_id = parse_ravdess_filename(filename)
        if emotion is not None:
            videos.append((path, emotion, actor_id))
    return videos


def run_openface_on_video(video_path, session_name):
    session_dir = os.path.join(OUTPUT_DIR, session_name)
    os.makedirs(session_dir, exist_ok=True)
    csv_path = os.path.join(session_dir, "session.csv")

    if os.path.exists(csv_path):
        return csv_path  # already processed, skip re-running OpenFace

    subprocess.run([
        OPENFACE_EXE, "-f", video_path,
        "-out_dir", session_dir,
        "-of", "session",
    ], capture_output=True, text=True)

    return csv_path if os.path.exists(csv_path) else None


def extract_mean_aus(csv_path):
    """Returns a dict of {AU_column: mean_intensity} for one processed video,
    using only frames OpenFace tracked with reasonable confidence."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"  Could not read {csv_path}: {e}")
        return None

    for col in ALL_AU_COLUMNS + ["confidence", "success"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "success" in df.columns:
        df = df[df["success"] == 1]
    if "confidence" in df.columns:
        df = df[df["confidence"] >= 0.75]

    if len(df) == 0:
        return None

    return {au: df[au].mean() for au in ALL_AU_COLUMNS if au in df.columns}


def main():
    videos = find_ravdess_videos(VIDEO_ROOT)
    print(f"Found {len(videos)} RAVDESS video files with recognizable emotion labels.")
    if not videos:
        print(f"No videos found under {VIDEO_ROOT} - check the path and that files end in .mp4")
        return

    rows = []
    for i, (video_path, emotion, actor_id) in enumerate(videos):
        session_name = f"{actor_id}_{emotion}_{os.path.basename(video_path).replace('.mp4', '')}"
        print(f"[{i+1}/{len(videos)}] Processing {os.path.basename(video_path)} ({emotion})...")

        csv_path = run_openface_on_video(video_path, session_name)
        if not csv_path:
            print("  OpenFace failed to produce output, skipping.")
            continue

        au_means = extract_mean_aus(csv_path)
        if au_means is None:
            print("  No reliable frames, skipping.")
            continue

        au_means["emotion"] = emotion
        au_means["actor_id"] = actor_id
        au_means["file"] = os.path.basename(video_path)
        rows.append(au_means)

    if not rows:
        print("No usable results.")
        return

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "per_video_au_means.csv"), index=False)
    print(f"\nPer-video AU means saved to {os.path.join(OUTPUT_DIR, 'per_video_au_means.csv')}")

    # ---- Statistical comparison: tense-associated emotions vs baseline ----
    tense_rows = results_df[results_df["emotion"].isin(TENSE_GROUP)]
    baseline_rows = results_df[results_df["emotion"].isin(BASELINE_GROUP)]

    print(f"\nTense-associated group ({', '.join(TENSE_GROUP)}): {len(tense_rows)} videos")
    print(f"Baseline group ({', '.join(BASELINE_GROUP)}): {len(baseline_rows)} videos")

    ranking = []
    for au in ALL_AU_COLUMNS:
        if au not in results_df.columns:
            continue
        tense_vals = tense_rows[au].dropna()
        baseline_vals = baseline_rows[au].dropna()
        if len(tense_vals) < 3 or len(baseline_vals) < 3:
            continue

        tense_mean, baseline_mean = tense_vals.mean(), baseline_vals.mean()
        pooled_std = np.sqrt((tense_vals.std() ** 2 + baseline_vals.std() ** 2) / 2) or 1e-6
        cohens_d = (tense_mean - baseline_mean) / pooled_std  # effect size: how separated the two groups are

        ranking.append({
            "AU": au,
            "tense_mean": round(tense_mean, 3),
            "baseline_mean": round(baseline_mean, 3),
            "difference": round(tense_mean - baseline_mean, 3),
            "cohens_d": round(cohens_d, 3),
            "abs_cohens_d": round(abs(cohens_d), 3),
        })

    ranking_df = pd.DataFrame(ranking).sort_values("abs_cohens_d", ascending=False)
    ranking_df.to_csv(RESULTS_CSV, index=False)

    print(f"\n{'='*70}")
    print("AU RANKING - which Action Units best separate tense-associated")
    print("emotions (angry/fearful/disgust) from neutral/calm, by effect size")
    print(f"{'='*70}")
    print(ranking_df.to_string(index=False))
    print(f"\nFull results saved to {RESULTS_CSV}")

    # ---- Suggested new weights, normalized from the top discriminative AUs ----
    top_n = 6
    top_aus = ranking_df.head(top_n).copy()
    top_aus = top_aus[top_aus["cohens_d"] > 0]  # only keep AUs that go UP with tension, not down
    if len(top_aus) > 0:
        total = top_aus["abs_cohens_d"].sum()
        top_aus["suggested_weight"] = (top_aus["abs_cohens_d"] / total).round(3)

        print(f"\n{'='*70}")
        print(f"SUGGESTED replacement for TENSION_WEIGHTS (top {len(top_aus)} AUs")
        print("that most increase with tense-associated emotions):")
        print(f"{'='*70}")
        for _, row in top_aus.iterrows():
            print(f'    "{row["AU"]}": {row["suggested_weight"]},')
        print("\nCompare this against the original hand-picked set:")
        print('    "AU04_r": 0.3, "AU07_r": 0.2, "AU20_r": 0.3, "AU23_r": 0.2')


if __name__ == "__main__":
    main()
