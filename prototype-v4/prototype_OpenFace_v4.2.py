import subprocess
import textwrap
import os
import csv
import shutil
import glob
import re
import cv2
import threading
import pandas as pd
import numpy as np
import time
from io import StringIO
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, RadioButtons, CheckButtons

from voice_analyzer import VoiceAnalyzer  # separate file, as requested
from transcriber import Transcriber, TRANSCRIBER_BACKEND, group_episode_contexts       # separate file, local/offline speech-to-text
# (No DeepFace import here - this variant has no identity/facial-recognition
# lookup at all; see prototype_OpenFace_v3_3_transcript.py for that version.)

# ---------------------------------------------------------------------------
# Original config - unchanged
# ---------------------------------------------------------------------------
# CSV_PATH and LOG_PATH used to be fixed, flat paths shared across every
# session (overwritten/stamped-by-suffix each run). They're now computed
# PER SESSION instead, under a dedicated session_<ID>/ folder - see
# session_dir, current_csv_path, current_log_path, current_snapshot_dir
# (all set in start_openface(), near OPENFACE_OUT_DIR below) and
# GUIDE_PATH (the one exception: a single shared file, not per-session).

TENSION_WEIGHTS = {
    # Replaced from the original hand-picked guess with data-derived weights,
    # based on Cohen's d effect sizes from au_feature_selection.py, run
    # across all 24 RAVDESS actors (tense-associated: angry/fearful/disgust
    # vs. baseline: neutral/calm). These are UNIVARIATE effect sizes, not
    # multivariate model coefficients - each AU's contribution here is
    # measured independently, so (unlike the v7 trained-model approach)
    # moving one AU always moves the score in a consistent, predictable
    # direction, with no risk of one AU's signal cancelling another's.
    #
    # AU07 and AU23 (the two weakest AUs from the original hand-picked set,
    # Cohen's d of 0.228 and 0.084 respectively) have been dropped entirely -
    # your own data showed they barely separate tense from baseline faces.
    "AU04_r": 0.228,  # brow lowerer - strongest discriminator (d=1.221)
    "AU09_r": 0.192,  # nose wrinkler (d=1.027)
    "AU10_r": 0.181,  # upper lip raiser (d=0.969)
    "AU20_r": 0.146,  # lip stretcher (d=0.781)
    "AU15_r": 0.131,  # lip corner depressor (d=0.699)
    "AU05_r": 0.122,  # upper lid raiser - wide eyes (d=0.652)
}

# Short human-readable labels for each AU, used to auto-generate the
# "Indikator yang dihitung" panel below - kept as a lookup separate from
# TENSION_WEIGHTS so the panel can be generated dynamically FROM the actual
# weights dict, instead of a hand-typed string that could silently drift
# out of sync if the weights are ever changed.
AU_LABELS = {
    "AU04_r": "brow lowerer",
    "AU09_r": "nose wrinkler",
    "AU10_r": "upper lip raiser",
    "AU20_r": "lip stretcher",
    "AU15_r": "lip corner depressor",
    "AU05_r": "upper lid raiser",
}

WINDOW_SIZE = 10
BASELINE_DURATION_SECONDS = 10

# ---------------------------------------------------------------------------
# New config - OpenFace launch + gaze + voice + threshold zones
# ---------------------------------------------------------------------------
OPENFACE_EXE = r"C:\OpenFace\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
WEBCAM_DEVICE_INDEX = 0  # matches the "-device" index passed to OpenFace below

# OpenFace's default output format for aligned face crops (the
# session1_aligned/frame_det_00_<frame>.<ext> files). ASSUMPTION: this is
# OpenFace 2.x's default (.bmp) - our OpenFace invocation above doesn't
# pass any -simalign/-format flag that would change it, so this should be
# correct, but if your install IS configured differently, update this to
# match (otherwise the clickable links in report.txt/report.html will
# point at a file that doesn't exist).
ALIGNED_FRAME_EXT = "bmp"

# ---------------------------------------------------------------------------
# v4.3 switches - set True / False
# ---------------------------------------------------------------------------
# Snapshot source #1 (NEW): copy the face image OpenFace already wrote for the
# exact frame where tension rose (session_<id>/session1_aligned/frame_det_00_<frame>.<ext>)
# into the snapshots/ folder. Does not touch the webcam, never blocks, cannot fail
# because the camera is busy. NOTE: it is OpenFace's aligned FACE CROP (small, 112x112 by
# default, pose-normalised), not the full camera picture - see OPENFACE_EXTRA_ARGS below
# to make it larger.
SNAPSHOT_FROM_ALIGNED = True
# Snapshot source #2 (the OLD way): open the webcam briefly and grab a full frame.
# Now runs on its own thread so it no longer freezes the window. Needs the webcam to
# be free while OpenFace is running, which often is NOT the case - hence off by default.
SNAPSHOT_FROM_WEBCAM = False
# Both True = two images per rise (one of each kind).

ALIGNED_WAIT_SECONDS = 5.0    # how long to wait for OpenFace to finish writing the aligned image
ALIGNED_NEAREST_FRAMES = 10   # if that exact frame wasn't written, use the nearest one within this many frames

# Extra command-line arguments appended to the FeatureExtraction call, e.g. a larger
# aligned face: ["-simsize", "224"]. Check the flag names with FeatureExtraction.exe -h.
OPENFACE_EXTRA_ARGS = []

# Use OpenFace's own per-frame "timestamp" column as the time axis instead of the moment
# the script happened to read the row (rows are read in ~0.5 s batches, so the old way gave
# every row in a batch the same time). Set False to go back to the old behaviour.
USE_VIDEO_TIMESTAMPS = True

# Seconds without any reliable face frame before the big status shows "WAJAH HILANG".
FACE_LOST_SECONDS = 2.0

# Save the tension log every N seconds to session_<id>/tension_log.partial.csv (LOCAL file
# only - nothing is sent anywhere), so a crash / power loss doesn't lose the whole session.
# Removed automatically after the normal end-of-session save. 0 = off.
AUTOSAVE_LOG_SECONDS = 30

# Count one continuous ELEVATED/HIGH stretch as ONE episode (listed under the highest level it
# reached) instead of a new episode every time the score crosses the ELEVATED/HIGH line.
# Changes the E:/H: counts and the report, so it is OFF by default.
MERGE_ELEVATED_HIGH_EPISODES = False

# ---------------------------------------------------------------------------
# Output layout - one folder per session, everything that session produced
# lives inside it:
#
#   C:\OpenFace\live_output\
#   |-- panduan_kolom.txt                  <- ONE shared copy, not per-session
#   `-- session_20260908_182023\
#       |-- session1.csv / .hog / .avi / session1_aligned\   (OpenFace's own output)
#       |-- tension_log.csv
#       |-- report.txt
#       `-- snapshots\
#           |-- BASELINE_12.05s_20260908_182035.jpg
#           `-- ...
#
# session_dir and the current_*_path globals below are only valid while a
# session is running/just finished - they're (re)computed once per session,
# in start_openface(), the same place session_id itself is generated.
# ---------------------------------------------------------------------------
OPENFACE_OUT_DIR = r"C:\OpenFace\live_output"          # base dir: holds the shared guide + one subfolder per session
GUIDE_PATH = os.path.join(OPENFACE_OUT_DIR, "panduan_kolom.txt")  # shared across all sessions - written once, not regenerated per run

session_dir = None             # OPENFACE_OUT_DIR/session_<id> - this session's own folder
current_csv_path = None        # session_dir/session1.csv - OpenFace's own output
current_log_path = None        # session_dir/tension_log.csv
current_snapshot_dir = None    # session_dir/snapshots
current_voice_path = None      # session_dir/voice_recording.wav

# (Identity/facial-recognition match lookup - ARCFACE_MODEL_NAME,
# FACE_DATABASE_DIR, find_target_in_database(), the target photo/info
# panel, etc. - has been removed in this variant. See
# prototype_OpenFace_v3_3_transcript.py for the version that keeps it.)

GAZE_WINDOW = 30            # frames used for gaze variance calculation

GAZE_VARIANCE_WEIGHT = 0.3  # how much gaze variability contributes to the raw score
VOICE_CHANNEL_WEIGHT = 0.25  # facial model validated at ROC-AUC 0.899 (RAVDESS cross-validation) vs
VISUAL_CHANNEL_WEIGHT = 0.75 # voice at ~0.70-0.72 - the split now roughly reflects each channel's
                              # actual demonstrated evidence quality, not an arbitrary 60/40 guess

# Same heuristic thresholds used elsewhere in this project - not validated,
# just a starting point for coloring the "is this elevated" zones.
ELEVATED_THRESHOLD = 0.5
HIGH_THRESHOLD = 1.0
HIGH_VARIABILITY_THRESHOLD = 0.6  # std-dev above this -> "fluctuating" note in the summary

# How long the score must stay OUT of a level before a re-entry counts as a
# fresh episode, rather than the same one continuing. Prevents a score
# oscillating right at a threshold boundary from being miscounted as many
# separate short episodes.
LEVEL_EXIT_COOLDOWN_SECONDS = 2.0


class LevelEpisodeTracker:
    """
    Tracks distinct ELEVATED/HIGH episodes across a session (not periodic
    snapshots) - e.g. "reached HIGH 3 separate times, totaling 18 seconds".
    A transition INTO a level starts an episode; a transition out closes it,
    but only after LEVEL_EXIT_COOLDOWN_SECONDS of staying out, so brief
    boundary flicker doesn't inflate the count.
    """

    def __init__(self):
        self.episodes = {"ELEVATED": [], "HIGH": []}
        self.active_level = None
        self.episode_start = None
        self.episode_last_seen = None
        self.pending_exit_since = None

    @staticmethod
    def classify(value):
        if value >= HIGH_THRESHOLD:
            return "HIGH"
        elif value >= ELEVATED_THRESHOLD:
            return "ELEVATED"
        return None

    def _finalize_active_episode(self, end_t):
        if self.active_level is None or self.episode_start is None:
            return
        duration = self.episode_last_seen - self.episode_start
        if duration > 0:
            self.episodes[self.active_level].append({
                "start": self.episode_start, "end": self.episode_last_seen, "duration": duration
            })
        self.active_level = None
        self.episode_start = None
        self.episode_last_seen = None
        self.pending_exit_since = None

    def update(self, t, value):
        level = self.classify(value)

        if MERGE_ELEVATED_HIGH_EPISODES and level is not None and self.active_level is not None:
            # still inside one continuous ELEVATED/HIGH stretch: just extend it; remember the highest level
            self.episode_last_seen = t
            self.pending_exit_since = None
            if level == "HIGH":
                self.active_level = "HIGH"
            return

        if level == self.active_level:
            if level is not None:
                self.episode_last_seen = t
                self.pending_exit_since = None  # still in it, cancel any pending exit
            return

        if level is not None:
            # a genuine change to a different tracked level (or from None
            # into one) - close whatever was active, start fresh immediately
            self._finalize_active_episode(t)
            self.active_level = level
            self.episode_start = t
            self.episode_last_seen = t
            self.pending_exit_since = None
            return

        # level is None - dropped below ELEVATED entirely. Don't close the
        # episode immediately; wait out the cooldown in case it's brief flicker.
        if self.active_level is not None:
            if self.pending_exit_since is None:
                self.pending_exit_since = t
            elif t - self.pending_exit_since >= LEVEL_EXIT_COOLDOWN_SECONDS:
                self._finalize_active_episode(t)

    def finalize(self, end_t):
        """Call once at session end to close out any still-active episode."""
        self._finalize_active_episode(end_t)

    def summary(self):
        result = {}
        for lvl in ("ELEVATED", "HIGH"):
            eps = self.episodes[lvl]
            durations = [e["duration"] for e in eps]
            result[lvl] = {
                "count": len(eps),
                "total_duration": sum(durations),
                "longest": max(durations) if durations else 0.0,
            }
        return result


level_tracker = LevelEpisodeTracker()
manual_events = []       # (elapsed_seconds, label) - operator-marked moments, e.g. "new question asked"
manual_event_lines = []  # matplotlib Line2D refs for the vertical markers, cleared on new Start

PEAK_HOLD_SECONDS = 8   # how long the peak marker stays before it starts decaying back down
PEAK_DECAY_PER_SECOND = 0.15  # once decaying, how fast the peak marker falls (score units/sec)

TIMEFRAME_OPTIONS = {
    "30s": 30,
    "1min": 60,
    "2min": 120,
}
DEFAULT_TIMEFRAME_LABEL = "1min"

# ---------------------------------------------------------------------------
# State - mostly unchanged from the original, with gaze/voice additions
# ---------------------------------------------------------------------------
tension_history = deque(maxlen=WINDOW_SIZE)
plot_timestamps = deque(maxlen=200)
plot_scores = deque(maxlen=200)

# Uncapped full-session history, separate from the capped deques above.
# plot_timestamps/plot_scores are capped for chart RENDERING (drawing every
# individual video frame's point isn't necessary or even visible at 200+
# points), but the periodic timeframe Summary needs the FULL interval's
# data to compute an accurate average - a capped ~200-point buffer at
# ~30fps only holds a few seconds of real history, which was silently
# making the 30s/1min/2min summary only reflect the last few seconds
# instead of the whole selected window.
all_points = []  # (elapsed_seconds, smoothed_score) - never trimmed during a session

# Separate per-channel tracking, so the chart can show visual-only and
# voice-only trends alongside the combined score - lets you see WHICH
# channel is actually driving a given spike, since the combined line alone
# can't distinguish "face flagged this" from "voice flagged this".
visual_tension_history = deque(maxlen=WINDOW_SIZE)
voice_tension_history = deque(maxlen=WINDOW_SIZE)
visual_plot_scores = deque(maxlen=200)
voice_plot_scores = deque(maxlen=200)
visual_baseline_scores = []
voice_baseline_scores = []
visual_baseline_mean = None
voice_baseline_mean = None

gaze_x_history = deque(maxlen=GAZE_WINDOW)
gaze_y_history = deque(maxlen=GAZE_WINDOW)

baseline_scores = []
baseline_mean = None
baseline_established = False
start_time = None
rows_seen = 0                # OpenFace rows read this session (for the report's data-quality line)
rows_unreliable = 0          # ...of which skipped (low confidence / face not detected)
first_video_ts = None        # OpenFace timestamp of the first reliable row (time axis starts there)
last_reliable_wall = None    # wall-clock time of the last reliable face frame
face_lost = False
_openface_exit_reported = False
_transcript_thread = None    # background transcript finalize thread (waited for on exit)
_autosave = {"last": 0.0, "written": 0}

last_row_count = 0
csv_header = None       # cached header line, set once per session
csv_byte_offset = 0     # how far into the CSV we've already read - avoids re-parsing the whole file every poll
log_rows = []

openface_process = None
voice_analyzer = None
session_started = False  # gates the update loop until the Start button is pressed
voice_enabled = False    # toggled via the "Enable Voice Detection" checkbox - takes effect on next Start
                         # off by default; operator must explicitly turn it on
# (No "Record Audio" checkbox any more: the microphone is ALWAYS recorded to
# session_<id>/voice_recording.wav, because the transcript is made from that file.)

transcriber = None           # the active Transcriber instance for this session, or None
voice_session_start = None   # wall-clock time the mic stream was started (to align transcript times)
_voice_stop_thread = None    # background thread finishing voice_analyzer.stop() (writes the .wav)
transcription_enabled = True   # transcription is ALWAYS on now (no checkbox). The session's audio is
                                # recorded automatically and transcribed after Stop; the result feeds the
                                # post-session "what was being said" context list.

selected_timeframe_seconds = TIMEFRAME_OPTIONS[DEFAULT_TIMEFRAME_LABEL]
last_conclusion_bucket = -1

# peak-hold gauge state
current_peak_value = 0.0
current_peak_last_rise_time = None

# Frames below this OpenFace tracking confidence are skipped entirely -
# a poorly tracked frame (bad lighting, extreme head angle, partial
# occlusion) produces unreliable AU values that would otherwise inject
# noise into the score as if it were a genuine expression change.
MIN_TRACKING_CONFIDENCE = 0.75


def row_is_reliable(row):
    success = row.get("success", 1)
    confidence = row.get("confidence", 1.0)
    try:
        if pd.notna(success) and float(success) == 0:
            return False
        if pd.notna(confidence) and float(confidence) < MIN_TRACKING_CONFIDENCE:
            return False
    except (TypeError, ValueError):
        pass
    return True


def _safe_round(val, ndigits=4):
    return round(float(val), ndigits) if pd.notna(val) else ""


def compute_au_score(row):
    score = 0
    for au, weight in TENSION_WEIGHTS.items():
        val = row.get(au, np.nan)
        if pd.notna(val):
            score += val * weight
    return score


def compute_gaze_variance_score():
    if len(gaze_x_history) < 2:
        return 0.0
    return (np.std(gaze_x_history) + np.std(gaze_y_history)) / 2.0


# Plain-language names for the log - lets an expert reviewing the CSV
# understand what moved without needing to know FACS AU codes by heart.
AU_HUMAN_LABELS = {
    "AU04_r": "Brow Lowerer (alis turun/mengerut)",
    "AU09_r": "Nose Wrinkler (hidung mengerut)",
    "AU10_r": "Upper Lip Raiser (bibir atas terangkat)",
    "AU20_r": "Lip Stretcher (bibir tertarik ke samping)",
    "AU15_r": "Lip Corner Depressor (sudut bibir turun)",
    "AU05_r": "Upper Lid Raiser (mata melebar)",
}


def _describe_dominant_signal(row):
    """Plain-language note on which specific facial action unit
    contributed most to this row's au_score - answers 'what moved' in
    words, not just 'a number moved', without needing AU-code knowledge
    to interpret. This is the main thing that lets an expert scan the
    log and connect a score change to an actual visible facial movement."""
    contributions = {}
    for au, weight in TENSION_WEIGHTS.items():
        val = row.get(au, np.nan)
        if pd.notna(val):
            contributions[au] = val * weight
    if not contributions:
        return ""
    top_au = max(contributions, key=contributions.get)
    if contributions[top_au] <= 0:
        return "Tidak ada AU menonjol"
    return AU_HUMAN_LABELS.get(top_au, top_au)


# ---------------------------------------------------------------------------
# Plot setup - same base line chart as the original, with threshold zones added
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(
    left=0.422,
    right=0.862,
    bottom=0.39,
    top=0.90,
)
# plt.subplots_adjust(bottom=0.39, right=0.70, top=0.90)  # move the main panel upward while preserving its height

line, = ax.plot([], [], color='black', linewidth=1.6, zorder=5, label="Combined")
visual_line, = ax.plot([], [], color='#1565c0', linewidth=1.0, alpha=0.6, zorder=4, label="Face only")
voice_line, = ax.plot([], [], color='#2e7d32', linewidth=1.0, alpha=0.6, zorder=4, label="Voice only")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Tension Score (relative to baseline)")
ax.set_title("Live Tension Monitor - PROTOTYPE")
ax.axhline(0, color='gray', linestyle='--', linewidth=1)

# ---------------------------------------------------------------------------
# Big glanceable status - one large word + trend arrow, meant to be readable
# without needing to read numbers, since the officer's attention should be
# on the interview, not the screen. Overlaid on the chart itself using axes-
# relative coordinates so it stays in place regardless of the data's scale.
# ---------------------------------------------------------------------------
STATUS_LABELS_ID = {
    "HIGH": "TINGGI",
    "ELEVATED": "MENINGKAT",
    "BASELINE": "STABIL",
    "BELOW": "TENANG",
}
big_status_text = ax.text(
    0.98, 0.95, "-", transform=ax.transAxes,
    fontsize=26, fontweight="bold", ha="right", va="top", color="#2e7d32", zorder=10,
)
trend_arrow_text = ax.text(
    0.98, 0.80, "", transform=ax.transAxes,
    fontsize=16, ha="right", va="top", color="dimgray", zorder=10,
)


def draw_threshold_zones():
    """Adds the colored background bands (baseline/elevated/high/below-baseline)
    without touching the line itself - called once up front and again whenever
    axis limits are recalculated, so the shading always matches the current view."""
    y_min, y_max = ax.get_ylim()
    ax.axhspan(HIGH_THRESHOLD, y_max, color="#b00020", alpha=0.10, zorder=0)
    ax.axhspan(ELEVATED_THRESHOLD, HIGH_THRESHOLD, color="#c77700", alpha=0.10, zorder=0)
    ax.axhspan(-ELEVATED_THRESHOLD, ELEVATED_THRESHOLD, color="#2e7d32", alpha=0.08, zorder=0)
    ax.axhspan(y_min, -ELEVATED_THRESHOLD, color="#1565c0", alpha=0.08, zorder=0)


# initial placeholder zones so the chart isn't blank before data arrives
ax.set_ylim(-2, 2)
draw_threshold_zones()


# ---------------------------------------------------------------------------
# NOTE (this variant only): the automatic identity/facial-recognition
# match panel that used to occupy this spot (target_photo_ax/target_info_ax
# - photo + matched-record lookup via find_target_in_database/DeepFace) has
# been removed entirely in this variant. The transcript panel below now
# lives in that freed space instead. The lower empty area in the left
# column (below this box, above the Timeframe selector) is intentionally
# left unused - only this one swap was requested.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Transcript panel - shows nothing but a short status message during the
# session. Transcription is always on and there is no live text: once Stop
# is pressed, the recording is transcribed and this panel shows the "what
# was being said" context for each ELEVATED/HIGH episode (full text, no
# truncation) - see _finalize_transcript_and_context /
# _check_transcript_finalize_result below.
# ---------------------------------------------------------------------------
# transcript_panel_ax = plt.axes([0.025, 0.190, 0.345, 0.660])
transcript_panel_bottom = 0.25
transcript_panel_top = 0.901 - 1 / (fig.get_figheight() * fig.dpi)
# axes now spans the whole bordered box (was starting ~0.05 below its top), so the title/text sit higher
transcript_panel_ax = plt.axes([0.04, transcript_panel_bottom, 0.33, transcript_panel_top - transcript_panel_bottom])
transcript_panel_ax.axis("off")
fig.add_artist(plt.Rectangle(
    (0.040, transcript_panel_bottom), 0.33,
    transcript_panel_top - transcript_panel_bottom, transform=fig.transFigure,
    fill=False, edgecolor="black", linewidth=1.2,
))
transcript_title = transcript_panel_ax.text(
    0.5, 0.955, "Transkrip", fontsize=10, fontweight="bold",
    ha="center", va="top", transform=transcript_panel_ax.transAxes,
)
# NOT wrap=True: matplotlib's auto-wrap measures against the whole FIGURE
# width, not this panel's width, which is what let long text run across the
# entire window. Wrapping is done manually (_wrap_to_panel, below) to this
# panel's real width instead, and the artist is hard-clipped to the panel
# box as a backstop so nothing can ever draw outside it.
transcript_text = transcript_panel_ax.text(
    0.04, 0.84, "Menunjukan konteks percakapan saat tingkat tension meningkat.", fontsize=8,
    ha="left", va="top", color="dimgray",
    transform=transcript_panel_ax.transAxes,
    clip_on=True,
)
transcript_text.set_clip_box(transcript_panel_ax.bbox)

TRANSCRIPT_GAP_PT = 20          # gap (points) from box top to title, and the SAME gap from title to the contents


def _layout_transcript_vertical():
    """Title sits TRANSCRIPT_GAP_PT below the top of the box; the contents start
    the same distance below the title. Returns the pixel height left for text."""
    renderer = fig.canvas.get_renderer()
    H = transcript_panel_ax.get_window_extent().height
    g = TRANSCRIPT_GAP_PT * fig.dpi / 72.0
    transcript_title.set_y(1 - g / H)
    title_h = transcript_title.get_window_extent(renderer).height
    transcript_text.set_y(1 - (2 * g + title_h) / H)
    return H - (2 * g + title_h) - g


TRANSCRIPT_SIDE_MARGIN = 0.04   # fraction of the panel width left empty on BOTH the left and the right
TRANSCRIPT_BASE_FONT = 8.0
TRANSCRIPT_FONT_SIZES = (8.0, 7.4, 6.8, 6.2, 5.6, 5.0)   # largest that fits the panel is used
_transcript_entries = []        # last post-session entries shown (re-wrapped if the window is resized)


def _wrap_to_panel(text, fontsize):
    """Word-wraps text to the panel's REAL pixel width (minus equal left/right
    margins), measured with the actual font - so lines run all the way across
    the panel instead of stopping at a guessed character count."""
    renderer = fig.canvas.get_renderer()
    fp = transcript_text.get_fontproperties().copy()
    fp.set_size(fontsize)
    max_w = transcript_panel_ax.get_window_extent().width * (1 - 2 * TRANSCRIPT_SIDE_MARGIN)

    def width(s):
        return renderer.get_text_width_height_descent(s, fp, False)[0]

    lines = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
            continue
        cur = ""
        for word in para.split(" "):
            trial = word if not cur else cur + " " + word
            if not cur or width(trial) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return "\n".join(lines)


def _render_transcript_entries(entries):
    """Shows ALL entries in full. If they don't fit the panel at the normal
    size, the font is shrunk (down to a minimum) until they do; only if they
    still don't fit at the smallest size are trailing entries left off, with
    a note pointing to report.txt (which always has everything)."""
    global _transcript_entries
    _transcript_entries = list(entries)
    renderer = fig.canvas.get_renderer()
    avail_h = _layout_transcript_vertical()

    n = len(entries)
    while True:
        for fs in TRANSCRIPT_FONT_SIZES:
            body = "\n\n".join(entries[:n])
            if n < len(entries):
                body += f"\n\n(+{len(entries) - n} konteks lainnya - lihat report.txt)"
            transcript_text.set_fontsize(fs)
            transcript_text.set_text(_wrap_to_panel(body, fs))
            if transcript_text.get_window_extent(renderer).height <= avail_h:
                return
        if n <= 1:
            return
        n -= 1


def _on_transcript_resize(event):
    _layout_transcript_vertical()
    if _transcript_entries:
        _render_transcript_entries(_transcript_entries)


fig.canvas.mpl_connect("resize_event", _on_transcript_resize)
_layout_transcript_vertical()


def update_transcript_panel():
    """Called every update cycle during a session - only a status message
    (there is no live text). After Stop, the finalize-and-poll pair below
    takes over this same artist for the post-session context list."""
    transcript_text.set_fontsize(TRANSCRIPT_BASE_FONT)
    transcript_text.set_color("dimgray")
    if transcriber is None:
        transcript_text.set_text("Transkripsi: mikrofon tidak aktif\n(gagal dibuka atau belum siap).")
        return
    transcript_text.set_text(
        "Rekaman suara akan diproses setelah session selesai."
    )


# Result handoff for the background post-session transcript finalize step -
# same "poll a one-slot mailbox from a timer" pattern already used for
# _identity_lookup_result above: None = nothing pending, "pending" = the
# background thread is still running (joining Transcriber.stop(), which can
# take a few seconds), a tuple = ready to apply to the GUI.
_pending_transcript_result = [None]


REPORT_TRANSCRIPT_PLACEHOLDER = "   (Transkrip belum tersedia - sedang diproses. Bagian ini terisi otomatis begitu selesai.)"


def _insert_into_report(report_path, body, report_ready=None):
    """Runs on the transcript thread (no GUI involved), so the report is completed even if
    the window has been closed in the meantime. Replaces the placeholder in section 4 of
    report.txt (the report keeps its closing notes at the very end)."""
    if not report_path:
        return
    if report_ready is not None:
        report_ready.wait(timeout=60)   # report.txt is created by stop_openface
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            text = f.read()
        if REPORT_TRANSCRIPT_PLACEHOLDER in text:
            text = text.replace(REPORT_TRANSCRIPT_PLACEHOLDER, body)
        else:
            text += "\n" + body + "\n"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Transcript context added to {report_path}")
    except Exception as e:
        print(f"Could not add transcript context to report: {e}")


def _finalize_transcript_and_context(transcriber_obj, episodes, report_path,
                                     voice_stop_thread=None, wav_path=None, time_offset=0.0,
                                     report_ready=None, ep_lookup=None):
    """Runs on a background thread (spawned from stop_openface) - NOT on
    the GUI thread, since transcriber_obj.stop() blocks for a few seconds
    while Whisper finishes anything still queued. Builds the heuristic
    post-session context list for every ELEVATED/HIGH episode and leaves
    the result in _pending_transcript_result for the timer below to pick
    up and apply to the GUI."""
    if TRANSCRIBER_BACKEND == "assemblyai_async":
        if voice_stop_thread is not None:
            voice_stop_thread.join(timeout=30)  # wait until the .wav is fully written
        transcriber_obj.transcribe_file(wav_path, time_offset)
    else:
        transcriber_obj.stop()

    if transcriber_obj._load_error and not transcriber_obj.get_segments():
        err = transcriber_obj._load_error
        _insert_into_report(report_path, f"   Transkripsi gagal: {err}", report_ready)
        _pending_transcript_result[0] = ([f"Transkripsi gagal:\n{err}"], "", None)
        return

    ep_lookup = ep_lookup or {}
    all_entries = []  # (block for report.txt, text for the on-screen panel)
    any_ctx = False
    for k, g in enumerate(group_episode_contexts(transcriber_obj, episodes), start=1):
        ctx = g["ctx"]
        title = " + ".join(g["levels"])
        if len(g["levels"]) == 1:
            when = ", ".join(f"{t:.0f}s" for t in g["times"][g["levels"][0]])
        else:
            when = "; ".join(f"{l}: " + ", ".join(f"{t:.0f}s" for t in g["times"][l]) for l in g["levels"])

        # which numbered spikes (section 2 of the report) belong to this context
        spikes = sorted(
            (t, lvl, ep_lookup.get((lvl, t))) for lvl in g["levels"] for t in g["times"][lvl])
        spike_txt = ", ".join(
            (f"#{n} " if n else "") + f"{lvl} {_mmss(t)}" for t, lvl, n in spikes)

        if ctx is None:
            panel = f"{title} (naik pada {when}): (tidak ada ucapan terekam di sekitar momen ini)"
            block = (f"   Konteks {k}  -  tidak ada ucapan terekam di sekitar momen ini\n"
                     f"   Lonjakan  : {spike_txt}")
        else:
            q_tag = " [berisi pertanyaan]" if ctx["has_question"] else ""
            span = f"[{ctx['context_start']:.0f}s-{ctx['context_end']:.0f}s]"
            quote = ctx["text"]
            panel = f"{title} {span}{q_tag}\n(naik pada {when}):\n\"{quote}\""   # full quote, never truncated
            head = f"   Konteks {k}  -  {_mmss(ctx['context_start'])} s/d {_mmss(ctx['context_end'])}"
            if ctx["has_question"]:
                head += "  -  berisi pertanyaan"
            quote_block = textwrap.fill(f'"{quote}"', width=REPORT_WIDTH,
                                        initial_indent="      ", subsequent_indent="      ")
            block = f"{head}\n   Lonjakan  : {spike_txt}\n   Ucapan    :\n{quote_block}"
            any_ctx = True
        all_entries.append((block, panel))

    body_lines = []
    for block, _panel in all_entries:
        body_lines.append(block)
        body_lines.append("")
    if not all_entries:
        body_lines = [_wrap("(Tidak ada lonjakan ELEVATED/HIGH pada sesi ini, jadi tidak ada konteks yang ditampilkan.)", 3)]
    elif not any_ctx:
        body_lines.append("   (tidak ada ucapan yang berhasil dicocokkan ke lonjakan manapun)")
    while body_lines and body_lines[-1] == "":
        body_lines.pop()

    if not all_entries:
        panel_entries = ["(Tidak ada episode ELEVATED/HIGH pada sesi ini.)"]
    else:
        panel_entries = [p for _b, p in all_entries]
    _insert_into_report(report_path, "\n".join(body_lines), report_ready)
    _pending_transcript_result[0] = (panel_entries, "", None)   # GUI only displays; the file is already written


def _check_transcript_finalize_result():
    """Polled by a dedicated timer (started below), same pattern as
    _check_identity_lookup_result - cheap no-op until the background
    finalize thread actually finishes, then applies the result to the GUI
    here, on the main thread."""
    result = _pending_transcript_result[0]
    if result is None:
        return
    _pending_transcript_result[0] = None

    panel_entries, report_section, report_path = result
    transcript_text.set_color("black")
    _render_transcript_entries(panel_entries)
    if report_path:
        try:
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(report_section + "\n")
            print(f"Transcript context appended to {report_path}")
        except Exception as e:
            print(f"Could not append transcript context to report: {e}")
    fig.canvas.draw_idle()


_transcript_finalize_timer = fig.canvas.new_timer(interval=400)
_transcript_finalize_timer.add_callback(_check_transcript_finalize_result)
_transcript_finalize_timer.start()


def _grab_clean_webcam_frame(device_index=0, warmup_frames=3):
    """Briefly opens its OWN capture of the webcam - separate from
    OpenFace's - grabs one clean frame, and immediately releases it.

    Risk worth knowing: most webcams only allow one process to hold the
    device open at a time, and OpenFace already has it open during
    calibration. This is a brief open-grab-release (not a persistent
    second stream), which is much likelier to succeed than holding a
    second handle open continuously - but it can still fail on some
    hardware/drivers. Returns None on any failure so the caller can fall
    back gracefully instead of crashing."""
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        for _ in range(warmup_frames):
            cap.read()
        ret, frame = cap.read()
    finally:
        cap.release()
    return frame if ret else None


# ---------------------------------------------------------------------------
# Tension-episode snapshots (WEBCAM-GRAB VARIANT) - takes a photo the
# instant tension crosses INTO ELEVATED or HIGH (not continuously while
# active), so an expert can see exactly what the face looked like at each
# rise, not just that a number crossed a threshold.
#
# This variant reuses _grab_clean_webcam_frame() above - a brief, separate
# cv2.VideoCapture open-grab-release. Unlike the identity lookup's use of
# the same helper (which runs BEFORE OpenFace claims the device), this
# runs MID-SESSION, competing with OpenFace for a device it already holds
# exclusively. Whether this succeeds depends entirely on your webcam
# driver - test this on your actual hardware. If it fails, you'll see a
# console message and no snapshot for that episode, but the session
# continues normally either way. The upside if it works: a genuinely
# clean, unannotated photo (no tracking overlay).
#
# Snapshots are saved inside THIS session's own folder (session_dir/
# snapshots/, set in start_openface) - same place as that session's log
# CSV and report, so one session's files are never scattered across
# session-ID-named subfolders in a separate top-level snapshots/ tree.
# ---------------------------------------------------------------------------
session_id = None       # set once per session in start_openface, e.g. "20260908_182023"
tension_snapshots = []  # (elapsed_seconds, level, saved_path) - persisted into the log at Stop


_webcam_grab_lock = threading.Lock()


def _save_webcam_snapshot(level, elapsed, snap_dir):
    """OLD method (SNAPSHOT_FROM_WEBCAM): brief separate webcam open-grab-release.
    Runs on its own thread so a slow/failed grab can't freeze the window."""
    with _webcam_grab_lock:
        frame = _grab_clean_webcam_frame(WEBCAM_DEVICE_INDEX)
    if frame is None:
        print(f"Could not capture webcam snapshot for {level} episode at {elapsed:.0f}s "
              f"(webcam likely unavailable while OpenFace holds it)")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    os.makedirs(snap_dir, exist_ok=True)
    filename = f"{level}_{elapsed:.0f}s_{timestamp}.jpg"
    path = os.path.join(snap_dir, filename)
    try:
        cv2.imwrite(path, frame)
        tension_snapshots.append((elapsed, level, path))
        print(f"Webcam snapshot saved for {level} episode at {elapsed:.0f}s: {path}")
    except Exception as e:
        print(f"Could not save snapshot: {e}")


def _save_aligned_snapshot(level, elapsed, frame_no, aligned_dir, snap_dir):
    """NEW method (SNAPSHOT_FROM_ALIGNED): copy OpenFace's own aligned face image for
    the frame where tension rose. OpenFace writes images a moment after the CSV row,
    so wait (on this thread) until the file exists and has stopped growing."""
    if frame_no is None or frame_no < 0:
        print(f"Aligned snapshot skipped for {level} at {elapsed:.0f}s: unknown frame number")
        return
    src = os.path.join(aligned_dir, f"frame_det_00_{frame_no:06d}.{ALIGNED_FRAME_EXT}")
    deadline = time.time() + ALIGNED_WAIT_SECONDS
    ready = False
    while time.time() < deadline:
        if os.path.exists(src):
            size1 = os.path.getsize(src)
            time.sleep(0.15)
            if size1 > 0 and os.path.getsize(src) == size1:
                ready = True
                break
        time.sleep(0.1)

    used_frame = frame_no
    if not ready:
        # that exact frame wasn't written (OpenFace can skip badly aligned frames) - use the nearest one
        best = None
        for f in glob.glob(os.path.join(aligned_dir, f"frame_det_00_*.{ALIGNED_FRAME_EXT}")):
            m = re.search(r"frame_det_00_(\d+)\.", os.path.basename(f))
            if m:
                d = abs(int(m.group(1)) - frame_no)
                if d <= ALIGNED_NEAREST_FRAMES and (best is None or d < best[0]):
                    best = (d, int(m.group(1)), f)
        if best is None:
            print(f"Aligned snapshot not found for {level} at {elapsed:.0f}s (frame {frame_no}) in {aligned_dir} "
                  f"- is OpenFace writing aligned faces?")
            return
        _d, used_frame, src = best
        print(f"Aligned frame {frame_no} missing - using nearest frame {used_frame}")

    try:
        os.makedirs(snap_dir, exist_ok=True)
        dst = os.path.join(snap_dir, f"{level}_{elapsed:.0f}s_frame{used_frame:06d}_aligned.{ALIGNED_FRAME_EXT}")
        shutil.copy2(src, dst)
        tension_snapshots.append((elapsed, level, dst))
        print(f"Aligned snapshot saved for {level} episode at {elapsed:.0f}s (frame {used_frame}): {dst}")
    except Exception as e:
        print(f"Could not save aligned snapshot: {e}")


_snapshot_threads = []


def save_tension_snapshot(level, elapsed, frame_no=None):
    """Called the instant tension rises into a level (and once for BASELINE).
    Never blocks the window: each enabled method runs on its own thread."""
    _snapshot_threads[:] = [t for t in _snapshot_threads if t.is_alive()]
    if SNAPSHOT_FROM_ALIGNED and session_dir:
        t = threading.Thread(
            target=_save_aligned_snapshot,
            args=(level, elapsed, frame_no, os.path.join(session_dir, "session1_aligned"), current_snapshot_dir),
            daemon=True,
        )
        t.start()
        _snapshot_threads.append(t)
    if SNAPSHOT_FROM_WEBCAM and current_snapshot_dir:
        t = threading.Thread(
            target=_save_webcam_snapshot, args=(level, elapsed, current_snapshot_dir), daemon=True,
        )
        t.start()
        _snapshot_threads.append(t)


# ---------------------------------------------------------------------------
# report.txt and panduan_kolom.txt builders
# ---------------------------------------------------------------------------
REPORT_WIDTH = 76
_BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
          "Agustus", "September", "Oktober", "November", "Desember"]


def _mmss(t):
    t = max(0, int(round(t)))
    return f"{t // 60:02d}:{t % 60:02d}"


def _dur(t):
    if t >= 60:
        return f"{int(t // 60)}m {int(round(t % 60))}s"
    return f"{t:.1f}s"


def _tanggal(dt):
    return f"{dt.day} {_BULAN[dt.month - 1]} {dt.year}"


def number_episodes(episodes):
    """Numbers every ELEVATED/HIGH episode chronologically (#1, #2, ...) so the
    report's tables, details and transcript contexts can refer to each other.
    Returns ([(n, level, episode), ...], {(level, start): n})."""
    items = []
    for lvl in ("ELEVATED", "HIGH"):
        for ep in episodes.get(lvl, []):
            items.append((ep["start"], lvl, ep))
    items.sort(key=lambda x: x[0])
    numbered = [(i, lvl, ep) for i, (_t, lvl, ep) in enumerate(items, start=1)]
    return numbered, {(lvl, ep["start"]): n for n, lvl, ep in numbered}


def _wrap(text, indent=0, width=REPORT_WIDTH):
    return textwrap.fill(text, width=width, initial_indent=" " * indent, subsequent_indent=" " * indent)


def _kv(label, value, label_w=18):
    pad = " " * (1 + label_w + 3)
    return textwrap.fill(value, width=REPORT_WIDTH, initial_indent=f" {label:<{label_w}} : ",
                         subsequent_indent=pad)


def _episode_details(log_df, lvl, ep):
    """Per-episode facts from the log: peak/mean score, what moved, which channel drove it."""
    seg = log_df[(log_df["elapsed_seconds"] >= ep["start"] - 1e-6) & (log_df["elapsed_seconds"] <= ep["end"] + 1e-6)]
    d = {"peak": None, "mean": None, "signals": "", "driver": ""}
    if seg.empty:
        return d
    d["peak"] = float(seg["smoothed_score"].max())
    d["mean"] = float(seg["smoothed_score"].mean())
    dom = seg["dominant_signal"].astype(str)
    dom = dom[(dom != "") & (dom != "nan") & (dom != "Tidak ada AU menonjol")]
    if len(dom):
        top = dom.value_counts().head(2)
        d["signals"] = "; ".join(f"{name} ({cnt / len(dom):.0%})" for name, cnt in top.items())
    if "voice_score" in seg and seg["voice_score"].abs().sum() > 0:
        face = VISUAL_CHANNEL_WEIGHT * float(seg["visual_smoothed"].mean())
        voice = VOICE_CHANNEL_WEIGHT * float(seg["voice_smoothed"].mean())
        pos_face, pos_voice = max(face, 0.0), max(voice, 0.0)
        tot = pos_face + pos_voice
        if tot > 0:
            d["driver"] = f"wajah {pos_face / tot:.0%} / suara {pos_voice / tot:.0%}"
    else:
        d["driver"] = "wajah (analisis suara tidak aktif)"
    return d


def compute_session_summary(log_df):
    """Numbers behind the session summary - used by BOTH report.txt (section 1) and the
    on-screen Summary panel, so the two can never disagree."""
    secs = {"BELOW": 0.0, "BASELINE": 0.0, "ELEVATED": 0.0, "HIGH": 0.0}
    monitored = 0.0
    if len(log_df) > 1:
        dt = log_df["elapsed_seconds"].diff().shift(-1).fillna(0).clip(lower=0, upper=1.0)
        for lvl_name, grp in dt.groupby(log_df["level"]):
            secs[lvl_name] = float(grp.sum())
        monitored = float(dt.sum())
    peak_row = log_df.loc[log_df["smoothed_score"].idxmax()] if len(log_df) else None
    return {
        "secs": secs,
        "monitored": monitored,
        "peak": None if peak_row is None else (float(peak_row["smoothed_score"]),
                                               float(peak_row["elapsed_seconds"]), str(peak_row["level"])),
        "mean": float(log_df["smoothed_score"].mean()) if len(log_df) else None,
    }


def summary_table_lines(stats, summ, compact=False):
    """The summary block as (text, kind) lines, kind in header/ELEVATED/HIGH/text.
    Fixed-width layout - monospace in report.txt and in the Summary panel.
    compact=True (Summary panel): the long 'Waktu per level' line is split in two and the
    wording is shortened, so the widest line is shorter and the panel can use a bigger font."""
    monitored = summ["monitored"]
    out = [(f"{'Level':<10}{'Kejadian':>9}{'Total waktu':>14}{'% pemantauan':>14}{'Terlama':>10}", "header")]
    for lvl in ("ELEVATED", "HIGH"):
        st = stats[lvl]
        share = (st["total_duration"] / monitored * 100) if monitored else 0.0
        out.append((f"{lvl:<10}{str(st['count']) + 'x':>9}{_dur(st['total_duration']):>14}"
                    f"{share:>13.1f}%{_dur(st['longest']):>10}", lvl))
    if not compact:
        out.append(("", "text"))
    if monitored:
        values = "  ".join(f"{name} {summ['secs'][name] / monitored * 100:.0f}%"
                           for name in ("BELOW", "BASELINE", "ELEVATED", "HIGH"))
        if compact:
            out.append(("Waktu per level:", "text"))
            out.append(("  " + values, "text"))
        else:
            out.append((f"{'Waktu per level':<18} : " + values, "text"))
    if summ["peak"] is not None:
        pv, pt, pl = summ["peak"]
        if compact:
            out.append((f"Skor tertinggi : {pv:+.2f} pada {_mmss(pt)} ({pl})", "text"))
            out.append((f"Skor rata-rata : {summ['mean']:+.2f} (0 = baseline)", "text"))
        else:
            out.append((f"{'Skor tertinggi':<18} : {pv:+.2f} pada {_mmss(pt)}  ({pl})", "text"))
            out.append((f"{'Skor rata-rata':<18} : {summ['mean']:+.2f}  (0 = sama dengan kondisi awal)", "text"))
    return out


def build_report_text(log_df, stats, end_t, ended_at, transcript_pending):
    W = REPORT_WIDTH
    bar, thin = "=" * W, "-" * W
    started_at = datetime.strptime(session_id, "%Y%m%d_%H%M%S") if session_id else ended_at
    numbered, _lookup = number_episodes(level_tracker.episodes)

    summ = compute_session_summary(log_df)
    t_first = float(log_df["elapsed_seconds"].iloc[0]) if len(log_df) else 0.0

    L = [bar, " LAPORAN SESI - LIVE TENSION MONITOR", bar]
    L.append(_kv("Sesi", f"{os.path.basename(session_dir)}"))
    L.append(_kv("Tanggal", f"{_tanggal(started_at)}, {started_at:%H:%M:%S} - {ended_at:%H:%M:%S}"))
    L.append(_kv("Durasi", f"{_mmss(end_t)} total  (kalibrasi baseline {BASELINE_DURATION_SECONDS}s, "
                           f"pemantauan {_mmss(max(0.0, end_t - t_first))})"))
    L.append(_kv("Baseline", f"{baseline_mean:+.3f}  (rata-rata skor mentah selama kalibrasi = kondisi awal target)"
                 if baseline_mean is not None else "belum terbentuk"))
    valid = rows_seen - rows_unreliable
    pct = (valid / rows_seen * 100) if rows_seen else 0.0
    L.append(_kv("Kualitas data", f"{rows_seen} frame dibaca, {valid} valid ({pct:.1f}%), "
                                  f"{rows_unreliable} dilewati (wajah tidak terdeteksi / keyakinan < {MIN_TRACKING_CONFIDENCE})"))
    L.append(_kv("Sinyal dipakai", "wajah (AU + gaze)" + (" + suara" if voice_enabled else "  (analisis suara OFF)")))
    L.append(_kv("Rekaman suara", "voice_recording.wav (di folder sesi)"))
    L.append(_kv("Ambang level", f"ELEVATED >= {ELEVATED_THRESHOLD:.1f}, HIGH >= {HIGH_THRESHOLD:.1f}  (selisih dari baseline)"))
    if len(log_df) and pct < 80:
        L.append("")
        L.append(_wrap("PERHATIAN: kurang dari 80% frame valid - pencahayaan, sudut wajah atau posisi kamera "
                       "kemungkinan mengganggu; baca hasil dengan hati-hati.", 1))

    # ---- 1. ringkasan
    L += ["", bar, " 1. RINGKASAN", bar]
    tbl = summary_table_lines(stats, summ)
    L.append(" " + tbl[0][0])
    L.append(" " + thin[:57])
    for text, _kind in tbl[1:]:
        L.append((" " + text) if text else "")

    # ---- 2. lonjakan
    L += ["", bar, " 2. DAFTAR LONJAKAN TENSION (urut waktu)", bar]
    if not numbered:
        L.append(" Tidak ada lonjakan ELEVATED/HIGH pada sesi ini.")
    else:
        L.append(f" {'No':<4}{'Waktu':<8}{'Level':<10}{'Durasi':>8}{'Puncak':>9}{'Rata2':>8}   Pendorong")
        L.append(" " + thin[:70])
        details = {}
        for n, lvl, ep in numbered:
            details[n] = _episode_details(log_df, lvl, ep)
            dd = details[n]
            pk = f"{dd['peak']:+.2f}" if dd["peak"] is not None else "-"
            mn = f"{dd['mean']:+.2f}" if dd["mean"] is not None else "-"
            L.append(f" #{n:<3}{_mmss(ep['start']):<8}{lvl:<10}{_dur(ep['duration']):>8}{pk:>9}{mn:>8}   {dd['driver']}")
        L.append("")
        L.append(" Rincian tiap lonjakan:")
        for n, lvl, ep in numbered:
            dd = details[n]
            L.append(thin)
            L.append(f" #{n}  {lvl}  -  naik {_mmss(ep['start'])}, aktif sampai {_mmss(ep['end'])} (durasi {_dur(ep['duration'])})")
            if dd["signals"]:
                L.append(_kv("Yang bergerak", dd["signals"], 18))
            if dd["driver"]:
                L.append(_kv("Pendorong", dd["driver"], 18))
            if len(log_df):
                idx = (log_df["elapsed_seconds"] - ep["start"]).abs().idxmin()
                fno = int(log_df.loc[idx, "frame"])
                L.append(_kv("Frame", f"{fno:06d}  ->  session1_aligned/frame_det_00_{fno:06d}.{ALIGNED_FRAME_EXT}", 18))
            snaps = [os.path.basename(pth) for (se, sl, pth) in tension_snapshots
                     if sl == lvl and abs(se - ep["start"]) < 0.6]
            if snaps:
                L.append(_kv("Foto", "snapshots/" + ", snapshots/".join(sorted(snaps)), 18))
            before = [(t, lab) for (t, lab) in manual_events if ep["start"] - 30 <= t <= ep["start"]]
            if before:
                t_mark = max(before)[0]
                L.append(_kv("Penanda operator", f"ditandai pada {_mmss(t_mark)} ({ep['start'] - t_mark:.0f}s sebelum naik)", 18))
        L.append(thin)

    # ---- 3. operator
    L += ["", bar, " 3. CATATAN OPERATOR (tombol '.')", bar]
    if not manual_events:
        L.append(" Tidak ada penanda dari operator.")
    else:
        for t, label in sorted(manual_events):
            lvl_now = ""
            if len(log_df):
                idx = (log_df["elapsed_seconds"] - t).abs().idxmin()
                lvl_now = f"  (level saat itu: {log_df.loc[idx, 'level']})"
            L.append(f" {_mmss(t)}  {label}{lvl_now}")

    # ---- 4. konteks percakapan
    L += ["", bar, " 4. KONTEKS PERCAKAPAN SAAT TENSION NAIK", bar]
    L.append(_wrap("Heuristik dari jeda bicara - BUKAN AI/LLM. Teks hasil transkripsi otomatis bisa keliru; "
                   "cocokkan dengan rekaman (voice_recording.wav) sebelum dipakai.", 1))
    L.append("")
    L.append(REPORT_TRANSCRIPT_PLACEHOLDER if transcript_pending
             else "   (Transkripsi tidak aktif sesi ini - mikrofon gagal dibuka.)")

    # ---- 5. catatan
    L += ["", bar, " 5. CATATAN PENTING", bar]
    L.append(_wrap("Ini ringkasan POLA heuristik, BUKAN kesimpulan kejujuran/kebohongan target. Peningkatan "
                   "tension bisa karena gugup, berpikir keras, tidak nyaman, atau sekadar berbicara. Gunakan "
                   "sebagai satu bahan pertimbangan bersama observasi langsung dan konteks wawancara.", 1))
    L.append("")
    L.append(_wrap("Arti kolom tension_log.csv dan cara skor dihitung: lihat panduan_kolom.txt di folder utama "
                   "(satu level di atas folder sesi ini).", 1))
    L.append(bar)
    return "\n".join(L) + "\n"


def _cols(rows, name_w=27):
    """Two-column list: name on the left, wrapped description on the right (a name too long
    for the left column gets its own line instead of running into the description)."""
    out = []
    for name, desc in rows:
        if len(name) >= name_w:
            out.append(f"   {name}")
            out.append(textwrap.fill(desc, width=REPORT_WIDTH, initial_indent="   " + " " * name_w,
                                     subsequent_indent="   " + " " * name_w))
        else:
            out.append(textwrap.fill(desc, width=REPORT_WIDTH, initial_indent=f"   {name:<{name_w}}",
                                     subsequent_indent="   " + " " * name_w))
    return "\n".join(out)


def _item(text):
    """Numbered / bulleted item with a hanging indent."""
    hang = 2 if text.startswith("- ") else 3
    return textwrap.fill(text, width=REPORT_WIDTH, initial_indent=" ", subsequent_indent=" " * (1 + hang))


def build_column_guide():
    """Shared guide, generated from the real settings so it can never disagree with the code."""
    bar, thin = "=" * REPORT_WIDTH, "-" * REPORT_WIDTH
    au_rows = [(f"{au}  (bobot {w:.3f})", AU_HUMAN_LABELS.get(au, au))
               for au, w in sorted(TENSION_WEIGHTS.items(), key=lambda kv: -kv[1])]
    G = [bar, " PANDUAN MEMBACA HASIL SESI", bar,
         _wrap("Satu file untuk semua sesi (dibuat otomatis dari pengaturan skrip, jadi selalu cocok dengan "
               "angka yang dipakai). Dibuat untuk dibaca ahli yang belum mengenal kode AU / FACS.", 1),
         "", " ISI", "   A. Isi folder sesi", "   B. Cara cepat membaca tension_log.csv",
         "   C. Arti kolom tension_log.csv", "   D. Cara skor dihitung", "   E. Cara membaca report.txt",
         "   F. Batasan penting"]

    G += ["", bar, " A. ISI FOLDER SESI  (session_<tanggal_jam>\\)", bar,
          _cols([
              ("report.txt", "Laporan ringkas sesi: durasi, jumlah lonjakan, rincian tiap lonjakan, konteks ucapan."),
              ("tension_log.csv", "Log lengkap, satu baris per frame yang valid (lihat bagian C)."),
              ("voice_recording.wav", "Rekaman suara seluruh sesi (selalu direkam; dasar transkrip)."),
              ("snapshots\\", "Foto saat tension naik (dan satu foto baseline). Nama file = level_detik_frame."),
              ("session1.csv", "Output mentah OpenFace (semua kolom, tanpa penyaringan)."),
              ("session1_aligned\\", "Gambar wajah hasil OpenFace per frame (frame_det_00_<nomor frame>)."),
              ("tension_log.partial.csv", "Hanya ada jika sesi berhenti tiba-tiba: salinan log otomatis tiap "
                                           f"{AUTOSAVE_LOG_SECONDS}s. Normalnya dihapus saat Stop."),
          ], 24)]

    G += ["", bar, " B. CARA CEPAT MEMBACA tension_log.csv", bar,
          _item("1. Filter kolom 'level' = ELEVATED atau HIGH untuk melihat momen yang menonjol."),
          _item("2. Cek keandalan dulu: openface_success = 1 dan openface_confidence tinggi. "
                "Baris yang meragukan sebaiknya diabaikan."),
          _item("3. Lihat 'dominant_signal' dan kolom AU untuk tahu gerakan wajah apa yang naik; "
                "buka foto di kolom 'tension_snapshot' dan putar video pada 'elapsed_seconds' yang sama."),
          _item("4. Cocokkan dengan kolom 'manual_event' (penanda operator) dan bagian konteks ucapan di report.txt.")]

    G += ["", bar, " C. ARTI KOLOM tension_log.csv", bar, " Identitas baris", thin,
          _cols([
              ("timestamp", "Jam asli (ISO) saat baris dicatat."),
              ("elapsed_seconds", "Detik sejak awal rekaman OpenFace (jam internal frame, bukan jam komputer). "
                                  "Sama dengan sumbu waktu grafik, report.txt, foto, dan transkrip."),
              ("frame", "Nomor frame OpenFace -> gambar session1_aligned/frame_det_00_<frame>."),
          ]),
          "", " Keandalan data  (cek ini dulu)", thin,
          _cols([
              ("openface_success", "1 = wajah terdeteksi, 0 = tidak. Baris 0 jangan dipakai."),
              ("openface_confidence", f"0-1; makin tinggi makin yakin. Frame di bawah {MIN_TRACKING_CONFIDENCE} sudah "
                                      "otomatis dibuang dan tidak muncul di log."),
          ]),
          "", " Sinyal wajah mentah - Action Units (0 sampai ~5, makin tinggi makin kuat)", thin,
          _cols([
              ("AU04_brow_lowerer", "Alis turun/mengerut (mengernyit)"),
              ("AU09_nose_wrinkler", "Hidung mengerut (jijik/tidak nyaman)"),
              ("AU10_upper_lip_raiser", "Bibir atas terangkat"),
              ("AU20_lip_stretcher", "Bibir tertarik ke samping (tegang)"),
              ("AU15_lip_corner_depressor", "Sudut bibir turun (murung/kecewa)"),
              ("AU05_upper_lid_raiser", "Mata melebar (terkejut/waspada)"),
              ("dominant_signal", "AU yang paling besar sumbangannya ke au_score pada baris itu = jawaban singkat "
                                  "'apa yang bergerak'."),
          ]),
          "", " Gaze (arah pandangan)", thin,
          _cols([
              ("gaze_angle_x / _y", "Sudut pandangan mata mentah (radian)."),
              ("gaze_variance_score", f"Seberapa banyak arah pandangan berubah dalam {GAZE_WINDOW} frame terakhir "
                                      "(mata tidak stabil)."),
          ]),
          "", " Suara (terisi hanya jika 'Enable Voice Detection' aktif)", thin,
          _cols([
              ("voice_pitch_mean_hz", "Rata-rata nada suara (Hz)."),
              ("voice_pitch_variability", "Naik-turunnya nada."),
              ("voice_jitter", "Ketidakstabilan siklus pitch (getaran suara)."),
              ("voice_pause_ratio", "Proporsi jeda/diam pada potongan suara itu."),
              ("voice_intensity_variability", "Naik-turunnya volume."),
          ]),
          "", " Skor (hasil hitungan, bukan data mentah)", thin,
          _cols([
              ("au_score", "Wajah saja: jumlah (AU x bobot)."),
              ("voice_score", "Suara saja (0 jika analisis suara OFF)."),
              ("raw_score", "Gabungan sebelum dibandingkan ke baseline."),
              ("baseline", "Rata-rata raw_score selama kalibrasi = 'kondisi normal' target ini."),
              ("relative_score", "raw_score - baseline. 0 = sama seperti kondisi awal (bukan skala mutlak)."),
              ("smoothed_score", f"relative_score dirata-rata {WINDOW_SIZE} baris terakhir. INI garis utama di layar."),
              ("visual_smoothed / voice_smoothed", "Versi per-channel (wajah / suara) untuk melihat siapa pendorongnya."),
              ("level", "BELOW / BASELINE / ELEVATED / HIGH dari smoothed_score, dengan ambang yang sama seperti "
                        "di layar."),
          ]),
          "", " Catatan operator & foto", thin,
          _cols([
              ("manual_event", "Terisi hanya di baris terdekat dengan saat operator menekan '.'; lainnya kosong."),
              ("tension_snapshot", "Nama file foto yang diambil saat skor naik ke ELEVATED/HIGH (di folder "
                                   "snapshots). Kosong di baris lain."),
          ])]

    G += ["", bar, " D. CARA SKOR DIHITUNG", bar,
          _wrap("Skor wajah = jumlah (intensitas AU x bobot) + variasi gaze x "
                f"{GAZE_VARIANCE_WEIGHT:g}. Bobot AU berasal dari data (ukuran efek Cohen's d, dataset RAVDESS):", 1),
          "", _cols(au_rows, 30), "",
          _wrap(f"Jika analisis suara aktif, skor gabungan = wajah x {VISUAL_CHANNEL_WEIGHT:g} + suara x "
                f"{VOICE_CHANNEL_WEIGHT:g}; jika tidak, wajah dihitung penuh.", 1), "",
          _wrap(f"Baseline: {BASELINE_DURATION_SECONDS} detik pertama (target diminta netral). Setelah itu skor "
                "ditampilkan sebagai selisih dari baseline, lalu dihaluskan.", 1), "",
          _wrap(f"Level: ELEVATED jika smoothed >= {ELEVATED_THRESHOLD:.1f}; HIGH jika >= {HIGH_THRESHOLD:.1f}; BELOW "
                f"jika <= -{ELEVATED_THRESHOLD:.1f}; selain itu BASELINE. Angka ini heuristik, belum divalidasi.", 1), "",
          _wrap(f"Lonjakan (episode): satu masuk ke ELEVATED/HIGH = satu lonjakan; baru dianggap selesai setelah "
                f"skor turun di bawah ambang selama {LEVEL_EXIT_COOLDOWN_SECONDS:g} detik. "
                + ("Satu rangkaian ELEVATED/HIGH yang berkelanjutan dihitung SATU lonjakan (level tertinggi)."
                   if MERGE_ELEVATED_HIGH_EPISODES else
                   "Pindah ELEVATED <-> HIGH dihitung lonjakan baru."), 1)]

    G += ["", bar, " E. CARA MEMBACA report.txt", bar,
          _cols([
              ("1. Ringkasan", "Berapa kali & berapa lama ELEVATED/HIGH, skor puncak, porsi waktu per level."),
              ("2. Daftar lonjakan", "Tabel #1, #2, ... lalu rincian: apa yang bergerak, wajah vs suara, nomor "
                                     "frame, foto, penanda operator terdekat."),
              ("3. Catatan operator", "Semua penanda tombol '.' beserta level saat itu."),
              ("4. Konteks percakapan", "Apa yang sedang dibicarakan di sekitar tiap lonjakan (merujuk nomor #). "
                                        "Diisi otomatis setelah transkrip selesai (sekitar 1 menit setelah Stop)."),
          ], 24)]

    G += ["", bar, " F. BATASAN PENTING", bar,
          _item("- Semua ini ringkasan pola heuristik, BUKAN kesimpulan kejujuran/kebohongan target."),
          _item("- Bicara sendiri menggerakkan bibir (AU10/AU15/AU20) dan bisa menaikkan skor; baseline yang "
                "diambil saat diam bisa membuat ucapan biasa tampak seperti kenaikan."),
          _item("- Transkrip otomatis bisa keliru, terutama nama, istilah, dan campuran bahasa."),
          _item("- Gunakan bersama observasi langsung dan konteks wawancara."), bar]
    return "\n".join(G) + "\n"


# ---------------------------------------------------------------------------
# Start button - launches OpenFace + voice analyzer, then lets the existing
# update loop take over exactly as before
# ---------------------------------------------------------------------------
def start_openface(event):
    global openface_process, voice_analyzer, session_started, start_time, baseline_established
    global baseline_mean, last_row_count, log_rows, current_peak_value, current_peak_last_rise_time
    global csv_header, csv_byte_offset
    global level_tracker
    global visual_baseline_mean, voice_baseline_mean
    global session_id, session_dir, current_csv_path, current_log_path, current_snapshot_dir, current_voice_path
    global transcriber, voice_session_start
    global first_video_ts, last_reliable_wall, face_lost, _openface_exit_reported
    global rows_seen, rows_unreliable

    if session_started:
        return  # already running, ignore repeated clicks

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(OPENFACE_OUT_DIR, f"session_{session_id}")
    current_csv_path = os.path.join(session_dir, "session1.csv")
    current_log_path = os.path.join(session_dir, "tension_log.csv")
    current_snapshot_dir = os.path.join(session_dir, "snapshots")
    current_voice_path = os.path.join(session_dir, "voice_recording.wav")
    tension_snapshots.clear()

    os.makedirs(OPENFACE_OUT_DIR, exist_ok=True)
    os.makedirs(session_dir, exist_ok=True)
    # No more stale-CSV cleanup needed here: each session gets its own
    # fresh, uniquely-timestamped session_dir, so current_csv_path can
    # never already exist from a previous run the way the old flat,
    # shared CSV_PATH could.

    try:
        openface_process = subprocess.Popen([
            OPENFACE_EXE, "-device", str(WEBCAM_DEVICE_INDEX),
            "-out_dir", session_dir,
            "-of", "session1",
            *OPENFACE_EXTRA_ARGS,
            # "-cam_width", "150",   # smaller capture/display resolution
            # "-cam_height", "150",  # reduces CPU load, still enough for AU detection
        ])
    except FileNotFoundError:
        ax.set_title(f"ERROR: OpenFace exe not found at {OPENFACE_EXE}")
        fig.canvas.draw_idle()
        return

    # TEMPORARY diagnostics for the "stuck calibrating, OpenFace never
    # shows" issue when voice is on - safe to remove once resolved.
    print(f"DEBUG: OpenFace process started, PID={openface_process.pid}")

    # The mic stream is opened every session: audio is always recorded (the
    # transcript is made from the recording), while "Enable Voice Detection"
    # (voice_enabled) only decides whether the stress-scoring analysis runs.
    if voice_enabled or transcription_enabled:
        voice_analyzer = VoiceAnalyzer()
        try:
            print(f"DEBUG: calling voice_analyzer.start() "
                  f"(analysis_enabled={voice_enabled}, "
                  f"recording=on, "
                  f"transcription={'on' if transcription_enabled else 'off'})...")
            voice_session_start = time.time()
            voice_analyzer.start(
                record_path=current_voice_path,   # always recorded
                analysis_enabled=voice_enabled,
            )
            print("DEBUG: voice_analyzer.start() returned - main thread continuing normally.")
        except Exception as e:
            print(f"Voice analyzer failed to start (continuing with visual-only scoring): {e}")
            voice_analyzer = None
    else:
        voice_analyzer = None
        print("Voice detection, recording, and transcription are all OFF for this session (facial score only).")

    # Transcription rides on the SAME mic stream as voice_analyzer above
    # (see transcriber.py's module docstring) rather than opening a second,
    # competing InputStream - so it can only run if that stream actually
    # came up. If transcription_enabled but voice_analyzer failed to start
    # above, transcription is simply unavailable this session (the panel
    # says so - see update_transcript_panel).
    if transcription_enabled and voice_analyzer is not None:
        transcriber = Transcriber()
        transcriber.start()
        voice_analyzer.set_transcriber(transcriber)
        # transcript_text.set_text("Menunggu...")
        # transcript_text.set_color("dimgray")
    else:
        transcriber = None
        if transcription_enabled:
            print("Transcription enabled but the microphone failed to start - "
                  "transcription unavailable this session.")
        transcript_text.set_text("Transkripsi: mikrofon tidak aktif\n(gagal dibuka atau belum siap).")
        transcript_text.set_color("dimgray")

    # full reset so Start -> Stop -> Start again behaves as a clean new session
    start_time = None
    first_video_ts = None
    rows_seen = 0
    rows_unreliable = 0
    last_reliable_wall = None
    face_lost = False
    _openface_exit_reported = False
    _transcript_entries.clear()   # else a window resize would redraw the PREVIOUS session's transcript
    _autosave["last"] = time.time()
    _autosave["written"] = 0
    big_status_text.set_fontsize(26)
    baseline_established = False
    baseline_mean = None
    baseline_scores.clear()
    tension_history.clear()
    visual_baseline_mean = None
    voice_baseline_mean = None
    visual_baseline_scores.clear()
    voice_baseline_scores.clear()
    visual_tension_history.clear()
    voice_tension_history.clear()
    visual_plot_scores.clear()
    voice_plot_scores.clear()
    gaze_x_history.clear()
    gaze_y_history.clear()
    plot_timestamps.clear()
    plot_scores.clear()
    all_points.clear()
    last_row_count = 0
    csv_header = None
    csv_byte_offset = 0
    current_peak_value = 0.0
    current_peak_last_rise_time = None
    level_tracker = LevelEpisodeTracker()
    manual_events.clear()
    for vline in manual_event_lines:
        vline.remove()
    manual_event_lines.clear()
    log_rows = []
    _clear_session_table()
    conclusion_prefix_text.set_text("Menunggu data timeframe...")
    conclusion_level_text.set_text("")
    conclusion_level_text.set_visible(True)
    conclusion_detail_text.set_text("")
    conclusion_note_text.set_text("")
    session_elevated_text.set_visible(False)
    session_elevated_count_text.set_visible(False)
    session_separator_text.set_visible(False)
    session_high_text.set_visible(False)
    session_high_count_text.set_visible(False)

    session_started = True
    ax.set_title("Live Tension Monitor - PROTOTYPE (calibrating baseline...)")
    start_button.label.set_text("Running...")
    fig.canvas.draw_idle()
    print("OpenFace launched. Calibrating baseline - please stay neutral...")


def stop_openface(event):
    global openface_process, voice_analyzer, session_started, transcriber, _voice_stop_thread, _transcript_thread

    if not session_started:
        return  # nothing running

    session_started = False
    _voice_stop_thread = None

    if openface_process:
        openface_process.terminate()
        openface_process = None
    if voice_analyzer:
        # voice_analyzer.stop() blocks for up to ~CHUNK_SECONDS+1s (a few
        # seconds) to join its audio thread cleanly before writing the
        # .wav - see voice_analyzer.py. Running it in a background thread
        # here instead of calling it directly keeps stop_openface() (a
        # GUI button callback) from freezing the window for that whole
        # duration - the Stop button now returns immediately, and the
        # recording finishes saving a moment later in the background.
        # Daemon thread, no join: nothing past this point in
        # stop_openface() depends on the .wav already being on disk, so
        # there's nothing to wait for here.
        #
        # One edge case this trades in: if Stop is clicked and then Start
        # is clicked again within that same ~3s window, the OLD
        # VoiceAnalyzer's stream is still mid-shutdown while the NEW
        # session's VoiceAnalyzer opens its own stream - two input
        # streams briefly overlapping. Unlike the webcam's exclusive
        # access, Windows audio input generally supports multiple
        # concurrent opens of the same device (WASAPI shared mode), so
        # this is low-risk, but it's not a hard guarantee on every
        # driver - worth knowing if voice features ever glitch
        # specifically on a rapid Stop-then-Start.
        voice_to_stop = voice_analyzer
        _voice_stop_thread = threading.Thread(target=voice_to_stop.stop, daemon=True)
        _voice_stop_thread.start()
        voice_analyzer = None

    # finalize any still-active ELEVATED/HIGH episode so it's included in the report
    end_t = plot_timestamps[-1] if len(plot_timestamps) > 0 else 0.0
    level_tracker.finalize(end_t)
    stats = level_tracker.summary()

    # Hand the transcript off to a background thread to finalize (same
    # non-blocking pattern as the voice recording save above -
    # transcriber.stop() can take a few seconds while Whisper catches up on
    # anything still queued, and Stop should return immediately either
    # way). The post-session context list is built here regardless of
    # whether "Show Live Transcript" was ever turned on during the session
    # - transcription is always on.
    report_ready = threading.Event()   # set once report.txt exists, so the transcript thread can fill in section 4
    transcript_pending = False
    if transcriber is not None:
        transcript_pending = True
        transcriber_to_finalize = transcriber
        transcriber = None
        episodes_snapshot = {lvl: list(level_tracker.episodes[lvl]) for lvl in ("ELEVATED", "HIGH")}
        report_path_candidate = os.path.join(session_dir, "report.txt") if (session_dir and log_rows) else None
        # NOT a daemon thread: if the window is closed before the transcript is ready,
        # the program waits for it (see the end of the file) so report.txt still gets it.
        _transcript_thread = threading.Thread(
            target=_finalize_transcript_and_context,
            args=(transcriber_to_finalize, episodes_snapshot, report_path_candidate,
                  _voice_stop_thread, current_voice_path,
                  (start_time - voice_session_start) if (start_time and voice_session_start) else 0.0,
                  report_ready, number_episodes(episodes_snapshot)[1]),
            daemon=False,
        )
        _transcript_thread.start()
        transcript_text.set_text(
            "Memproses transkrip..."
            if TRANSCRIBER_BACKEND == "assemblyai_async"
            else "Menyelesaikan transkrip...")
        transcript_text.set_color("dimgray")

    # give any snapshot still being copied a moment to finish, so it gets tagged in the log
    _deadline = time.time() + 1.5
    for _t in list(_snapshot_threads):
        _t.join(timeout=max(0.0, _deadline - time.time()))

    # save whatever was captured this session, without needing to close the window
    if log_rows:
        log_df = pd.DataFrame(log_rows)

        # Tag the log row closest to each manual mark, so an expert
        # reviewing the log can see exactly what the raw AU/gaze/score
        # values were at the moment the operator flagged something -
        # anchoring the numbers to a specific real-life moment.
        log_df["manual_event"] = ""
        for ev_elapsed, ev_label in manual_events:
            nearest_idx = (log_df["elapsed_seconds"] - ev_elapsed).abs().idxmin()
            log_df.loc[nearest_idx, "manual_event"] = ev_label

        # Tag the log row closest to each tension-episode snapshot, so an
        # expert can jump straight from a row to the actual photo taken
        # at that rise.
        log_df["tension_snapshot"] = ""
        for snap_elapsed, snap_level, snap_path in tension_snapshots:
            nearest_idx = (log_df["elapsed_seconds"] - snap_elapsed).abs().idxmin()
            log_df.loc[nearest_idx, "tension_snapshot"] = os.path.basename(snap_path)

        log_df.to_csv(current_log_path, index=False)
        print(f"Session log saved to {current_log_path} ({len(log_df)} rows)")
        try:   # the final log is complete, the autosave copy is no longer needed
            os.remove(os.path.join(session_dir, "tension_log.partial.csv"))
        except OSError:
            pass
        ax.set_title(f"Stopped. Log saved: {os.path.basename(session_dir)}/{os.path.basename(current_log_path)}")

        # Shared column guide - generated from the real settings; rewritten only when its text changed.
        guide_text = build_column_guide()
        try:
            guide_current = False
            if os.path.exists(GUIDE_PATH):
                with open(GUIDE_PATH, "r", encoding="utf-8") as f:
                    guide_current = (f.read() == guide_text)
            if guide_current:
                print(f"Column guide already up to date at {GUIDE_PATH}")
            else:
                with open(GUIDE_PATH, "w", encoding="utf-8") as f:
                    f.write(guide_text)
                print(f"Column guide saved to {GUIDE_PATH}")
        except Exception as e:
            print(f"Could not save column guide: {e}")

        # ---- End-of-session report ----
        report_text_full = build_report_text(log_df, stats, end_t, datetime.now(), transcript_pending)
        print(report_text_full)

        report_path = os.path.join(session_dir, "report.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text_full)
            print(f"Session report saved to {report_path}")
        except Exception as e:
            print(f"Could not save report file: {e}")

        # repurpose the Summary panel to show the final session summary (same numbers
        # as section 1 of report.txt), replacing the periodic timeframe summary
        show_session_table(summary_table_lines(stats, compute_session_summary(log_df), compact=True))
    else:
        ax.set_title("Stopped. No data logged this session.")

    start_button.label.set_text("Start OpenFace")
    report_ready.set()   # report.txt is written (or there is none) - the transcript thread may append now
    try:
        fig.canvas.draw_idle()
    except Exception:
        pass   # window already closing
    print("Session stopped.")


def _on_window_close(event):
    """Closing the window mid-session = Stop (saves log + report, starts the transcript)."""
    if session_started:
        try:
            stop_openface(None)
        except Exception as e:
            print(f"Error while stopping on window close: {e}")


fig.canvas.mpl_connect("close_event", _on_window_close)


button_ax = plt.axes([0.476, 0.25, 0.15, 0.075])
start_button = Button(button_ax, "Start OpenFace")
start_button.on_clicked(start_openface)

stop_button_ax = plt.axes([0.66, 0.25, 0.15, 0.075])
stop_button = Button(stop_button_ax, "Stop")
stop_button.on_clicked(stop_openface)


def mark_manual_event(event):
    """Lets the operator flag 'something notable just happened' (e.g. a
    new question was asked) with one click, timestamped against the same
    clock as the tension scores - so a reaction can be traced back to what
    was actually being discussed. Was already in the Tkinter version of
    this project but had never been carried over to this matplotlib line."""
    if not session_started or start_time is None:
        return
    elapsed = time.time() - start_time
    manual_events.append((elapsed, "Ditandai operator"))
    vline = ax.axvline(elapsed, color="#6a1b9a", linestyle=":", linewidth=1.4, alpha=0.9, zorder=4)
    manual_event_lines.append(vline)
    fig.canvas.draw_idle()
    print(f"Event marked at t={elapsed:.1f}s")


# No on-screen button for this - the operator's attention should stay on
# the interview, not the screen. Triggered by pressing "." on the keyboard
# instead (see handle_key_press below).
def handle_key_press(event):
    if event.key in (".", "period", "decimal"):
        mark_manual_event(event)
    elif event.key == " ":
        show_quick_summary()


fig.canvas.mpl_connect("key_press_event", handle_key_press)


# ---------------------------------------------------------------------------
# On-demand toggles for the legend and technical indicator panels - both
# hidden by default so the live operational screen stays uncluttered;
# available with one click for onboarding a new officer or for a technical
# audit of the underlying weights.
# ---------------------------------------------------------------------------
def toggle_legend(event):
    legend_ax.set_visible(not legend_ax.get_visible())
    fig.canvas.draw_idle()

# Disabled for now - never deleted
# def toggle_indicator(event):
#     indicator_ax.set_visible(not indicator_ax.get_visible())
#     fig.canvas.draw_idle()

# Disabled for now - never deleted
# help_button_ax = plt.axes([0.32, 0.185, 0.10, 0.06])
# help_button = Button(help_button_ax, "? Panduan")
# help_button.on_clicked(toggle_legend)

# MARK THIS FOR UNUSED - never deleted, but not currently visible in the live interface. Could be re-enabled if a future audit or onboarding session needs to see the actual AU weights and gaze/voice contributions.
# tech_toggle_ax = plt.axes([0.43, 0.185, 0.15, 0.06])
# tech_toggle_button = Button(tech_toggle_ax, "Detail Teknis")
# tech_toggle_button.on_clicked(toggle_indicator)


# ---------------------------------------------------------------------------
# Voice detection on/off toggle - lets you isolate the facial score alone,
# e.g. to debug whether a trend is coming from the face or the voice
# channel. Only takes effect the NEXT time "Start OpenFace" is clicked -
# toggling mid-session doesn't retroactively start/stop the mic thread.
# ---------------------------------------------------------------------------
def toggle_voice(label):
    global voice_enabled
    voice_enabled = not voice_enabled
    print(f"Voice detection {'enabled' if voice_enabled else 'disabled'} - takes effect on next Start.")


# voice_toggle_ax = plt.axes([0.686, 0.39, 0.36, 0.06])
# Width halved (0.36 -> 0.175) so the new "Always Record Audio" checkbox
# (below) can sit beside it in the same row, rather than stacking - the
# row directly below this one is the full-width legend panel, so there's
# no vertical room to add a second row here.
voice_toggle_ax = plt.axes([0.542, 0.182, 0.36, 0.06])
voice_toggle_ax.set_zorder(10)
voice_toggle_ax.set_frame_on(False)

# Instantiate CheckButtons with enlarged frame and checkmark sizes
# Increase 'sizes' (area in pts^2) to make the square box larger (default is ~100)
voice_toggle = CheckButtons(
    voice_toggle_ax,
    ["Enable Voice Detection"],
    [False],
    frame_props={'sizes': [75], 'linewidth': 1},
    check_props={'sizes': [30], 'linewidth': 1}
)

# Adjust label text position and font size so it doesn't overlap the box
voice_toggle.labels[0].set_x(0.18)
voice_toggle.labels[0].set_fontsize(9)

# Attach callback function
voice_toggle.on_clicked(toggle_voice)


# ---------------------------------------------------------------------------
# Legend panel - explains what each tension state actually means, in plain
# language, for an officer who isn't familiar with the underlying AU/voice
# mechanics. Requested so the tool is interpretable by its actual users,
# not just by whoever built it.
# ---------------------------------------------------------------------------
legend_ax = plt.axes([0.39, 0.065, 0.5703, 0.115])
legend_ax.set_xticks([])
legend_ax.set_yticks([])
legend_ax.set_frame_on(True)
#legend_ax.set_visible(False)  # hidden by default - shown on demand via the "?" button, not permanently on screen
legend_ax.set_visible(True)  # hidden by default - shown on demand via the "?" button, not permanently on screen

legend_ax.text(
    0.5, 0.92,
    "Panduan Level Tension (dibanding baseline/kondisi awal target)",
    fontsize=8, color="black", fontweight="bold", va="top", ha="center", transform=legend_ax.transAxes
)

legend_ax.text(0.01, 0.74, "BASELINE", fontsize=8, color="#2e7d32", va="top", ha="left", transform=legend_ax.transAxes)
legend_ax.text(0.105, 0.74, " = Sesuai kondisi awal - menandai ekspresi normal/netral.", fontsize=8, color="black", va="top", ha="left", transform=legend_ax.transAxes)

legend_ax.text(0.01, 0.56, "ELEVATED", fontsize=8, color="#c77700", va="top", ha="left", transform=legend_ax.transAxes)
legend_ax.text(0.105, 0.56, " = Peningkatan pola tension - bisa karena gugup, berpikir keras, tidak nyaman, dll.", fontsize=8, color="black", va="top", ha="left", transform=legend_ax.transAxes)

legend_ax.text(0.01, 0.38, "HIGH", fontsize=8, color="#b00020", va="top", ha="left", transform=legend_ax.transAxes)
legend_ax.text(0.105, 0.38, " = Peningkatan pola tension yang signifikan - terindikasi tekanan tinggi.", fontsize=8, color="black", va="top", ha="left", transform=legend_ax.transAxes)

# Posisi PENTING diatur pada y = 0.20 untuk memberi jarak dari garis bawah (y = 0.0)
legend_ax.text(0.01, 0.20, "PENTING", fontsize=8, color="#b00020", fontweight="bold", va="top", ha="left", transform=legend_ax.transAxes)
legend_ax.text(0.074, 0.20, "-  Ini adalah ringkasan pola, BUKAN kesimpulan kejujuran/kebohongan target. Gunakan sebagai bahan pertimbangan tambahan, bukan bukti tunggal.", fontsize=8, color="black", va="top", ha="left", transform=legend_ax.transAxes)


# MARK THIS FOR UNUSED - never deleted, but not currently visible in the live interface. Could be re-enabled if a future audit or onboarding session needs to see the actual AU weights and gaze/voice contributions.
# indicator_ax = plt.axes([0.73, 0.065, 0.23, 0.115])
# indicator_ax.set_xticks([])
# indicator_ax.set_yticks([])
# indicator_ax.set_frame_on(True)
# indicator_ax.set_visible(False)  # hidden by default - technical/audit info, not needed during a live interview

# indicator_ax.text(
#     0.5, 0.92,
#     "Indikator yang dihitung",
#     fontsize=8, color="black", fontweight="bold", va="top", ha="center",
#     transform=indicator_ax.transAxes,
# )

# Generated directly from TENSION_WEIGHTS + AU_LABELS (not hand-typed), so
# this panel can never drift out of sync with the actual weights used to
# compute the score. Also includes the gaze and voice/visual channel
# weights, since those contribute to the final score too, not just the AUs.
_au_lines = [
    f"{au}  {weight:.3f}  {AU_LABELS.get(au, '')}"
    for au, weight in sorted(TENSION_WEIGHTS.items(), key=lambda kv: -kv[1])
]
indicator_text = "\n".join(_au_lines) + (
    f"\ngaze var.  {GAZE_VARIANCE_WEIGHT:.2f}  (dlm skor wajah)\n"
    f"---\n"
    f"wajah {VISUAL_CHANNEL_WEIGHT:.0%} : suara {VOICE_CHANNEL_WEIGHT:.0%}  (gabungan)"
)
# indicator_ax.text(
#     0.04, 0.76,
#     indicator_text,
#     fontsize=5.8, color="black", va="top", ha="left",
#     family="monospace", linespacing=1.15,
#     transform=indicator_ax.transAxes,
# )

# ---------------------------------------------------------------------------
# Timeframe selector - controls how often the expression/tension summary
# below recalculates (does not affect the underlying scoring/plot logic)
# ---------------------------------------------------------------------------
def on_timeframe_change(label):
    global selected_timeframe_seconds, last_conclusion_bucket
    selected_timeframe_seconds = TIMEFRAME_OPTIONS[label]
    last_conclusion_bucket = -1  # force a fresh summary on the next full window
    _clear_session_table()
    conclusion_prefix_text.set_text("Menunggu data timeframe...")
    conclusion_level_text.set_text("")
    conclusion_level_text.set_visible(True)
    session_elevated_text.set_visible(False)
    session_elevated_count_text.set_visible(False)
    session_separator_text.set_visible(False)
    session_high_text.set_visible(False)
    session_high_count_text.set_visible(False)
    conclusion_detail_text.set_text("")
    conclusion_note_text.set_text("")
    fig.canvas.draw_idle()


# summary_panel_ax = plt.axes([0.23, 0.065, 0.13, 0.15])
# radio_ax = plt.axes([0.73, 0.75, 0.13, 0.15])
radio_ax = plt.axes([0.04, 0.065, 0.08, 0.115])
radio_ax.set_title("Timeframe", fontsize=9)
timeframe_radio = RadioButtons(radio_ax, list(TIMEFRAME_OPTIONS.keys()), active=list(TIMEFRAME_OPTIONS.keys()).index(DEFAULT_TIMEFRAME_LABEL))
timeframe_radio.on_clicked(on_timeframe_change)


# ---------------------------------------------------------------------------
# Voice metrics visualizer - shows the live rolling voice_score plus its
# individual component features, so the voice channel isn't a silent
# black box running in the background
# ---------------------------------------------------------------------------
voice_panel_ax = plt.axes([0.73, 0.40, 0.13, 0.32])
voice_panel_ax.set_title("Voice Channel", fontsize=9)
voice_panel_ax.set_xticks([])
voice_panel_ax.set_yticks([])
voice_panel_ax.set_visible(False)
for spine in voice_panel_ax.spines.values():
    spine.set_visible(True)

voice_text = voice_panel_ax.text(
    0.05, 0.95, "Waiting for audio...",
    fontsize=8, va="top", ha="left", family="monospace",
    transform=voice_panel_ax.transAxes,
)


def update_voice_panel():
    """Called every update cycle - reads the analyzer's current rolling
    features directly (independent of the video frame rate, since audio
    chunks are computed on their own ~2s cadence)."""
    if not voice_enabled:
        # Stress-analysis scoring is off, but audio is still being recorded
        # (always, for the transcript) - say so rather than a flat "OFF".
        voice_text.set_text("Voice detection:\nOFF (unchecked)\n\nAudio IS being\nrecorded this\nsession.")
        return
    if voice_analyzer is None:
        voice_text.set_text("Voice channel:\nnot running")
        return

    feats = voice_analyzer.get_features()
    score = voice_analyzer.get_score()

    if not feats:
        voice_text.set_text("Voice channel:\nlistening,\nno data yet...")
        return

    text = (
        f"Voice score:\n {score:.3f}\n\n"
        f"Pitch (Hz):\n {feats.get('pitch_mean', 0):.1f}\n\n"
        f"Pitch var:\n {feats.get('pitch_variability', 0):.2f}\n\n"
        f"Jitter:\n {feats.get('jitter', 0):.4f}\n\n"
        f"Pause ratio:\n {feats.get('pause_ratio', 0):.2f}\n\n"
        f"Intensity var:\n {feats.get('intensity_variability', 0):.2f}"
    )
    voice_text.set_text(text)


# ---------------------------------------------------------------------------
# Live tension gauge - a vertical bar showing the CURRENT smoothed score
# (not the history), plus a peak-hold marker that captures the highest
# recent value and slowly decays back down. Meant to be readable at a
# glance, complementing the full line chart rather than replacing it.
# ---------------------------------------------------------------------------
gauge_ax = plt.axes([0.90, 0.22, 0.06, 0.68])
gauge_ax.set_title("Now", fontsize=9)
gauge_ax.set_xticks([])
gauge_ax.set_xlim(0, 1)
gauge_ax.set_ylim(-2, 2)

gauge_zero_line = gauge_ax.axhline(0, color="gray", linestyle="--", linewidth=1)
gauge_bar = gauge_ax.bar([0.5], [0], width=0.6, color="#2e7d32", zorder=5)[0]
gauge_peak_marker = gauge_ax.axhline(0, color="black", linewidth=1.5, xmin=0.15, xmax=0.85, zorder=6)
gauge_value_text = gauge_ax.text(0.5, 0, "", ha="center", va="bottom", fontsize=8, fontweight="bold", transform=gauge_ax.transData)


def level_color(value):
    """Same color scale used by the threshold zones, the gauge, and the
    summary text, kept as a single shared definition so every part of the
    display stays visually consistent."""
    if value >= HIGH_THRESHOLD:
        return "#b00020"
    elif value >= ELEVATED_THRESHOLD:
        return "#c77700"
    elif value <= -ELEVATED_THRESHOLD:
        return "#1565c0"
    return "#2e7d32"


def classify_full(value):
    """Same threshold boundaries as LevelEpisodeTracker.classify, but also
    distinguishes BASELINE from BELOW BASELINE for the big status word,
    rather than lumping everything under HIGH_THRESHOLD as just None."""
    if value >= HIGH_THRESHOLD:
        return "HIGH"
    elif value >= ELEVATED_THRESHOLD:
        return "ELEVATED"
    elif value <= -ELEVATED_THRESHOLD:
        return "BELOW"
    return "BASELINE"


def update_big_status():
    """Updates the large glanceable status word + trend arrow, using the
    CURRENT smoothed score (not a periodic timeframe average) - meant to
    reflect 'right now', readable without needing to parse any numbers."""
    if len(plot_scores) == 0 or face_lost:
        return

    current_value = plot_scores[-1]
    level = classify_full(current_value)

    big_status_text.set_fontsize(26)
    big_status_text.set_text(STATUS_LABELS_ID[level])
    big_status_text.set_color(level_color(current_value))

    # trend: compare current value against the value from ~10 seconds ago
    current_t = plot_timestamps[-1]
    reference_t = current_t - 10.0
    past_values = [s for t, s in zip(plot_timestamps, plot_scores) if t <= reference_t]

    if not past_values:
        trend_arrow_text.set_text("")
        return

    past_value = past_values[-1]
    delta = current_value - past_value

    if delta > 0.15:
        trend_arrow_text.set_text("\u2191 naik")
        trend_arrow_text.set_color("#b00020")
    elif delta < -0.15:
        trend_arrow_text.set_text("\u2193 turun")
        trend_arrow_text.set_color("#1565c0")
    else:
        trend_arrow_text.set_text("\u2192 stabil")
        trend_arrow_text.set_color("dimgray")


def update_gauge():
    """Called every update cycle. Mirrors the main chart's y-limits so the
    gauge and the line chart always read on the same visual scale."""
    global current_peak_value, current_peak_last_rise_time

    if len(plot_scores) == 0:
        return

    current_value = plot_scores[-1]
    now = time.time()

    # peak-hold logic: rise instantly, hold for PEAK_HOLD_SECONDS, then decay
    if current_value > current_peak_value:
        current_peak_value = current_value
        current_peak_last_rise_time = now
    elif current_peak_last_rise_time is not None:
        seconds_since_rise = now - current_peak_last_rise_time
        if seconds_since_rise > PEAK_HOLD_SECONDS:
            current_peak_value = max(current_value, current_peak_value - PEAK_DECAY_PER_SECOND * 0.5)

    # match the gauge's y-range to whatever the main chart is currently showing,
    # so bar height is visually consistent with the line chart's scale
    y_min, y_max = ax.get_ylim()
    gauge_ax.set_ylim(y_min, y_max)

    gauge_bar.set_height(current_value)
    gauge_bar.set_color(level_color(current_value))

    gauge_peak_marker.set_ydata([current_peak_value, current_peak_value])
    gauge_value_text.set_position((0.5, current_value))
    gauge_value_text.set_text(f"{current_value:+.2f}")
    gauge_value_text.set_va("bottom" if current_value >= 0 else "top")

    # live episode counter, shown as a compact title on the gauge - "E" for
    # ELEVATED episodes so far this session, "H" for HIGH episodes so far
    n_elevated = len(level_tracker.episodes["ELEVATED"])
    n_high = len(level_tracker.episodes["HIGH"])
    gauge_ax.set_title(f"Now  (E:{n_elevated} H:{n_high})", fontsize=9)


# ---------------------------------------------------------------------------
# Expression/tension summary text - recalculated once per full timeframe
# window. Positioned in the right-hand column, directly under the Voice
# Channel panel, and color-coded to match the current tension level.
# ---------------------------------------------------------------------------
# summary_panel_ax = plt.axes([0.73, 0.22, 0.13, 0.15])
summary_panel_ax = plt.axes([0.14, 0.065, 0.23, 0.115])
summary_panel_ax.set_title("Summary", fontsize=9)
summary_panel_ax.set_xticks([])
summary_panel_ax.set_yticks([])

conclusion_prefix_text = summary_panel_ax.text(
    0.03, 0.95,
    "Menunggu data timeframe...",
    fontsize=8, wrap=True, va="top", ha="left",
    transform=summary_panel_ax.transAxes,
)
conclusion_level_text = summary_panel_ax.text(
    0.03, 0.68,
    "",
    fontsize=8, fontweight="bold", wrap=True, va="top", ha="left",
    transform=summary_panel_ax.transAxes,
)
session_elevated_text = summary_panel_ax.text(
    0.03, 0.68, "", fontsize=8, fontweight="bold", color="#c77700",
    va="top", ha="left", transform=summary_panel_ax.transAxes, visible=False,
)
session_elevated_count_text = summary_panel_ax.text(
    0.21, 0.68, "", fontsize=8, fontweight="bold", color="black",
    va="top", ha="left", transform=summary_panel_ax.transAxes, visible=False,
)
session_separator_text = summary_panel_ax.text(
    0.29, 0.68, "|", fontsize=8, fontweight="bold", color="black",
    va="top", ha="left", transform=summary_panel_ax.transAxes, visible=False,
)
session_high_text = summary_panel_ax.text(
    0.34, 0.68, "", fontsize=8, fontweight="bold", color="#b00020",
    va="top", ha="left", transform=summary_panel_ax.transAxes, visible=False,
)
session_high_count_text = summary_panel_ax.text(
    0.45, 0.68, "", fontsize=8, fontweight="bold", color="black",
    va="top", ha="left", transform=summary_panel_ax.transAxes, visible=False,
)
conclusion_detail_text = summary_panel_ax.text(
    0.03, 0.50,
    "",
    fontsize=7, wrap=True, va="top", ha="left",
    transform=summary_panel_ax.transAxes,
)
conclusion_note_text = summary_panel_ax.text(
    0.03, 0.28,
    "",
    fontsize=6.5, style="italic", wrap=True, va="top", ha="left", color="dimgray",
    transform=summary_panel_ax.transAxes,
)

# ---------------------------------------------------------------------------
# End-of-session table in the Summary panel (same block as section 1 of
# report.txt). Monospace so the columns line up; the font shrinks until the
# whole table fits the panel (so it works at any window size) and is re-fitted
# when the window is resized.
# ---------------------------------------------------------------------------
SUMMARY_TABLE_FONT_SIZES = (12.0, 11.0, 10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0)   # biggest that fits is used
SUMMARY_TABLE_SIDE_MARGIN = 0.035
SUMMARY_TABLE_LINE_SPACING = 1.28   # line height = font size x this
SUMMARY_TABLE_COLORS = {"header": "dimgray", "ELEVATED": "#c77700", "HIGH": "#b00020", "text": "black"}
_summary_table_artists = []
_summary_table_rows = None


def _clear_session_table():
    """Removes the end-of-session table and brings the normal periodic summary texts back."""
    global _summary_table_rows
    for t in _summary_table_artists:
        t.remove()
    _summary_table_artists.clear()
    _summary_table_rows = None
    conclusion_prefix_text.set_visible(True)
    conclusion_detail_text.set_visible(True)
    conclusion_note_text.set_visible(True)


def _render_session_table():
    for t in _summary_table_artists:
        t.remove()
    _summary_table_artists.clear()
    if not _summary_table_rows:
        return
    renderer = fig.canvas.get_renderer()
    bbox = summary_panel_ax.get_window_extent()
    max_w = bbox.width * (1 - 2 * SUMMARY_TABLE_SIDE_MARGIN)
    avail_h = bbox.height * 0.94
    fp = conclusion_prefix_text.get_fontproperties().copy()
    fp.set_family("monospace")

    chosen = SUMMARY_TABLE_FONT_SIZES[-1]
    for fs in SUMMARY_TABLE_FONT_SIZES:
        fp.set_size(fs)
        widest = max(renderer.get_text_width_height_descent(txt, fp, False)[0] for txt, _k in _summary_table_rows)
        line_h = fs * SUMMARY_TABLE_LINE_SPACING * fig.dpi / 72.0
        if widest <= max_w and line_h * len(_summary_table_rows) <= avail_h:
            chosen = fs
            break
    line_h = chosen * SUMMARY_TABLE_LINE_SPACING * fig.dpi / 72.0
    for i, (txt, kind) in enumerate(_summary_table_rows):
        y = 0.97 - (i * line_h) / bbox.height
        _summary_table_artists.append(summary_panel_ax.text(
            SUMMARY_TABLE_SIDE_MARGIN, y, txt, fontsize=chosen, family="monospace", va="top", ha="left",
            fontweight="bold" if kind in ("header", "ELEVATED", "HIGH") else "normal",
            color=SUMMARY_TABLE_COLORS.get(kind, "black"), transform=summary_panel_ax.transAxes, clip_on=True,
        ))


def show_session_table(rows):
    """rows = [(text, kind), ...] from summary_table_lines()."""
    global _summary_table_rows
    _summary_table_rows = [r for r in rows if r[0]]
    for t in (conclusion_prefix_text, conclusion_level_text, conclusion_detail_text, conclusion_note_text,
              session_elevated_text, session_elevated_count_text, session_separator_text,
              session_high_text, session_high_count_text):
        t.set_visible(False)
    _render_session_table()


fig.canvas.mpl_connect("resize_event", lambda _e: _render_session_table() if _summary_table_rows else None)

disclaimer_note = fig.text(
    0.73, 0.005,
    "Heuristic pattern summary only - not a determination of nervousness, honesty, or intent.",
    fontsize=7, style="italic", color="dimgray", va="bottom", ha="left",
)


def describe_window(window_vals):
    """Shared by both the periodic (Timeframe radio) summary and the
    on-demand (spacebar) summary, so the two always agree on what counts
    as HIGH/ELEVATED/etc and never drift out of sync with each other.
    Returns (level, note, variability_note, mean_val)."""
    mean_val = float(np.mean(window_vals))
    std_val = float(np.std(window_vals))

    if mean_val >= HIGH_THRESHOLD:
        level = "HIGH"
        note = "Peningkatan signifikan dari baseline."
    elif mean_val >= ELEVATED_THRESHOLD:
        level = "ELEVATED"
        note = "Ada peningkatan dari baseline.\nBelum tentu berarti masalah\nbisa krn gugup wajar, dll."
    elif mean_val <= -ELEVATED_THRESHOLD:
        level = "BELOW BASELINE"
        note = "Lebih tenang dari kondisi awal target."
    else:
        level = "BASELINE"
        note = "Sesuai kondisi awal target,\ntidak ada indikasi perubahan."

    if std_val >= HIGH_VARIABILITY_THRESHOLD:
        variability_note = "pola berubah-ubah"
    else:
        variability_note = "pola stabil"

    return level, note, variability_note, mean_val


def update_conclusion():
    """Recomputes and displays the plain-language summary once the current
    timeframe window has fully elapsed. Heuristic thresholds, same caveat
    as the rest of this project - a pattern summary, not a diagnosis.

    Uses the UNCAPPED all_points history (not the capped plot_timestamps/
    plot_scores used for chart rendering), so the summary genuinely
    reflects the entire selected window - not just however many seconds
    happen to still fit in the 200-point chart display buffer."""
    global last_conclusion_bucket

    if len(all_points) == 0:
        return

    max_t = all_points[-1][0]
    current_bucket = int(max_t // selected_timeframe_seconds)

    if current_bucket == last_conclusion_bucket:
        return
    if max_t < selected_timeframe_seconds:
        return  # not enough data yet for a full window

    # summarise the window that has just FINISHED (the previous bucket). The old
    # code used the bucket that had only just started, i.e. ~0.5 s of data.
    window_start = (current_bucket - 1) * selected_timeframe_seconds
    window_vals = [y for t, y in all_points if window_start <= t < window_start + selected_timeframe_seconds]
    if not window_vals:
        return

    level, note, variability_note, mean_val = describe_window(window_vals)

    conclusion_prefix_text.set_text(f"[{selected_timeframe_seconds} detik terakhir]")

    conclusion_level_text.set_text(level)
    conclusion_level_text.set_color(level_color(mean_val))

    # plain-language description instead of raw avg/var numbers - the exact
    # numeric values are still saved in the CSV log for later technical
    # review, just not shown on the live operational screen
    conclusion_detail_text.set_text(variability_note)

    conclusion_note_text.set_text(note)

    last_conclusion_bucket = current_bucket


# ---------------------------------------------------------------------------
# On-demand "last N seconds" summary - triggered by pressing space, as an
# instant snapshot using whichever window the Timeframe radio (30s/1min/2min)
# is currently set to. Where update_conclusion() waits for a full bucket of
# the selected timeframe to close (aligned to session start) before
# refreshing, this always looks at a rolling window of that same length
# ending right now, so an officer can ask "how has the last N seconds
# looked" at any moment without waiting for a bucket boundary.
# ---------------------------------------------------------------------------
def show_quick_summary():
    if not session_started or len(all_points) == 0:
        return

    max_t = all_points[-1][0]
    window_start = max(0.0, max_t - selected_timeframe_seconds)
    window_vals = [y for t, y in all_points if window_start <= t <= max_t]
    if not window_vals:
        return

    level, note, variability_note, mean_val = describe_window(window_vals)

    conclusion_prefix_text.set_text(f"[{window_start:.0f}s - {max_t:.0f}s]")
    conclusion_level_text.set_text(level)
    conclusion_level_text.set_color(level_color(mean_val))
    conclusion_detail_text.set_text(variability_note)
    conclusion_note_text.set_text(note)
    fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Update loop - same structure as the original update_plot, gated behind
# session_started so nothing runs until the button is clicked, plus gaze
# and voice scoring layered into the existing compute step
# ---------------------------------------------------------------------------
_LOG_FIELDS = [
    "timestamp", "elapsed_seconds", "frame", "openface_success", "openface_confidence",
    "AU04_brow_lowerer", "AU09_nose_wrinkler", "AU10_upper_lip_raiser", "AU20_lip_stretcher",
    "AU15_lip_corner_depressor", "AU05_upper_lid_raiser", "dominant_signal",
    "gaze_angle_x", "gaze_angle_y", "au_score", "gaze_variance_score", "voice_score", "raw_score",
    "baseline", "relative_score", "smoothed_score", "visual_smoothed", "voice_smoothed", "level",
    "voice_pitch_mean_hz", "voice_pitch_variability", "voice_jitter", "voice_pause_ratio",
    "voice_intensity_variability",
]


def autosave_log():
    """Appends the rows logged since the last autosave to tension_log.partial.csv (local only)."""
    if AUTOSAVE_LOG_SECONDS <= 0 or not session_started or not session_dir:
        return
    now = time.time()
    if now - _autosave["last"] < AUTOSAVE_LOG_SECONDS:
        return
    _autosave["last"] = now
    new_rows = log_rows[_autosave["written"]:]
    if not new_rows:
        return
    path = os.path.join(session_dir, "tension_log.partial.csv")
    try:
        write_header = _autosave["written"] == 0 or not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_LOG_FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerows(new_rows)
        _autosave["written"] += len(new_rows)
    except Exception as e:
        print(f"Autosave failed (will retry): {e}")


def _update_face_lost_indicator():
    """If no reliable face frame arrives for FACE_LOST_SECONDS, say so, instead of leaving the
    last status (e.g. 'STABIL') on screen as if it were still current."""
    global face_lost
    if last_reliable_wall is None:
        return
    if time.time() - last_reliable_wall > FACE_LOST_SECONDS:
        face_lost = True
        big_status_text.set_fontsize(16)
        big_status_text.set_text("WAJAH HILANG")
        big_status_text.set_color("dimgray")
        trend_arrow_text.set_text("")
    elif face_lost:
        face_lost = False   # update_big_status repaints the real status


def _check_openface_alive():
    global _openface_exit_reported
    if openface_process is None or _openface_exit_reported:
        return
    code = openface_process.poll()
    if code is not None:
        _openface_exit_reported = True
        msg = f"ERROR: OpenFace berhenti (kode {code}) - cek kamera, lalu Stop & Start lagi"
        print(msg)
        ax.set_title(msg)


def update_plot(frame_num):
    global last_row_count, baseline_mean, baseline_established, start_time
    global csv_header, csv_byte_offset
    global visual_baseline_mean, voice_baseline_mean
    global first_video_ts, last_reliable_wall
    global rows_seen, rows_unreliable

    if not session_started:
        return line,

    _check_openface_alive()
    _update_face_lost_indicator()
    autosave_log()

    try:
        if not current_csv_path or not os.path.exists(current_csv_path):
            return line,

        with open(current_csv_path, "r", newline="") as f:
            if csv_header is None:
                csv_header = f.readline()
                csv_byte_offset = f.tell()

            f.seek(csv_byte_offset)
            new_lines = []
            while True:
                pos_before = f.tell()
                line_str = f.readline()
                if not line_str or not line_str.endswith("\n"):
                    # either EOF, or OpenFace is still mid-write on this line -
                    # rewind to before it so we re-read it whole next cycle
                    csv_byte_offset = pos_before
                    break
                new_lines.append(line_str)

        if not new_lines:
            return line,  # nothing new since the last poll - skip re-parsing entirely

        chunk_csv = csv_header + "".join(new_lines)
        new_rows = pd.read_csv(StringIO(chunk_csv), low_memory=False)
        new_rows.columns = new_rows.columns.str.strip()

        for au in TENSION_WEIGHTS.keys():
            if au in new_rows.columns:
                new_rows[au] = pd.to_numeric(new_rows[au], errors='coerce')
        for col in ("gaze_angle_x", "gaze_angle_y", "confidence", "success", "timestamp"):
            if col in new_rows.columns:
                new_rows[col] = pd.to_numeric(new_rows[col], errors='coerce')
    except Exception as e:
        print(f"Read error: {e}")
        return line,

    last_row_count += len(new_rows)  # kept for stats/logging only, no longer used for slicing

    skipped_low_confidence = 0
    for _, row in new_rows.iterrows():
        rows_seen += 1
        if not row_is_reliable(row):
            skipped_low_confidence += 1
            rows_unreliable += 1
            continue

        au_score = compute_au_score(row)
        if pd.isna(au_score):
            continue  # skip rows with no valid AU data (e.g. face not detected)

        last_reliable_wall = time.time()
        _fr = row.get("frame", np.nan)
        frame_no = int(_fr) if pd.notna(_fr) else -1

        gx, gy = row.get("gaze_angle_x", np.nan), row.get("gaze_angle_y", np.nan)
        if pd.notna(gx):
            gaze_x_history.append(gx)
        if pd.notna(gy):
            gaze_y_history.append(gy)
        gaze_score = compute_gaze_variance_score()
        visual_score = au_score + (gaze_score * GAZE_VARIANCE_WEIGHT)

        # Gate on voice_enabled, NOT just "voice_analyzer exists": since
        # "Always Record Audio" was added, voice_analyzer can now be a
        # real object (recording-only, analysis_enabled=False inside it)
        # even while voice detection/analysis is OFF. Checking only
        # `if voice_analyzer:` here would then apply
        # VISUAL_CHANNEL_WEIGHT/VOICE_CHANNEL_WEIGHT (0.75/0.25) instead
        # of giving the facial channel full weight - voice_score would
        # still read 0.0 (its scoring loop never runs), so raw_score
        # would be silently cut to 75% of visual_score, exactly the
        # "face line and combined line no longer match when voice is
        # off" gap this fixes.
        if voice_enabled and voice_analyzer:
            voice_score = voice_analyzer.get_score()
            active_visual_weight, active_voice_weight = VISUAL_CHANNEL_WEIGHT, VOICE_CHANNEL_WEIGHT
        else:
            # voice detection disabled (or failed to start) - give the
            # facial channel the full weight instead of silently losing
            # VOICE_CHANNEL_WEIGHT's share of the score (which would
            # otherwise shrink every reading)
            voice_score = 0.0
            active_visual_weight, active_voice_weight = 1.0, 0.0

        raw_score = (visual_score * active_visual_weight) + (voice_score * active_voice_weight)

        now = time.time()
        ts = row.get("timestamp", np.nan)
        if USE_VIDEO_TIMESTAMPS and pd.notna(ts):
            # OpenFace's own clock for this frame - every row gets its true time, not
            # the time of the batch it happened to be read in
            if first_video_ts is None:
                first_video_ts = float(ts)
            if start_time is None:
                start_time = now
            elapsed = float(ts) - first_video_ts
        else:
            if start_time is None:
                start_time = now
            elapsed = now - start_time

        if not baseline_established:
            baseline_scores.append(raw_score)
            visual_baseline_scores.append(visual_score)
            voice_baseline_scores.append(voice_score)
            if elapsed >= BASELINE_DURATION_SECONDS:
                baseline_mean = np.mean(baseline_scores)
                visual_baseline_mean = np.mean(visual_baseline_scores)
                voice_baseline_mean = np.mean(voice_baseline_scores)
                baseline_established = True
                ax.set_title("Live Tension Monitor - PROTOTYPE")
                print(f"Baseline established: {baseline_mean:.3f} (from {len(baseline_scores)} frames)")
                # One reference photo of the "normal/neutral" state calibration
                # was based on - same mechanism and naming convention as the
                # ELEVATED/HIGH episode snapshots below, just triggered once
                # here instead of on a level transition.
                save_tension_snapshot("BASELINE", elapsed, frame_no)
            continue

        relative_score = raw_score - baseline_mean
        tension_history.append(relative_score)
        smoothed = sum(tension_history) / len(tension_history)

        was_active_level = level_tracker.active_level
        level_tracker.update(elapsed, smoothed)
        if level_tracker.active_level is not None and level_tracker.active_level != was_active_level:
            save_tension_snapshot(level_tracker.active_level, elapsed, frame_no)

        # per-channel relative + smoothed scores, purely for the secondary
        # display lines - doesn't feed back into the combined score at all
        visual_relative = visual_score - visual_baseline_mean
        voice_relative = voice_score - voice_baseline_mean
        visual_tension_history.append(visual_relative)
        voice_tension_history.append(voice_relative)
        visual_smoothed = sum(visual_tension_history) / len(visual_tension_history)
        voice_smoothed = sum(voice_tension_history) / len(voice_tension_history)

        plot_timestamps.append(elapsed)
        plot_scores.append(smoothed)
        all_points.append((elapsed, smoothed))
        visual_plot_scores.append(visual_smoothed)
        voice_plot_scores.append(voice_smoothed)

        log_row = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "frame": frame_no,

            # OpenFace's own reliability signals - lets an expert judge
            # whether a given row's data should even be trusted before
            # reading anything into a score change
            "openface_success": _safe_round(row.get("success", np.nan), 0),
            "openface_confidence": _safe_round(row.get("confidence", np.nan), 3),

            # Raw Action Unit intensities behind au_score - the single
            # most important addition for expert validation: without
            # these, au_score is an opaque number with no way to check
            # WHICH facial movement actually drove it
            "AU04_brow_lowerer": _safe_round(row.get("AU04_r", np.nan), 3),
            "AU09_nose_wrinkler": _safe_round(row.get("AU09_r", np.nan), 3),
            "AU10_upper_lip_raiser": _safe_round(row.get("AU10_r", np.nan), 3),
            "AU20_lip_stretcher": _safe_round(row.get("AU20_r", np.nan), 3),
            "AU15_lip_corner_depressor": _safe_round(row.get("AU15_r", np.nan), 3),
            "AU05_upper_lid_raiser": _safe_round(row.get("AU05_r", np.nan), 3),
            "dominant_signal": _describe_dominant_signal(row),

            # Raw gaze angles behind gaze_variance_score
            "gaze_angle_x": _safe_round(gx),
            "gaze_angle_y": _safe_round(gy),

            "au_score": round(au_score, 4),
            "gaze_variance_score": round(gaze_score, 4),
            "voice_score": round(voice_score, 4),
            "raw_score": round(raw_score, 4),
            "baseline": round(baseline_mean, 4),
            "relative_score": round(relative_score, 4),
            "smoothed_score": round(smoothed, 4),
            "visual_smoothed": round(visual_smoothed, 4),
            "voice_smoothed": round(voice_smoothed, 4),

            # Classified level using the same thresholds already shown
            # on-screen (BASELINE/ELEVATED/HIGH/BELOW) - so an expert
            # doesn't have to manually re-apply the thresholds themselves
            "level": classify_full(smoothed),
        }

        # Individual acoustic features behind voice_score, when voice
        # scoring is active - same transparency principle as the AUs:
        # a combined voice_score alone can't be validated on its own.
        # Gated on voice_enabled (not just voice_analyzer existing) for
        # the same reason as the weighting above - a recording-only
        # session (analysis off) never runs _analyze_chunk, so these
        # would otherwise log as empty/stale NaN columns every row.
        if voice_enabled and voice_analyzer:
            voice_features = voice_analyzer.get_features()
            log_row["voice_pitch_mean_hz"] = _safe_round(voice_features.get("pitch_mean", np.nan), 1)
            log_row["voice_pitch_variability"] = _safe_round(voice_features.get("pitch_variability", np.nan))
            log_row["voice_jitter"] = _safe_round(voice_features.get("jitter", np.nan))
            log_row["voice_pause_ratio"] = _safe_round(voice_features.get("pause_ratio", np.nan))
            log_row["voice_intensity_variability"] = _safe_round(voice_features.get("intensity_variability", np.nan))

        log_rows.append(log_row)

    if len(plot_timestamps) > 1:
        line.set_data(list(plot_timestamps), list(plot_scores))
        visual_line.set_data(list(plot_timestamps), list(visual_plot_scores))
        voice_line.set_data(list(plot_timestamps), list(voice_plot_scores))
        ax.set_xlim(max(0, plot_timestamps[0]), plot_timestamps[-1] + 1)

        all_visible_scores = list(plot_scores) + list(visual_plot_scores) + list(voice_plot_scores)
        y_min = min(all_visible_scores) - 0.2
        y_max = max(all_visible_scores) + 0.2
        if not (np.isnan(y_min) or np.isnan(y_max)):
            ax.set_ylim(y_min, y_max)

        # redraw the shaded zones so they always cover the current y-range
        for patch in list(ax.patches):
            patch.remove()
        draw_threshold_zones()
        ax.legend(loc="upper left", fontsize=7, framealpha=0.7)

        update_conclusion()
        update_gauge()
        update_big_status()

    update_voice_panel()
    update_transcript_panel()

    return line,


ani = animation.FuncAnimation(fig, update_plot, interval=500, cache_frame_data=False)

print("Click 'Start OpenFace' in the plot window to begin capture.")
plt.show()

# ---------------------------------------------------------------------------
# Cleanup on window close - only relevant if the user closed the window
# without clicking Stop first (Stop already handles termination + saving)
# ---------------------------------------------------------------------------
if openface_process:
    openface_process.terminate()
if voice_analyzer:
    # Deliberately NOT backgrounded like the Stop-button path above: the
    # script is about to exit right after this block, with no event loop
    # left running to keep a daemon thread alive - backgrounding this
    # call would risk the process exiting before the .wav finishes
    # writing. There's also no GUI left to freeze at this point (plt.show()
    # has already returned), so blocking here costs nothing visible.
    voice_analyzer.stop()

if _transcript_thread is not None and _transcript_thread.is_alive():
    print("Menunggu transkrip selesai - jangan tutup jendela terminal...")
    _transcript_thread.join()
    print("Transkrip selesai.")

if session_started and log_rows:
    log_df = pd.DataFrame(log_rows)
    # session_dir/current_log_path are normally already set (start_openface
    # sets them before session_started goes True) - this fallback only
    # matters if something unexpected left them unset.
    if session_dir is None:
        fallback_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(OPENFACE_OUT_DIR, f"session_{fallback_id}")
        os.makedirs(session_dir, exist_ok=True)
    fallback_log_path = current_log_path or os.path.join(session_dir, "tension_log.csv")
    log_df.to_csv(fallback_log_path, index=False)
    print(f"Session log saved to {fallback_log_path} ({len(log_df)} rows)")
elif not session_started:
    print("Session was already stopped and saved before the window closed.")
else:
    print("No data logged this session.")
