import subprocess
import os
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

# ---------------------------------------------------------------------------
# Original config - unchanged
# ---------------------------------------------------------------------------
CSV_PATH = r"C:\OpenFace\live_output\session1.csv"
LOG_PATH = r"C:\OpenFace\live_output\tension_log.csv"

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

WINDOW_SIZE = 10
BASELINE_DURATION_SECONDS = 10

# ---------------------------------------------------------------------------
# New config - OpenFace launch + gaze + voice + threshold zones
# ---------------------------------------------------------------------------
OPENFACE_EXE = r"C:\OpenFace\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
OPENFACE_OUT_DIR = r"C:\OpenFace\live_output"

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

last_row_count = 0
csv_header = None       # cached header line, set once per session
csv_byte_offset = 0     # how far into the CSV we've already read - avoids re-parsing the whole file every poll
log_rows = []

openface_process = None
voice_analyzer = None
session_started = False  # gates the update loop until the Start button is pressed
voice_enabled = True     # toggled via the "Enable Voice Detection" checkbox - takes effect on next Start

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


# ---------------------------------------------------------------------------
# Plot setup - same base line chart as the original, with threshold zones added
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.39, right=0.70, top=0.90)  # move the main panel upward while preserving its height

line, = ax.plot([], [], color='black', linewidth=1.6, zorder=5, label="Combined")
visual_line, = ax.plot([], [], color='#1565c0', linewidth=1.0, alpha=0.6, zorder=4, label="Face only")
voice_line, = ax.plot([], [], color='#2e7d32', linewidth=1.0, alpha=0.6, zorder=4, label="Voice only")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Tension Score (relative to baseline)")
ax.set_title("Live Tension Monitor - PROTOTYPE")
ax.axhline(0, color='gray', linestyle='--', linewidth=1)


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
# Start button - launches OpenFace + voice analyzer, then lets the existing
# update loop take over exactly as before
# ---------------------------------------------------------------------------
def start_openface(event):
    global openface_process, voice_analyzer, session_started, start_time, baseline_established
    global baseline_mean, last_row_count, log_rows, current_peak_value, current_peak_last_rise_time
    global csv_header, csv_byte_offset
    global level_tracker
    global visual_baseline_mean, voice_baseline_mean

    if session_started:
        return  # already running, ignore repeated clicks

    os.makedirs(OPENFACE_OUT_DIR, exist_ok=True)

    # remove any stale CSV from a previous run so the baseline calibration
    # window reflects genuinely fresh data, not old leftover rows
    try:
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
    except Exception as e:
        print(f"Could not remove old CSV (continuing anyway): {e}")

    try:
        openface_process = subprocess.Popen([
            OPENFACE_EXE, "-device", "0",
            "-out_dir", OPENFACE_OUT_DIR,
            "-of", "session1",
        ])
    except FileNotFoundError:
        ax.set_title(f"ERROR: OpenFace exe not found at {OPENFACE_EXE}")
        fig.canvas.draw_idle()
        return

    if voice_enabled:
        voice_analyzer = VoiceAnalyzer()
        try:
            voice_analyzer.start()
        except Exception as e:
            print(f"Voice analyzer failed to start (continuing with visual-only scoring): {e}")
            voice_analyzer = None
    else:
        voice_analyzer = None
        print("Voice detection is OFF for this session (facial score only).")

    # full reset so Start -> Stop -> Start again behaves as a clean new session
    start_time = None
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
    last_row_count = 0
    csv_header = None
    csv_byte_offset = 0
    current_peak_value = 0.0
    current_peak_last_rise_time = None
    level_tracker = LevelEpisodeTracker()
    log_rows = []
    conclusion_level_text.set_visible(True)
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
    global openface_process, voice_analyzer, session_started

    if not session_started:
        return  # nothing running

    session_started = False

    if openface_process:
        openface_process.terminate()
        openface_process = None
    if voice_analyzer:
        voice_analyzer.stop()
        voice_analyzer = None

    # finalize any still-active ELEVATED/HIGH episode so it's included in the report
    end_t = plot_timestamps[-1] if len(plot_timestamps) > 0 else 0.0
    level_tracker.finalize(end_t)
    stats = level_tracker.summary()

    # save whatever was captured this session, without needing to close the window
    if log_rows:
        log_df = pd.DataFrame(log_rows)
        stamped_path = LOG_PATH.replace(".csv", f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        log_df.to_csv(stamped_path, index=False)
        print(f"Session log saved to {stamped_path} ({len(log_df)} rows)")
        ax.set_title(f"Stopped. Log saved: {os.path.basename(stamped_path)}")

        # ---- End-of-session episode report ----
        report_lines = [
            f"RINGKASAN SESI (total durasi: {end_t:.0f} detik)",
            f"ELEVATED: {stats['ELEVATED']['count']}x kejadian, total {stats['ELEVATED']['total_duration']:.0f}s, "
            f"terlama {stats['ELEVATED']['longest']:.0f}s",
            f"HIGH: {stats['HIGH']['count']}x kejadian, total {stats['HIGH']['total_duration']:.0f}s, "
            f"terlama {stats['HIGH']['longest']:.0f}s",
            "",
            "Catatan: hitungan ini menunjukkan berapa kali pola tension naik ke level tersebut "
            "secara terpisah (bukan jumlah pembacaan individual). Ini adalah ringkasan pola, "
            "BUKAN kesimpulan kejujuran/kebohongan target. Gunakan sebagai salah satu bahan "
            "pertimbangan bersama observasi langsung dan konteks wawancara.",
        ]
        report_text_full = "\n".join(report_lines)
        print(report_text_full)

        report_path = stamped_path.replace(".csv", "_report.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text_full)
            print(f"Session report saved to {report_path}")
        except Exception as e:
            print(f"Could not save report file: {e}")

        # repurpose the Summary panel to show the final session totals,
        # replacing the periodic timeframe summary now that the session is over
        conclusion_prefix_text.set_text("RINGKASAN SESI:")
        conclusion_level_text.set_visible(False)
        session_elevated_text.set_text("Elevated")
        session_elevated_count_text.set_text(f" {stats['ELEVATED']['count']}x")
        session_high_text.set_text("High")
        session_high_count_text.set_text(f" {stats['HIGH']['count']}x")
        session_elevated_text.set_visible(True)
        session_elevated_count_text.set_visible(True)
        session_separator_text.set_visible(True)
        session_high_text.set_visible(True)
        session_high_count_text.set_visible(True)
        conclusion_detail_text.set_text(
            f"Elevated: {stats['ELEVATED']['total_duration']:.0f}s total, "
            f"terlama {stats['ELEVATED']['longest']:.0f}s\n"
            f"High: {stats['HIGH']['total_duration']:.0f}s total, "
            f"terlama {stats['HIGH']['longest']:.0f}s"
        )
        conclusion_note_text.set_text(
            "Bukan kesimpulan kejujuran\npertimbangkan dengan konteks wawancara."
        )
    else:
        ax.set_title("Stopped. No data logged this session.")

    start_button.label.set_text("Start OpenFace")
    fig.canvas.draw_idle()
    print("Session stopped.")


button_ax = plt.axes([0.32, 0.25, 0.17, 0.075])
start_button = Button(button_ax, "Start OpenFace")
start_button.on_clicked(start_openface)

stop_button_ax = plt.axes([0.51, 0.25, 0.17, 0.075])
stop_button = Button(stop_button_ax, "Stop")
stop_button.on_clicked(stop_openface)


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


# voice_toggle_ax = plt.axes([0.32, 0.09, 0.36, 0.06])
# voice_toggle_ax = plt.axes([0.38, 0.12, 0.36, 0.06])
# voice_toggle_ax = plt.axes([0.686, 0.32, 0.36, 0.06])
# voice_toggle_ax.set_zorder(10)
# voice_toggle_ax.set_frame_on(False)
# voice_toggle = CheckButtons(voice_toggle_ax, ["Enable Voice Detection"], [True])
# voice_toggle.labels[0].set_x(0.19)
# voice_toggle.on_clicked(toggle_voice)
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

# Create axes for the checkbox
voice_toggle_ax = plt.axes([0.686, 0.39, 0.36, 0.06])
voice_toggle_ax.set_zorder(10)
voice_toggle_ax.set_frame_on(False)

# Instantiate CheckButtons with enlarged frame and checkmark sizes
# Increase 'sizes' (area in pts^2) to make the square box larger (default is ~100)
voice_toggle = CheckButtons(
    voice_toggle_ax, 
    ["Enable Voice Detection"], 
    [True],
    frame_props={'sizes': [75], 'linewidth': 1},
    check_props={'sizes': [30], 'linewidth': 1}
)

# Adjust label text position and font size so it doesn't overlap the box
voice_toggle.labels[0].set_x(0.18)
voice_toggle.labels[0].set_fontsize(8)

# Attach callback function
voice_toggle.on_clicked(toggle_voice)


# ---------------------------------------------------------------------------
# Legend panel - explains what each tension state actually means, in plain
# language, for an officer who isn't familiar with the underlying AU/voice
# mechanics. Requested so the tool is interpretable by its actual users,
# not just by whoever built it.
# ---------------------------------------------------------------------------
# legend_ax = plt.axes([0.125, 0.036, 0.5741, 0.09])
# legend_ax.set_xticks([])
# legend_ax.set_yticks([])
# legend_ax.set_frame_on(True)

# legend_text = (
#     "Panduan Level Tension (dibanding baseline/kondisi awal target sendiri):\n"
#     "BASELINE (hijau) = sesuai kondisi awal, tidak ada indikasi perubahan\n"
#     "ELEVATED (kuning) = ada peningkatan pola tension - bisa krn gugup, berpikir keras, tidak nyaman, dll\n"
#     "HIGH (merah) = peningkatan pola tension yang signifikan\n"
#     "PENTING: BUKAN alat deteksi kebohongan. Gunakan sebagai bahan pertimbangan tambahan, bukan bukti tunggal."
# )
# legend_display = legend_ax.text(
#     0.01, 0.95, legend_text, fontsize=8, va="top", ha="left",
#     transform=legend_ax.transAxes, linespacing=1.4,
# )
# Perbesar tinggi axes dan turunkan posisi Y sedikit agar tidak mengubah ukuran elemen lain
legend_ax = plt.axes([0.125, 0.065, 0.5743, 0.115])
legend_ax.set_xticks([])
legend_ax.set_yticks([])
legend_ax.set_frame_on(True)

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


# radio_ax = plt.axes([0.73, 0.75, 0.13, 0.15])
# gauge_ax = plt.axes([0.90, 0.22, 0.06, 0.68])
indicator_ax = plt.axes([0.73, 0.065, 0.23, 0.115])
indicator_ax.set_xticks([])
indicator_ax.set_yticks([])
indicator_ax.set_frame_on(True)

indicator_ax.text(
    0.5, 0.92,
    "Indikator yang dihitung",
    fontsize=8, color="black", fontweight="bold", va="top", ha="center",
    transform=indicator_ax.transAxes,
)

indicator_text = (
    "AU04_r  0.228  brow lowerer\n"
    "AU09_r  0.192  nose wrinkler\n"
    "AU10_r  0.181  upper lip raiser\n"
    "AU20_r  0.146  lip stretcher\n"
    "AU15_r  0.131  lip corner depressor\n"
    "AU05_r  0.122  upper lid raiser"
)
indicator_ax.text(
    0.04, 0.74,
    indicator_text,
    fontsize=6.5, color="black", va="top", ha="left",
    family="monospace", linespacing=1.25,
    transform=indicator_ax.transAxes,
)

# ---------------------------------------------------------------------------
# Timeframe selector - controls how often the expression/tension summary
# below recalculates (does not affect the underlying scoring/plot logic)
# ---------------------------------------------------------------------------
def on_timeframe_change(label):
    global selected_timeframe_seconds, last_conclusion_bucket
    selected_timeframe_seconds = TIMEFRAME_OPTIONS[label]
    last_conclusion_bucket = -1  # force a fresh summary on the next full window
    conclusion_prefix_text.set_text("Menunggu data timeframe penuh...")
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


radio_ax = plt.axes([0.73, 0.75, 0.13, 0.15])
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
        voice_text.set_text("Voice detection:\nOFF\n(unchecked)")
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
    if value >= HIGH_THRESHOLD:
        return "#b00020"
    elif value >= ELEVATED_THRESHOLD:
        return "#c77700"
    elif value <= -ELEVATED_THRESHOLD:
        return "#1565c0"
    return "#2e7d32"


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
summary_panel_ax = plt.axes([0.73, 0.22, 0.13, 0.15])
summary_panel_ax.set_title("Summary", fontsize=9)
summary_panel_ax.set_xticks([])
summary_panel_ax.set_yticks([])

conclusion_prefix_text = summary_panel_ax.text(
    0.03, 0.95,
    "Menunggu data timeframe penuh...",
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
    0.32, 0.68, "", fontsize=8, fontweight="bold", color="black",
    va="top", ha="left", transform=summary_panel_ax.transAxes, visible=False,
)
session_separator_text = summary_panel_ax.text(
    0.45, 0.68, "|", fontsize=8, fontweight="bold", color="black",
    va="top", ha="left", transform=summary_panel_ax.transAxes, visible=False,
)
session_high_text = summary_panel_ax.text(
    0.53, 0.68, "", fontsize=8, fontweight="bold", color="#b00020",
    va="top", ha="left", transform=summary_panel_ax.transAxes, visible=False,
)
session_high_count_text = summary_panel_ax.text(
    0.70, 0.68, "", fontsize=8, fontweight="bold", color="black",
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

disclaimer_note = fig.text(
    0.73, 0.005,
    "Heuristic pattern summary only - not a determination of nervousness, honesty, or intent.",
    fontsize=7, style="italic", color="dimgray", va="bottom", ha="left",
)


def level_color(value):
    """Same color scale used by the threshold zones and the gauge, so the
    summary text visually matches the rest of the display."""
    if value >= HIGH_THRESHOLD:
        return "#b00020"       # red
    elif value >= ELEVATED_THRESHOLD:
        return "#c77700"       # amber/yellow
    elif value <= -ELEVATED_THRESHOLD:
        return "#1565c0"       # blue
    return "#2e7d32"           # green


def update_conclusion():
    """Recomputes and displays the plain-language summary once the current
    timeframe window has fully elapsed. Heuristic thresholds, same caveat
    as the rest of this project - a pattern summary, not a diagnosis."""
    global last_conclusion_bucket

    if len(plot_timestamps) == 0:
        return

    max_t = plot_timestamps[-1]
    current_bucket = int(max_t // selected_timeframe_seconds)

    if current_bucket == last_conclusion_bucket:
        return
    if max_t < selected_timeframe_seconds:
        return  # not enough data yet for a full window

    window_start = current_bucket * selected_timeframe_seconds
    window_vals = [y for t, y in zip(plot_timestamps, plot_scores) if window_start <= t < window_start + selected_timeframe_seconds]
    if not window_vals:
        return

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

    variability_note = ""
    if std_val >= HIGH_VARIABILITY_THRESHOLD:
        variability_note = "\n(fluktuatif)"

    conclusion_prefix_text.set_text(f"[{selected_timeframe_seconds} detik terakhir]")

    conclusion_level_text.set_text(level)
    conclusion_level_text.set_color(level_color(mean_val))

    detail_text = f"avg {mean_val:+.2f}, var {std_val:.2f}{variability_note}"
    conclusion_detail_text.set_text(detail_text)

    conclusion_note_text.set_text(note)

    last_conclusion_bucket = current_bucket


# ---------------------------------------------------------------------------
# Update loop - same structure as the original update_plot, gated behind
# session_started so nothing runs until the button is clicked, plus gaze
# and voice scoring layered into the existing compute step
# ---------------------------------------------------------------------------
def update_plot(frame_num):
    global last_row_count, baseline_mean, baseline_established, start_time
    global csv_header, csv_byte_offset
    global visual_baseline_mean, voice_baseline_mean

    if not session_started:
        return line,

    try:
        if not os.path.exists(CSV_PATH):
            return line,

        with open(CSV_PATH, "r", newline="") as f:
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
        for col in ("gaze_angle_x", "gaze_angle_y", "confidence", "success"):
            if col in new_rows.columns:
                new_rows[col] = pd.to_numeric(new_rows[col], errors='coerce')
    except Exception as e:
        print(f"Read error: {e}")
        return line,

    last_row_count += len(new_rows)  # kept for stats/logging only, no longer used for slicing

    skipped_low_confidence = 0
    for _, row in new_rows.iterrows():
        if not row_is_reliable(row):
            skipped_low_confidence += 1
            continue

        au_score = compute_au_score(row)
        if pd.isna(au_score):
            continue  # skip rows with no valid AU data (e.g. face not detected)

        gx, gy = row.get("gaze_angle_x", np.nan), row.get("gaze_angle_y", np.nan)
        if pd.notna(gx):
            gaze_x_history.append(gx)
        if pd.notna(gy):
            gaze_y_history.append(gy)
        gaze_score = compute_gaze_variance_score()
        visual_score = au_score + (gaze_score * GAZE_VARIANCE_WEIGHT)

        if voice_analyzer:
            voice_score = voice_analyzer.get_score()
            active_visual_weight, active_voice_weight = VISUAL_CHANNEL_WEIGHT, VOICE_CHANNEL_WEIGHT
        else:
            # voice disabled (or failed to start) - give the facial channel
            # the full weight instead of silently losing VOICE_CHANNEL_WEIGHT's
            # share of the score (which would otherwise shrink every reading)
            voice_score = 0.0
            active_visual_weight, active_voice_weight = 1.0, 0.0

        raw_score = (visual_score * active_visual_weight) + (voice_score * active_voice_weight)

        now = time.time()
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
            continue

        relative_score = raw_score - baseline_mean
        tension_history.append(relative_score)
        smoothed = sum(tension_history) / len(tension_history)

        level_tracker.update(elapsed, smoothed)

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
        visual_plot_scores.append(visual_smoothed)
        voice_plot_scores.append(voice_smoothed)

        log_rows.append({
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "frame": int(row.get("frame", -1)),
            "au_score": round(au_score, 4),
            "gaze_variance_score": round(gaze_score, 4),
            "voice_score": round(voice_score, 4),
            "raw_score": round(raw_score, 4),
            "baseline": round(baseline_mean, 4),
            "relative_score": round(relative_score, 4),
            "smoothed_score": round(smoothed, 4),
            "visual_smoothed": round(visual_smoothed, 4),
            "voice_smoothed": round(voice_smoothed, 4),
        })

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

    update_voice_panel()

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
    voice_analyzer.stop()

if session_started and log_rows:
    log_df = pd.DataFrame(log_rows)
    stamped_path = LOG_PATH.replace(".csv", f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    log_df.to_csv(stamped_path, index=False)
    print(f"Session log saved to {stamped_path} ({len(log_df)} rows)")
elif not session_started:
    print("Session was already stopped and saved before the window closed.")
else:
    print("No data logged this session.")
