# Import necessary libraries
import os
import subprocess
import time
from collections import deque
from datetime import datetime
from io import StringIO

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, CheckButtons, RadioButtons

from voice_analyzer import VoiceAnalyzer

# Import custom modules voice analyzer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# OpenFace log and csv file paths
CSV_PATH = r"C:\OpenFace\live_output\session1.csv"
LOG_PATH = r"C:\OpenFace\live_output\tension_log.csv"

# OpenFace launch
OPENFACE_EXE = r"C:\OpenFace\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
OPENFACE_OUT_DIR = r"C:\OpenFace\live_output"

# Tension weights
TENSION_WEIGHTS = {
    "AU04_r": 0.228,
    "AU09_r": 0.192,
    "AU10_r": 0.181,
    "AU20_r": 0.146,
    "AU15_r": 0.131,
    "AU05_r": 0.122,
}

AU_LABELS = {
    "AU04_r": "brow lowerer",
    "AU09_r": "nose wrinkler",
    "AU10_r": "upper lip raiser",
    "AU20_r": "lip stretcher",
    "AU15_r": "lip corner depressor",
    "AU05_r": "upper lid raiser",
}

# Smoothing and calibrating the initial baseline
WINDOW_SIZE = 10
BASELINE_DURATION_SECONDS = 10

# Gaze and voice channel settings
GAZE_WINDOW = 30
GAZE_VARIANCE_WEIGHT = 0.3
VOICE_CHANNEL_WEIGHT = 0.25
VISUAL_CHANNEL_WEIGHT = 0.75

# Level thresholds and episode settings
ELEVATED_THRESHOLD = 0.5
HIGH_THRESHOLD = 1.0
HIGH_VARIABILITY_THRESHOLD = 0.6
LEVEL_EXIT_COOLDOWN_SECONDS = 2.0

# Gauge and timeframe settings
PEAK_HOLD_SECONDS = 8
PEAK_DECAY_PER_SECOND = 0.15
TIMEFRAME_OPTIONS = {"30s": 30, "1min": 60, "2min": 120}
DEFAULT_TIMEFRAME_LABEL = "1min"
MIN_TRACKING_CONFIDENCE = 0.75

# ---------------------------------------------------------------------------
# State and scoring helpers
# ---------------------------------------------------------------------------

# Track high/elevated episodes over time instead of counting individual spikes.
class LevelEpisodeTracker:
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
        if value >= ELEVATED_THRESHOLD:
            return "ELEVATED"
        return None

    def _finalize_active_episode(self, end_t):
        if self.active_level is None or self.episode_start is None:
            return
        duration = self.episode_last_seen - self.episode_start
        if duration > 0:
            self.episodes[self.active_level].append(
                {"start": self.episode_start, "end": self.episode_last_seen, "duration": duration}
            )
        self.active_level = None
        self.episode_start = None
        self.episode_last_seen = None
        self.pending_exit_since = None

    def update(self, t, value):
        level = self.classify(value)

        if level == self.active_level:
            if level is not None:
                self.episode_last_seen = t
                self.pending_exit_since = None
            return

        if level is not None:
            self._finalize_active_episode(t)
            self.active_level = level
            self.episode_start = t
            self.episode_last_seen = t
            self.pending_exit_since = None
            return

        if self.active_level is not None:
            if self.pending_exit_since is None:
                self.pending_exit_since = t
            elif t - self.pending_exit_since >= LEVEL_EXIT_COOLDOWN_SECONDS:
                self._finalize_active_episode(t)

    def finalize(self, end_t):
        self._finalize_active_episode(end_t)

    def summary(self):
        result = {}
        for level in ("ELEVATED", "HIGH"):
            episodes = self.episodes[level]
            durations = [item["duration"] for item in episodes]
            result[level] = {
                "count": len(episodes),
                "total_duration": sum(durations),
                "longest": max(durations) if durations else 0.0,
            }
        return result


level_tracker = LevelEpisodeTracker()
manual_events = []
manual_event_lines = []

tension_history = deque(maxlen=WINDOW_SIZE)
plot_timestamps = deque(maxlen=200)
plot_scores = deque(maxlen=200)
all_points = []

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
csv_header = None
csv_byte_offset = 0
log_rows = []

openface_process = None
voice_analyzer = None
session_started = False
voice_enabled = False

selected_timeframe_seconds = TIMEFRAME_OPTIONS[DEFAULT_TIMEFRAME_LABEL]
last_conclusion_bucket = -1

current_peak_value = 0.0
current_peak_last_rise_time = None

# Session state: rolling history, baseline values, and live OpenFace/audio objects.


# Reject frames where OpenFace could not track the face reliably.
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


# Combine the selected facial action units using their configured weights.
def compute_au_score(row):
    score = 0.0
    for au, weight in TENSION_WEIGHTS.items():
        value = row.get(au, np.nan)
        if pd.notna(value):
            score += float(value) * weight
    return score


# Use recent gaze movement as an additional visual variability signal.
def compute_gaze_variance_score():
    if len(gaze_x_history) < 2:
        return 0.0
    return (float(np.std(gaze_x_history)) + float(np.std(gaze_y_history))) / 2.0


def level_color(value):
    if value >= HIGH_THRESHOLD:
        return "#b00020"
    if value >= ELEVATED_THRESHOLD:
        return "#c77700"
    if value <= -ELEVATED_THRESHOLD:
        return "#1565c0"
    return "#2e7d32"


def classify_full(value):
    if value >= HIGH_THRESHOLD:
        return "HIGH"
    if value >= ELEVATED_THRESHOLD:
        return "ELEVATED"
    if value <= -ELEVATED_THRESHOLD:
        return "BELOW"
    return "BASELINE"

# ---------------------------------------------------------------------------
# Dashboard setup
# ---------------------------------------------------------------------------

# Main chart, threshold bands, status text, and live gauge.
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(left=0.422, right=0.862, bottom=0.39, top=0.90)

line, = ax.plot([], [], color="black", linewidth=1.6, zorder=5, label="Combined")
visual_line, = ax.plot([], [], color="#1565c0", linewidth=1.0, alpha=0.6, zorder=4, label="Face only")
voice_line, = ax.plot([], [], color="#2e7d32", linewidth=1.0, alpha=0.6, zorder=4, label="Voice only")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Tension Score (relative to baseline)")
ax.set_title("Live Tension Monitor - PROTOTYPE")
ax.axhline(0, color="gray", linestyle="--", linewidth=1)

STATUS_LABELS_ID = {
    "HIGH": "TINGGI",
    "ELEVATED": "MENINGKAT",
    "BASELINE": "STABIL",
    "BELOW": "TENANG",
}

big_status_text = ax.text(
    0.98, 0.95, "-", transform=ax.transAxes, fontsize=26, fontweight="bold",
    ha="right", va="top", color="#2e7d32", zorder=10,
)
trend_arrow_text = ax.text(
    0.98, 0.80, "", transform=ax.transAxes, fontsize=16, ha="right", va="top",
    color="dimgray", zorder=10,
)


def draw_threshold_zones():
    y_min, y_max = ax.get_ylim()
    ax.axhspan(HIGH_THRESHOLD, y_max, color="#b00020", alpha=0.10, zorder=0)
    ax.axhspan(ELEVATED_THRESHOLD, HIGH_THRESHOLD, color="#c77700", alpha=0.10, zorder=0)
    ax.axhspan(-ELEVATED_THRESHOLD, ELEVATED_THRESHOLD, color="#2e7d32", alpha=0.08, zorder=0)
    ax.axhspan(y_min, -ELEVATED_THRESHOLD, color="#1565c0", alpha=0.08, zorder=0)


ax.set_ylim(-2, 2)
draw_threshold_zones()


# ---------------------------------------------------------------------------
# Session controls
# ---------------------------------------------------------------------------

# Start/stop the session and reset state between runs.
def start_openface(event):
    global openface_process, voice_analyzer, session_started, start_time, baseline_established
    global baseline_mean, log_rows, current_peak_value, current_peak_last_rise_time
    global csv_header, csv_byte_offset
    global level_tracker
    global visual_baseline_mean, voice_baseline_mean

    if session_started:
        return

    # OpenFace writes its CSV continuously while FeatureExtraction is running.
    os.makedirs(OPENFACE_OUT_DIR, exist_ok=True)

    try:
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
    except Exception as exc:
        print(f"Could not remove old CSV (continuing anyway): {exc}")

    try:
        openface_process = subprocess.Popen([
            OPENFACE_EXE,
            "-device", "0",
            "-out_dir", OPENFACE_OUT_DIR,
            "-of", "session1",
        ])
    except FileNotFoundError:
        ax.set_title(f"ERROR: OpenFace exe not found at {OPENFACE_EXE}")
        fig.canvas.draw_idle()
        return

    # Voice analysis is optional; facial scoring continues if the microphone is unavailable.
    if voice_enabled:
        voice_analyzer = VoiceAnalyzer()
        try:
            voice_analyzer.start()
        except Exception as exc:
            print(f"Voice analyzer failed to start (continuing with visual-only scoring): {exc}")
            voice_analyzer = None
    else:
        voice_analyzer = None
        print("Voice detection is OFF for this session (facial score only).")

    # Reset all session-specific state before accepting new frames.
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
    global openface_process, voice_analyzer, session_started

    if not session_started:
        return

    session_started = False

    # Stop both live data sources before calculating the final session report.
    if openface_process:
        openface_process.terminate()
        openface_process = None
    if voice_analyzer:
        voice_analyzer.stop()
        voice_analyzer = None

    end_t = plot_timestamps[-1] if len(plot_timestamps) > 0 else 0.0
    level_tracker.finalize(end_t)
    stats = level_tracker.summary()

    if log_rows:
        # Save the detailed frame log and a compact episode report with a timestamp.
        log_df = pd.DataFrame(log_rows)
        stamped_path = LOG_PATH.replace(".csv", f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        log_df.to_csv(stamped_path, index=False)
        print(f"Session log saved to {stamped_path} ({len(log_df)} rows)")
        ax.set_title(f"Stopped. Log saved: {os.path.basename(stamped_path)}")

        report_lines = [
            f"RINGKASAN SESI (total durasi: {end_t:.0f} detik)",
            f"ELEVATED: {stats['ELEVATED']['count']}x kejadian, total {stats['ELEVATED']['total_duration']:.0f}s, "
            f"terlama {stats['ELEVATED']['longest']:.0f}s",
            f"HIGH: {stats['HIGH']['count']}x kejadian, total {stats['HIGH']['total_duration']:.0f}s, "
            f"terlama {stats['HIGH']['longest']:.0f}s",
            "",
            "Catatan: hitungan ini menunjukkan berapa kali pola tension naik ke level tersebut secara terpisah.",
        ]
        report_text = "\n".join(report_lines)
        print(report_text)

        report_path = stamped_path.replace(".csv", "_report.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as handle:
                handle.write(report_text)
            print(f"Session report saved to {report_path}")
        except Exception as exc:
            print(f"Could not save report file: {exc}")

        conclusion_prefix_text.set_text("RINGKASAN SESI:")
        conclusion_level_text.set_visible(False)
        session_elevated_text.set_text("Elevated")
        session_elevated_count_text.set_text(f"{stats['ELEVATED']['count']}x")
        session_high_text.set_text("High")
        session_high_count_text.set_text(f"{stats['HIGH']['count']}x")
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
        conclusion_note_text.set_text("Bukan kesimpulan kejujuran\npertimbangkan dengan konteks wawancara.")
    else:
        ax.set_title("Stopped. No data logged this session.")

    start_button.label.set_text("Start OpenFace")
    fig.canvas.draw_idle()
    print("Session stopped.")


def toggle_voice(label):
    global voice_enabled
    voice_enabled = not voice_enabled
    print(f"Voice detection {'enabled' if voice_enabled else 'disabled'} - takes effect on next Start.")


# Add a timestamp marker when the operator observes a notable event.
def mark_manual_event(event):
    if not session_started or start_time is None:
        return
    elapsed = time.time() - start_time
    manual_events.append((elapsed, "Ditandai operator"))
    vline = ax.axvline(elapsed, color="#6a1b9a", linestyle=":", linewidth=1.4, alpha=0.9, zorder=4)
    manual_event_lines.append(vline)
    fig.canvas.draw_idle()
    print(f"Event marked at t={elapsed:.1f}s")


def handle_key_press(event):
    if event.key in (".", "period", "decimal"):
        mark_manual_event(event)
    elif event.key == " ":
        show_quick_summary()


fig.canvas.mpl_connect("key_press_event", handle_key_press)


# Dashboard buttons and panels.
button_ax = plt.axes([0.476, 0.25, 0.15, 0.075])
start_button = Button(button_ax, "Start OpenFace")
start_button.on_clicked(start_openface)

stop_button_ax = plt.axes([0.66, 0.25, 0.15, 0.075])
stop_button = Button(stop_button_ax, "Stop")
stop_button.on_clicked(stop_openface)

voice_toggle_ax = plt.axes([0.547, 0.18, 0.36, 0.06])
voice_toggle_ax.set_zorder(10)
voice_toggle_ax.set_frame_on(False)
voice_toggle = CheckButtons(
    voice_toggle_ax,
    ["Enable Voice Detection"],
    [False],
    frame_props={"sizes": [75], "linewidth": 1},
    check_props={"sizes": [30], "linewidth": 1},
)
voice_toggle.labels[0].set_x(0.18)
voice_toggle.labels[0].set_fontsize(8)
voice_toggle.on_clicked(toggle_voice)

legend_ax = plt.axes([0.39, 0.065, 0.5703, 0.115])
legend_ax.set_xticks([])
legend_ax.set_yticks([])
legend_ax.set_frame_on(True)
legend_ax.set_visible(True)
legend_ax.text(
    0.5, 0.92, "Panduan Level Tension (dibanding baseline/kondisi awal target)",
    fontsize=8, color="black", fontweight="bold", va="top", ha="center", transform=legend_ax.transAxes,
)
legend_ax.text(0.01, 0.74, "BASELINE", fontsize=8, color="#2e7d32", va="top", ha="left", transform=legend_ax.transAxes)
legend_ax.text(0.105, 0.74, " = Sesuai kondisi awal - menandai ekspresi normal/netral.", fontsize=8, color="black", va="top", ha="left", transform=legend_ax.transAxes)
legend_ax.text(0.01, 0.56, "ELEVATED", fontsize=8, color="#c77700", va="top", ha="left", transform=legend_ax.transAxes)
legend_ax.text(0.105, 0.56, " = Peningkatan pola tension - bisa karena gugup, berpikir keras, tidak nyaman, dll.", fontsize=8, color="black", va="top", ha="left", transform=legend_ax.transAxes)
legend_ax.text(0.01, 0.38, "HIGH", fontsize=8, color="#b00020", va="top", ha="left", transform=legend_ax.transAxes)
legend_ax.text(0.105, 0.38, " = Peningkatan pola tension yang signifikan - terindikasi tekanan tinggi.", fontsize=8, color="black", va="top", ha="left", transform=legend_ax.transAxes)
legend_ax.text(0.01, 0.20, "PENTING", fontsize=8, color="#b00020", fontweight="bold", va="top", ha="left", transform=legend_ax.transAxes)
legend_ax.text(0.074, 0.20, "- Ini adalah ringkasan pola, BUKAN kesimpulan kejujuran/kebohongan target.", fontsize=8, color="black", va="top", ha="left", transform=legend_ax.transAxes)

radio_ax = plt.axes([0.04, 0.065, 0.08, 0.115])
radio_ax.set_title("Timeframe", fontsize=9)
timeframe_radio = RadioButtons(radio_ax, list(TIMEFRAME_OPTIONS.keys()), active=list(TIMEFRAME_OPTIONS.keys()).index(DEFAULT_TIMEFRAME_LABEL))
timeframe_radio.on_clicked(lambda label: on_timeframe_change(label))

voice_panel_ax = plt.axes([0.73, 0.40, 0.13, 0.32])
voice_panel_ax.set_title("Voice Channel", fontsize=9)
voice_panel_ax.set_xticks([])
voice_panel_ax.set_yticks([])
voice_panel_ax.set_visible(False)
for spine in voice_panel_ax.spines.values():
    spine.set_visible(True)
voice_text = voice_panel_ax.text(0.05, 0.95, "Waiting for audio...", fontsize=8, va="top", ha="left", family="monospace", transform=voice_panel_ax.transAxes)


def update_voice_panel():
    if not voice_enabled:
        voice_text.set_text("Voice detection:\nOFF\n(unchecked)")
        return
    if voice_analyzer is None:
        voice_text.set_text("Voice channel:\nnot running")
        return

    # Display the latest voice score and its component features when available.
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


# Live gauge and session summary panels.
gauge_ax = plt.axes([0.90, 0.22, 0.06, 0.68])
gauge_ax.set_title("Now", fontsize=9)
gauge_ax.set_xticks([])
gauge_ax.set_xlim(0, 1)
gauge_ax.set_ylim(-2, 2)

gauge_zero_line = gauge_ax.axhline(0, color="gray", linestyle="--", linewidth=1)
gauge_bar = gauge_ax.bar([0.5], [0], width=0.6, color="#2e7d32", zorder=5)[0]
gauge_peak_marker = gauge_ax.axhline(0, color="black", linewidth=1.5, xmin=0.15, xmax=0.85, zorder=6)
gauge_value_text = gauge_ax.text(0.5, 0, "", ha="center", va="bottom", fontsize=8, fontweight="bold", transform=gauge_ax.transData)


def update_big_status():
    if len(plot_scores) == 0:
        return

    # Compare the current smoothed score with the score from roughly ten seconds ago.
    current_value = plot_scores[-1]
    level = classify_full(current_value)
    big_status_text.set_text(STATUS_LABELS_ID[level])
    big_status_text.set_color(level_color(current_value))

    current_t = plot_timestamps[-1]
    reference_t = current_t - 10.0
    past_values = [s for t, s in zip(plot_timestamps, plot_scores) if t <= reference_t]

    if not past_values:
        trend_arrow_text.set_text("")
        return

    past_value = past_values[-1]
    delta = current_value - past_value

    if delta > 0.15:
        trend_arrow_text.set_text("↑ naik")
        trend_arrow_text.set_color("#b00020")
    elif delta < -0.15:
        trend_arrow_text.set_text("↓ turun")
        trend_arrow_text.set_color("#1565c0")
    else:
        trend_arrow_text.set_text("→ stabil")
        trend_arrow_text.set_color("dimgray")


def update_gauge():
    global current_peak_value, current_peak_last_rise_time

    if len(plot_scores) == 0:
        return

    # Hold the highest recent value briefly so short peaks remain visible to the operator.
    current_value = plot_scores[-1]
    now = time.time()

    if current_value > current_peak_value:
        current_peak_value = current_value
        current_peak_last_rise_time = now
    elif current_peak_last_rise_time is not None:
        seconds_since_rise = now - current_peak_last_rise_time
        if seconds_since_rise > PEAK_HOLD_SECONDS:
            current_peak_value = max(current_value, current_peak_value - PEAK_DECAY_PER_SECOND * 0.5)

    y_min, y_max = ax.get_ylim()
    gauge_ax.set_ylim(y_min, y_max)
    gauge_bar.set_height(current_value)
    gauge_bar.set_color(level_color(current_value))
    gauge_peak_marker.set_ydata([current_peak_value, current_peak_value])
    gauge_value_text.set_position((0.5, current_value))
    gauge_value_text.set_text(f"{current_value:+.2f}")
    gauge_value_text.set_va("bottom" if current_value >= 0 else "top")

    n_elevated = len(level_tracker.episodes["ELEVATED"])
    n_high = len(level_tracker.episodes["HIGH"])
    gauge_ax.set_title(f"Now  (E:{n_elevated} H:{n_high})", fontsize=9)


summary_panel_ax = plt.axes([0.14, 0.065, 0.23, 0.115])
summary_panel_ax.set_title("Summary", fontsize=9)
summary_panel_ax.set_xticks([])
summary_panel_ax.set_yticks([])

conclusion_prefix_text = summary_panel_ax.text(0.03, 0.95, "Menunggu data timeframe...", fontsize=8, wrap=True, va="top", ha="left", transform=summary_panel_ax.transAxes)
conclusion_level_text = summary_panel_ax.text(0.03, 0.68, "", fontsize=8, fontweight="bold", wrap=True, va="top", ha="left", transform=summary_panel_ax.transAxes)
session_elevated_text = summary_panel_ax.text(0.03, 0.68, "", fontsize=8, fontweight="bold", color="#c77700", va="top", ha="left", transform=summary_panel_ax.transAxes, visible=False)
session_elevated_count_text = summary_panel_ax.text(0.21, 0.68, "", fontsize=8, fontweight="bold", color="black", va="top", ha="left", transform=summary_panel_ax.transAxes, visible=False)
session_separator_text = summary_panel_ax.text(0.29, 0.68, "|", fontsize=8, fontweight="bold", color="black", va="top", ha="left", transform=summary_panel_ax.transAxes, visible=False)
session_high_text = summary_panel_ax.text(0.34, 0.68, "", fontsize=8, fontweight="bold", color="#b00020", va="top", ha="left", transform=summary_panel_ax.transAxes, visible=False)
session_high_count_text = summary_panel_ax.text(0.45, 0.68, "", fontsize=8, fontweight="bold", color="black", va="top", ha="left", transform=summary_panel_ax.transAxes, visible=False)
conclusion_detail_text = summary_panel_ax.text(0.03, 0.50, "", fontsize=7, wrap=True, va="top", ha="left", transform=summary_panel_ax.transAxes)
conclusion_note_text = summary_panel_ax.text(0.03, 0.28, "", fontsize=6.5, style="italic", wrap=True, va="top", ha="left", color="dimgray", transform=summary_panel_ax.transAxes)

disclaimer_note = fig.text(0.73, 0.005, "Heuristic pattern summary only - not a determination of nervousness, honesty, or intent.", fontsize=7, style="italic", color="dimgray", va="bottom", ha="left")


def describe_window(window_vals):
    # Classify a timeframe by its average score and variability.
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

    variability_note = "pola berubah-ubah" if std_val >= HIGH_VARIABILITY_THRESHOLD else "pola stabil"
    return level, note, variability_note, mean_val


def on_timeframe_change(label):
    global selected_timeframe_seconds, last_conclusion_bucket
    # Clear the previous conclusion so the new timeframe is recalculated.
    selected_timeframe_seconds = TIMEFRAME_OPTIONS[label]
    last_conclusion_bucket = -1
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


# ---------------------------------------------------------------------------
# Summary and reporting
# ---------------------------------------------------------------------------

# Convert recent score history into a plain-language assessment.
def update_conclusion():
    global last_conclusion_bucket

    if len(all_points) == 0:
        return

    # Update once per completed timeframe bucket instead of every animation tick.
    max_t = all_points[-1][0]
    current_bucket = int(max_t // selected_timeframe_seconds)

    if current_bucket == last_conclusion_bucket:
        return
    if max_t < selected_timeframe_seconds:
        return

    window_start = current_bucket * selected_timeframe_seconds
    window_vals = [y for t, y in all_points if window_start <= t < window_start + selected_timeframe_seconds]
    if not window_vals:
        return

    level, note, variability_note, mean_val = describe_window(window_vals)
    conclusion_prefix_text.set_text(f"[{selected_timeframe_seconds} detik terakhir]")
    conclusion_level_text.set_text(level)
    conclusion_level_text.set_color(level_color(mean_val))
    conclusion_detail_text.set_text(variability_note)
    conclusion_note_text.set_text(note)
    last_conclusion_bucket = current_bucket


def show_quick_summary():
    if not session_started or len(all_points) == 0:
        return

    # The keyboard shortcut summarizes the most recent partial window immediately.
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
# Live update loop
# ---------------------------------------------------------------------------

# Read new OpenFace rows, compute scores, and refresh the dashboard.
def update_plot(frame_num):
    global last_row_count, baseline_mean, baseline_established, start_time
    global csv_header, csv_byte_offset
    global visual_baseline_mean, voice_baseline_mean

    if not session_started:
        return line,

    try:
        # Read only complete CSV lines added since the previous animation tick.
        if not os.path.exists(CSV_PATH):
            return line,

        with open(CSV_PATH, "r", newline="") as handle:
            if csv_header is None:
                csv_header = handle.readline()
                csv_byte_offset = handle.tell()

            handle.seek(csv_byte_offset)
            new_lines = []
            while True:
                pos_before = handle.tell()
                line_str = handle.readline()
                if not line_str or not line_str.endswith("\n"):
                    csv_byte_offset = pos_before
                    break
                new_lines.append(line_str)

        if not new_lines:
            return line,

        chunk_csv = csv_header + "".join(new_lines)
        new_rows = pd.read_csv(StringIO(chunk_csv), low_memory=False)
        new_rows.columns = new_rows.columns.str.strip()

        for au in TENSION_WEIGHTS.keys():
            if au in new_rows.columns:
                new_rows[au] = pd.to_numeric(new_rows[au], errors="coerce")
        for col in ("gaze_angle_x", "gaze_angle_y", "confidence", "success"):
            if col in new_rows.columns:
                new_rows[col] = pd.to_numeric(new_rows[col], errors="coerce")
    except Exception as exc:
        print(f"Read error: {exc}")
        return line,

    last_row_count += len(new_rows)

    # Convert each reliable OpenFace frame into visual, voice, and combined scores.
    for _, row in new_rows.iterrows():
        if not row_is_reliable(row):
            continue

        au_score = compute_au_score(row)
        if pd.isna(au_score):
            continue

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
            voice_score = 0.0
            active_visual_weight, active_voice_weight = 1.0, 0.0

        raw_score = (visual_score * active_visual_weight) + (voice_score * active_voice_weight)

        now = time.time()
        if start_time is None:
            start_time = now
        elapsed = now - start_time

        # The first calibration period defines the target's neutral reference level.
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

        # Report tension as a change from baseline, then smooth short-term noise.
        relative_score = raw_score - baseline_mean
        tension_history.append(relative_score)
        smoothed = sum(tension_history) / len(tension_history)
        level_tracker.update(elapsed, smoothed)

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

        # Keep the raw components so the session can be inspected after the run.
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
        # Refresh all visual elements after new scored frames are added.
        line.set_data(list(plot_timestamps), list(plot_scores))
        visual_line.set_data(list(plot_timestamps), list(visual_plot_scores))
        voice_line.set_data(list(plot_timestamps), list(voice_plot_scores))
        ax.set_xlim(max(0, plot_timestamps[0]), plot_timestamps[-1] + 1)

        all_visible_scores = list(plot_scores) + list(visual_plot_scores) + list(voice_plot_scores)
        y_min = min(all_visible_scores) - 0.2
        y_max = max(all_visible_scores) + 0.2
        if not (np.isnan(y_min) or np.isnan(y_max)):
            ax.set_ylim(y_min, y_max)

        for patch in list(ax.patches):
            patch.remove()
        draw_threshold_zones()
        ax.legend(loc="upper left", fontsize=7, framealpha=0.7)

        update_conclusion()
        update_gauge()
        update_big_status()

    update_voice_panel()
    return line,


# Start the animation loop and show the dashboard.
ani = animation.FuncAnimation(fig, update_plot, interval=500, cache_frame_data=False)
print("Click 'Start OpenFace' in the plot window to begin capture.")
plt.show()

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
