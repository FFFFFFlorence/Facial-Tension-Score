import subprocess
import os
import pandas as pd
import numpy as np
import time
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, RadioButtons

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
VOICE_CHANNEL_WEIGHT = 0.4  # how much the voice score contributes to the raw score
VISUAL_CHANNEL_WEIGHT = 0.6 # facial (AU + gaze) contribution - these two should sum to 1.0

# Same heuristic thresholds used elsewhere in this project - not validated,
# just a starting point for coloring the "is this elevated" zones.
ELEVATED_THRESHOLD = 0.5
HIGH_THRESHOLD = 1.0
HIGH_VARIABILITY_THRESHOLD = 0.6  # std-dev above this -> "fluctuating" note in the summary

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

gaze_x_history = deque(maxlen=GAZE_WINDOW)
gaze_y_history = deque(maxlen=GAZE_WINDOW)

baseline_scores = []
baseline_mean = None
baseline_established = False
start_time = None

last_row_count = 0
log_rows = []

openface_process = None
voice_analyzer = None
session_started = False  # gates the update loop until the Start button is pressed

selected_timeframe_seconds = TIMEFRAME_OPTIONS[DEFAULT_TIMEFRAME_LABEL]
last_conclusion_bucket = -1

# peak-hold gauge state
current_peak_value = 0.0
current_peak_last_rise_time = None


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
plt.subplots_adjust(bottom=0.32, right=0.70, top=0.83)  # top=0.83 matches the gauge's top edge, so everything lines up

line, = ax.plot([], [], color='black', linewidth=1.6, zorder=5)
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

    voice_analyzer = VoiceAnalyzer()
    try:
        voice_analyzer.start()
    except Exception as e:
        print(f"Voice analyzer failed to start (continuing with visual-only scoring): {e}")
        voice_analyzer = None

    # full reset so Start -> Stop -> Start again behaves as a clean new session
    start_time = None
    baseline_established = False
    baseline_mean = None
    baseline_scores.clear()
    tension_history.clear()
    gaze_x_history.clear()
    gaze_y_history.clear()
    plot_timestamps.clear()
    plot_scores.clear()
    last_row_count = 0
    current_peak_value = 0.0
    current_peak_last_rise_time = None
    log_rows = []

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

    # save whatever was captured this session, without needing to close the window
    if log_rows:
        log_df = pd.DataFrame(log_rows)
        stamped_path = LOG_PATH.replace(".csv", f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        log_df.to_csv(stamped_path, index=False)
        print(f"Session log saved to {stamped_path} ({len(log_df)} rows)")
        ax.set_title(f"Stopped. Log saved: {os.path.basename(stamped_path)}")
    else:
        ax.set_title("Stopped. No data logged this session.")

    start_button.label.set_text("Start OpenFace")
    fig.canvas.draw_idle()
    print("Session stopped.")


button_ax = plt.axes([0.32, 0.18, 0.17, 0.075])
start_button = Button(button_ax, "Start OpenFace")
start_button.on_clicked(start_openface)

stop_button_ax = plt.axes([0.51, 0.18, 0.17, 0.075])
stop_button = Button(stop_button_ax, "Stop")
stop_button.on_clicked(stop_openface)


# ---------------------------------------------------------------------------
# Timeframe selector - controls how often the expression/tension summary
# below recalculates (does not affect the underlying scoring/plot logic)
# ---------------------------------------------------------------------------
def on_timeframe_change(label):
    global selected_timeframe_seconds, last_conclusion_bucket
    selected_timeframe_seconds = TIMEFRAME_OPTIONS[label]
    last_conclusion_bucket = -1  # force a fresh summary on the next full window
    conclusion_prefix_text.set_text("Waiting for a full\ntimeframe window...")
    conclusion_level_text.set_text("")
    conclusion_detail_text.set_text("")
    fig.canvas.draw_idle()


radio_ax = plt.axes([0.73, 0.68, 0.13, 0.15])
radio_ax.set_title("Timeframe", fontsize=9)
timeframe_radio = RadioButtons(radio_ax, list(TIMEFRAME_OPTIONS.keys()), active=list(TIMEFRAME_OPTIONS.keys()).index(DEFAULT_TIMEFRAME_LABEL))
timeframe_radio.on_clicked(on_timeframe_change)


# ---------------------------------------------------------------------------
# Voice metrics visualizer - shows the live rolling voice_score plus its
# individual component features, so the voice channel isn't a silent
# black box running in the background
# ---------------------------------------------------------------------------
voice_panel_ax = plt.axes([0.73, 0.33, 0.13, 0.32])
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
gauge_ax = plt.axes([0.90, 0.15, 0.06, 0.68])
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


# ---------------------------------------------------------------------------
# Expression/tension summary text - recalculated once per full timeframe
# window. Positioned in the right-hand column, directly under the Voice
# Channel panel, and color-coded to match the current tension level.
# ---------------------------------------------------------------------------
summary_panel_ax = plt.axes([0.73, 0.15, 0.13, 0.15])
summary_panel_ax.set_title("Summary", fontsize=9)
summary_panel_ax.set_xticks([])
summary_panel_ax.set_yticks([])

conclusion_prefix_text = summary_panel_ax.text(
    0.03, 0.95,
    "Waiting for a full\ntimeframe window...",
    fontsize=8, wrap=True, va="top", ha="left",
    transform=summary_panel_ax.transAxes,
)
conclusion_level_text = summary_panel_ax.text(
    0.03, 0.65,
    "",
    fontsize=8, fontweight="bold", wrap=True, va="top", ha="left",
    transform=summary_panel_ax.transAxes,
)
conclusion_detail_text = summary_panel_ax.text(
    0.03, 0.45,
    "",
    fontsize=8, wrap=True, va="top", ha="left",
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
        level = "HIGH tension"
    elif mean_val >= ELEVATED_THRESHOLD:
        level = "ELEVATED tension"
    elif mean_val <= -ELEVATED_THRESHOLD:
        level = "BELOW BASELINE"
    else:
        level = "WITHIN baseline"

    variability_note = ""
    if std_val >= HIGH_VARIABILITY_THRESHOLD:
        variability_note = "\n(fluctuating)"

    conclusion_prefix_text.set_text(f"[Last {selected_timeframe_seconds}s]")

    conclusion_level_text.set_text(level)
    conclusion_level_text.set_color(level_color(mean_val))

    detail_text = f"avg {mean_val:+.2f}, var {std_val:.2f}{variability_note}"
    conclusion_detail_text.set_text(detail_text)

    last_conclusion_bucket = current_bucket


# ---------------------------------------------------------------------------
# Update loop - same structure as the original update_plot, gated behind
# session_started so nothing runs until the button is clicked, plus gaze
# and voice scoring layered into the existing compute step
# ---------------------------------------------------------------------------
def update_plot(frame_num):
    global last_row_count, baseline_mean, baseline_established, start_time

    if not session_started:
        return line,

    try:
        df = pd.read_csv(CSV_PATH, low_memory=False)
        df.columns = df.columns.str.strip()

        for au in TENSION_WEIGHTS.keys():
            if au in df.columns:
                df[au] = pd.to_numeric(df[au], errors='coerce')
        for col in ("gaze_angle_x", "gaze_angle_y"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    except Exception as e:
        print(f"Read error: {e}")
        return line,

    if len(df) <= last_row_count:
        return line,

    new_rows = df.iloc[last_row_count:]
    last_row_count = len(df)

    for _, row in new_rows.iterrows():
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

        voice_score = voice_analyzer.get_score() if voice_analyzer else 0.0
        raw_score = (visual_score * VISUAL_CHANNEL_WEIGHT) + (voice_score * VOICE_CHANNEL_WEIGHT)

        now = time.time()
        if start_time is None:
            start_time = now
        elapsed = now - start_time

        if not baseline_established:
            baseline_scores.append(raw_score)
            if elapsed >= BASELINE_DURATION_SECONDS:
                baseline_mean = np.mean(baseline_scores)
                baseline_established = True
                ax.set_title("Live Tension Monitor - PROTOTYPE")
                print(f"Baseline established: {baseline_mean:.3f} (from {len(baseline_scores)} frames)")
            continue

        relative_score = raw_score - baseline_mean
        tension_history.append(relative_score)
        smoothed = sum(tension_history) / len(tension_history)

        plot_timestamps.append(elapsed)
        plot_scores.append(smoothed)

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
        })

    if len(plot_timestamps) > 1:
        line.set_data(list(plot_timestamps), list(plot_scores))
        ax.set_xlim(max(0, plot_timestamps[0]), plot_timestamps[-1] + 1)
        y_min = min(plot_scores) - 0.2
        y_max = max(plot_scores) + 0.2
        if not (np.isnan(y_min) or np.isnan(y_max)):
            ax.set_ylim(y_min, y_max)

        # redraw the shaded zones so they always cover the current y-range
        for patch in list(ax.patches):
            patch.remove()
        draw_threshold_zones()

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
