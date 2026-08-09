import pandas as pd
import numpy as np
import time
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

CSV_PATH = r"C:\OpenFace\live_output\session1.csv"
LOG_PATH = r"C:\OpenFace\live_output\tension_log.csv"

TENSION_WEIGHTS = {
    "AU04_r": 0.3,
    "AU07_r": 0.2,
    "AU20_r": 0.3,
    "AU23_r": 0.2,
}

WINDOW_SIZE = 10
BASELINE_DURATION_SECONDS = 10

tension_history = deque(maxlen=WINDOW_SIZE)
plot_timestamps = deque(maxlen=200)
plot_scores = deque(maxlen=200)

baseline_scores = []
baseline_mean = None
baseline_established = False
start_time = None

last_row_count = 0
log_rows = []

def compute_tension(row):
    score = 0
    for au, weight in TENSION_WEIGHTS.items():
        val = row.get(au, np.nan)
        if pd.notna(val):
            score += val * weight
    return score

fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot([], [], color='crimson')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Tension Score (relative to baseline)")
ax.set_title("Live Tension Monitor - PROTOTYPE")
ax.axhline(0, color='gray', linestyle='--', linewidth=1)

def update_plot(frame_num):
    global last_row_count, baseline_mean, baseline_established, start_time

    try:
        df = pd.read_csv(CSV_PATH, low_memory=False)
        df.columns = df.columns.str.strip()

        # force AU columns to numeric, invalid entries become NaN
        for au in TENSION_WEIGHTS.keys():
            if au in df.columns:
                df[au] = pd.to_numeric(df[au], errors='coerce')
    except Exception as e:
        print(f"Read error: {e}")
        return line,

    if len(df) <= last_row_count:
        return line,

    new_rows = df.iloc[last_row_count:]
    last_row_count = len(df)

    for _, row in new_rows.iterrows():
        raw_score = compute_tension(row)
        if pd.isna(raw_score):
            continue  # skip rows with no valid AU data (e.g. face not detected)

        now = time.time()
        if start_time is None:
            start_time = now
        elapsed = now - start_time

        if not baseline_established:
            baseline_scores.append(raw_score)
            if elapsed >= BASELINE_DURATION_SECONDS:
                baseline_mean = np.mean(baseline_scores)
                baseline_established = True
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

    return line,

ani = animation.FuncAnimation(fig, update_plot, interval=500, cache_frame_data=False)

print(f"Calibrating baseline for {BASELINE_DURATION_SECONDS} seconds — please stay neutral...")
plt.show()

if log_rows:
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_PATH, index=False)
    print(f"Session log saved to {LOG_PATH} ({len(log_df)} rows)")
else:
    print("No data logged this session.")