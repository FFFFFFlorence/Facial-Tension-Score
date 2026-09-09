"""
voice_analyzer.py
------------------
Captures microphone audio in rolling chunks and extracts acoustic features
associated with vocal stress/arousal in the speech science literature:
  - Pitch (F0) mean and variability
  - Jitter (cycle-to-cycle pitch instability / vocal tremor)
  - Pause ratio (proportion of silence vs speech - hesitation proxy)
  - Intensity (energy) variability

IMPORTANT: As with the facial scoring module, these acoustic features are
general correlates of arousal/stress reported in speech science research,
not a validated deception or lie-detection instrument. Voice can be tense
for many reasons unrelated to dishonesty (language barrier, general anxiety,
room acoustics, microphone quality). Treat as one more advisory signal only.
"""

import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import parselmouth
from parselmouth.praat import call

SAMPLE_RATE = 16000
CHUNK_SECONDS = 2.0          # analyze audio in 2-second rolling chunks
SILENCE_THRESHOLD_DB = -40   # below this = considered silence/pause

# Above this pause_ratio, a chunk is treated as "not currently speaking"
# rather than "hesitating while speaking". Sustained silence after calming
# down is a healthy, expected behavior - it should NOT keep pushing the
# tension score up. Only partial/intermittent pausing (someone still
# talking, but breaking up their speech) counts as a hesitation signal.
NEAR_TOTAL_SILENCE_THRESHOLD = 0.85


def pause_hesitation_score(pause_ratio):
    """
    Converts a raw pause_ratio (fraction of a chunk that was silent) into a
    hesitation signal. Raw pause_ratio increases both when someone hesitates
    mid-sentence AND when someone has simply stopped talking altogether -
    but only the former is actually tension-related. Near-total silence
    (barely any speech in the chunk at all) is zeroed out here so that
    "I calmed down and stopped talking" doesn't itself read as more tense.
    """
    if pause_ratio >= NEAR_TOTAL_SILENCE_THRESHOLD:
        return 0.0
    return pause_ratio


class VoiceAnalyzer:
    """
    Runs in its own thread, continuously capturing short audio chunks and
    producing a rolling voice_score (0+ range, higher = more stress-associated
    acoustic markers), analogous to the facial au_score.
    """

    def __init__(self, history_len=5):
        self.running = False
        self.thread = None

        self.pitch_history = deque(maxlen=history_len)
        self.jitter_history = deque(maxlen=history_len)
        self.pause_ratio_history = deque(maxlen=history_len)
        self.intensity_std_history = deque(maxlen=history_len)

        self.latest_voice_score = 0.0
        self.latest_features = {}

        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

    # -----------------------------------------------------------------
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    # -----------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        with self._lock:
            self._buffer = np.concatenate([self._buffer, indata[:, 0]])

    # -----------------------------------------------------------------
    def _run(self):
        chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=self._audio_callback
        ):
            while self.running:
                time.sleep(CHUNK_SECONDS)

                with self._lock:
                    if len(self._buffer) < chunk_samples:
                        continue
                    chunk = self._buffer[-chunk_samples:].copy()
                    self._buffer = np.zeros(0, dtype=np.float32)

                try:
                    self._analyze_chunk(chunk)
                except Exception as e:
                    print(f"Voice analysis error: {e}")

    # -----------------------------------------------------------------
    def _analyze_chunk(self, chunk):
        # Skip chunks that are essentially silent (no useful speech signal)
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < 1e-4:
            self.pause_ratio_history.append(1.0)  # fully silent chunk
            self._update_score()
            return

        snd = parselmouth.Sound(chunk, sampling_frequency=SAMPLE_RATE)

        # ---- Pitch (F0) ----
        pitch = snd.to_pitch()
        f0_values = pitch.selected_array["frequency"]
        f0_values = f0_values[f0_values > 0]  # remove unvoiced frames
        pitch_mean = float(np.mean(f0_values)) if len(f0_values) > 0 else 0.0

        # ---- Jitter (cycle-to-cycle pitch instability) ----
        jitter = 0.0
        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            if jitter is None or np.isnan(jitter):
                jitter = 0.0
        except Exception:
            jitter = 0.0

        # ---- Pause ratio (silence proportion within this chunk) ----
        intensity = snd.to_intensity()
        intensity_values = intensity.values[0]
        silent_frames = np.sum(intensity_values < SILENCE_THRESHOLD_DB)
        pause_ratio = float(silent_frames / len(intensity_values)) if len(intensity_values) > 0 else 0.0

        # ---- Intensity variability ----
        intensity_std = float(np.std(intensity_values)) if len(intensity_values) > 0 else 0.0

        self.pitch_history.append(pitch_mean)
        self.jitter_history.append(jitter)
        self.pause_ratio_history.append(pause_ratio)
        self.intensity_std_history.append(intensity_std)

        self._update_score()

    # -----------------------------------------------------------------
    def _update_score(self):
        """
        Combine features into a single voice_score. Like the facial AU
        weights, these weights are heuristic starting points, not validated
        coefficients from a calibration study.
        """
        pitch_var = float(np.std(self.pitch_history)) if len(self.pitch_history) > 1 else 0.0
        jitter_mean = float(np.mean(self.jitter_history)) if self.jitter_history else 0.0
        pause_mean_raw = float(np.mean(self.pause_ratio_history)) if self.pause_ratio_history else 0.0
        pause_mean = pause_hesitation_score(pause_mean_raw)
        intensity_var = float(np.mean(self.intensity_std_history)) if self.intensity_std_history else 0.0

        # normalize rough scales so no single feature dominates purely due to units
        score = (
            (pitch_var / 20.0) * 0.30 +      # pitch variability in Hz, scaled down
            (jitter_mean * 100) * 0.30 +     # jitter is a small fraction, scale up
            (pause_mean) * 0.20 +            # hesitation-adjusted, not raw silence
            (intensity_var / 10.0) * 0.20    # intensity std in dB, scaled down
        )

        self.latest_voice_score = score
        self.latest_features = {
            "pitch_mean": self.pitch_history[-1] if self.pitch_history else 0.0,
            "pitch_variability": pitch_var,
            "jitter": jitter_mean,
            "pause_ratio": pause_mean_raw,  # report the raw value for transparency in the UI
            "intensity_variability": intensity_var,
        }

    # -----------------------------------------------------------------
    @staticmethod
    def analyze_offline(samples, sample_rate):
        """
        Analyze a single pre-extracted audio chunk (numpy float32 array) and
        return the same feature dict as the live pipeline, without needing
        a live microphone stream. Used for prerecorded video analysis.
        """
        rms = np.sqrt(np.mean(samples ** 2)) if len(samples) > 0 else 0.0
        if rms < 1e-4 or len(samples) < sample_rate * 0.2:
            return {"pitch_mean": 0.0, "pitch_variability": 0.0, "jitter": 0.0,
                    "pause_ratio": 1.0, "intensity_variability": 0.0}

        snd = parselmouth.Sound(samples, sampling_frequency=sample_rate)

        pitch = snd.to_pitch()
        f0_values = pitch.selected_array["frequency"]
        f0_values = f0_values[f0_values > 0]
        pitch_mean = float(np.mean(f0_values)) if len(f0_values) > 0 else 0.0

        jitter = 0.0
        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            if jitter is None or np.isnan(jitter):
                jitter = 0.0
        except Exception:
            jitter = 0.0

        intensity = snd.to_intensity()
        intensity_values = intensity.values[0]
        silent_frames = np.sum(intensity_values < SILENCE_THRESHOLD_DB)
        pause_ratio = float(silent_frames / len(intensity_values)) if len(intensity_values) > 0 else 0.0
        intensity_std = float(np.std(intensity_values)) if len(intensity_values) > 0 else 0.0

        return {
            "pitch_mean": pitch_mean,
            "jitter": jitter,
            "pause_ratio": pause_ratio,
            "intensity_variability": intensity_std,
        }

    @staticmethod
    def score_from_rolling(pitch_hist, jitter_hist, pause_hist, intensity_hist):
        """Same weighting formula as the live _update_score, exposed for reuse offline."""
        pitch_var = float(np.std(pitch_hist)) if len(pitch_hist) > 1 else 0.0
        jitter_mean = float(np.mean(jitter_hist)) if jitter_hist else 0.0
        pause_mean_raw = float(np.mean(pause_hist)) if pause_hist else 0.0
        pause_mean = pause_hesitation_score(pause_mean_raw)
        intensity_var = float(np.mean(intensity_hist)) if intensity_hist else 0.0
        return (
            (pitch_var / 20.0) * 0.30 +
            (jitter_mean * 100) * 0.30 +
            (pause_mean) * 0.20 +
            (intensity_var / 10.0) * 0.20
        )

    # -----------------------------------------------------------------
    def get_score(self):
        return self.latest_voice_score

    def get_features(self):
        return dict(self.latest_features)
