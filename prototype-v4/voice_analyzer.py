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

#update .....

import threading
import time
import wave
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
        self.analysis_enabled = True  # overwritten by start()'s argument

        self.pitch_history = deque(maxlen=history_len)
        self.jitter_history = deque(maxlen=history_len)
        self.pause_ratio_history = deque(maxlen=history_len)
        self.intensity_std_history = deque(maxlen=history_len)

        self.latest_voice_score = 0.0
        self.latest_features = {}

        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

        # Full-session recording - separate from self._buffer above (which
        # the analysis loop repeatedly drains/resets every CHUNK_SECONDS).
        # Appending raw chunks to a list and concatenating once at the end
        # (in _save_recording) is cheap per-callback and avoids repeatedly
        # growing one big array on every audio callback for the whole
        # session's duration.
        self._record_path = None
        self._record_chunks = []
        self._first_callback_seen = False  # TEMPORARY diagnostic flag, see _audio_callback

        # Optional - set via set_transcriber() before/at start(). When set,
        # every audio chunk this analyzer captures is ALSO fed to it, so
        # live speech-to-text runs off the exact same mic stream instead of
        # opening a second, competing InputStream. None by default (no
        # behavior change for any existing caller that doesn't use this).
        self._transcriber = None

    # -----------------------------------------------------------------
    def set_transcriber(self, transcriber):
        """Attach a transcriber.Transcriber instance - every chunk this
        analyzer's audio callback receives from here on is also forwarded
        to it (see _audio_callback). Call this any time before stop();
        typically right after start(), from prototype_OpenFace_v3_2.py."""
        self._transcriber = transcriber

    # -----------------------------------------------------------------
    def start(self, record_path=None, analysis_enabled=True):
        """record_path: optional - if given, every audio sample captured
        this session is saved to this .wav path once stop() completes
        (mono, 16-bit PCM, at SAMPLE_RATE). None (default) disables
        recording entirely - no behavior change from before.

        analysis_enabled: whether the rolling pitch/jitter/pause-ratio
        stress scoring (_analyze_chunk, below) actually runs. Recording
        and analysis are independent - this is what lets "Always Record
        Audio" capture a full session even when "Enable Voice Detection"
        (the stress-analysis toggle) is off, and vice versa. Defaults to
        True so any existing caller that doesn't pass this keeps behaving
        exactly as before."""
        self._record_path = record_path
        self._record_chunks = []
        self.analysis_enabled = analysis_enabled
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        # _run()'s loop only notices self.running went False on its next
        # wake-up (every CHUNK_SECONDS - see _run below), and the
        # InputStream (and therefore _audio_callback) only actually closes
        # once that loop exits its `with` block. Joining here - rather
        # than returning immediately - guarantees the stream has fully
        # stopped and _record_chunks can't still be appended to from the
        # callback thread while we're about to read it below: without
        # this, stopping right after a fresh chunk would both lose up to
        # ~CHUNK_SECONDS of trailing audio AND risk reading
        # _record_chunks mid-append from the other thread.
        if self.thread is not None:
            self.thread.join(timeout=CHUNK_SECONDS + 1.0)
        self._save_recording()

    # -----------------------------------------------------------------
    def _save_recording(self):
        """Writes everything accumulated in _record_chunks to
        self._record_path as a standard mono 16-bit PCM .wav (stdlib
        `wave` module - no extra dependency beyond what's already
        imported). No-op if recording wasn't enabled (record_path was
        never set) or nothing was actually captured. Never raises - a
        failure here shouldn't take down session shutdown."""
        if not self._record_path:
            return

        with self._lock:
            chunks = self._record_chunks
            self._record_chunks = []

        if not chunks:
            print("Voice recording: no audio was captured this session - nothing to save.")
            return

        try:
            audio = np.concatenate(chunks)
            # sounddevice gives float32 samples in roughly [-1.0, 1.0] -
            # standard PCM .wav needs 16-bit integers, so scale and clip
            # before converting (clip guards against the rare sample that
            # exceeds +-1.0, which would otherwise wrap around as noise
            # instead of just clamping like real clipping would).
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
            with wave.open(self._record_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # bytes per sample = 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm16.tobytes())
            duration_s = len(audio) / SAMPLE_RATE
            print(f"Voice recording saved to {self._record_path} ({duration_s:.1f}s)")
        except Exception as e:
            print(f"Could not save voice recording: {e}")

    # -----------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        # TEMPORARY diagnostics, all safe to remove once the short-
        # recording issue is resolved:
        #
        # 1. try/except around the ENTIRE body, printing a full traceback.
        #    This did not exist before and is the most important addition
        #    here: sounddevice/PortAudio runs this callback on its own
        #    real-time thread via a C callback trampoline. If this
        #    function raises ANY uncaught exception, sounddevice silently
        #    aborts the whole input stream - no crash, no dialog, nothing
        #    printed anywhere - and _run()'s outer loop below has no way
        #    to notice, since it only checks self.running (which nothing
        #    sets to False when this happens). Recording (and analysis)
        #    would then simply stop dead, mid-session, with the rest of
        #    the app (OpenFace, tension scoring) completely unaffected -
        #    which matches the symptom exactly. Without this try/except,
        #    we had NO visibility into whether this is actually happening.
        #
        # 2. self._last_callback_time, updated on every successful
        #    callback - lets _run()'s loop (below) detect and print
        #    exactly when callbacks stopped arriving, even if no
        #    exception was involved (e.g., the stream was aborted by the
        #    OS/driver for a reason that doesn't surface as a Python
        #    exception at all).
        try:
            if not self._first_callback_seen:
                self._first_callback_seen = True
                print("DEBUG voice_analyzer: first audio callback received - stream is live.")
            if status:
                print(f"DEBUG voice_analyzer: callback status flag set: {status}")

            with self._lock:
                self._buffer = np.concatenate([self._buffer, indata[:, 0]])
                if self._record_path is not None:
                    self._record_chunks.append(indata[:, 0].copy())

            # Fed OUTSIDE self._lock, deliberately - Transcriber.feed() is
            # its own cheap, independently-locked buffer append (see
            # transcriber.py), and keeping it off VoiceAnalyzer's own lock
            # minimizes how long this realtime callback holds it.
            if self._transcriber is not None:
                self._transcriber.feed(indata[:, 0].copy())

            self._last_callback_time = time.time()
        except Exception:
            import traceback
            print("DEBUG voice_analyzer: EXCEPTION INSIDE AUDIO CALLBACK - "
                  "this is almost certainly why recording/analysis stopped "
                  "early. Full traceback:")
            traceback.print_exc()

    # -----------------------------------------------------------------
    def _run(self):
        chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)

        # TEMPORARY diagnostics - safe to remove once resolved. If
        # "InputStream opened" never prints, sd.InputStream(...) itself is
        # what's hanging/failing (most likely: a device conflict with
        # OpenFace's webcam, if they share one physical USB device, or a
        # PortAudio/driver-level problem opening the input device at all).
        print(f"DEBUG voice_analyzer: opening InputStream "
              f"(device={sd.default.device}, samplerate={SAMPLE_RATE})...")
        self._first_callback_seen = False
        self._last_callback_time = time.time()
        session_start = time.time()

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=self._audio_callback
        ):
            print("DEBUG voice_analyzer: InputStream opened successfully, entering analysis loop.")
            while self.running:
                time.sleep(CHUNK_SECONDS)

                # Heartbeat check: if nothing has arrived from the
                # callback in way longer than one chunk interval, the
                # stream has gone silent - print how far into the
                # session this happened (in seconds since start()), so
                # it can be compared directly against the on-screen
                # elapsed time / OpenFace's own session duration.
                silence = time.time() - self._last_callback_time
                if silence > CHUNK_SECONDS * 2:
                    elapsed_since_start = time.time() - session_start
                    print(f"DEBUG voice_analyzer: no audio callback received in "
                          f"{silence:.1f}s (now {elapsed_since_start:.1f}s into "
                          f"this session) - the input stream has likely died "
                          f"silently. See whether an exception was printed "
                          f"above around the {elapsed_since_start - silence:.1f}s mark.")

                with self._lock:
                    if len(self._buffer) < chunk_samples:
                        continue
                    chunk = self._buffer[-chunk_samples:].copy()
                    self._buffer = np.zeros(0, dtype=np.float32)

                # Skip the actual stress-scoring (pitch/jitter/pause-ratio
                # via parselmouth/Praat) when analysis is disabled - e.g.
                # a session with "Always Record Audio" on but "Enable
                # Voice Detection" off. The buffer is still drained above
                # either way, so recording keeps working regardless.
                if not self.analysis_enabled:
                    continue

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
