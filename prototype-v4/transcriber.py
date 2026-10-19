"""
transcriber.py
---------------
Speech-to-text for the live interview session, with two interchangeable
backends (set TRANSCRIBER_BACKEND below):

  "openai" (default) - OpenAI's gpt-live-transcribe over the Realtime API
      (WebSocket). NOTE: the audio of the session is SENT TO OPENAI. Needs
      the OPENAI_API_KEY environment variable and `pip install
      websocket-client`.
  "local" - faster-whisper (CTranslate2 Whisper) running entirely on this
      machine; nothing leaves it.

(AssemblyAI Universal-3.5 Pro Realtime was considered but its documented
language list does not include Indonesian, so it isn't offered here.)

One background transcription feeds TWO things:

1. LIVE display - a short, frequently-refreshed snippet of "what's being
   said right now", for the optional on-screen scrolling panel
   (get_live_text()).
2. POST-SESSION CONTEXT - once the session ends, a heuristic pass over the
   full transcript (get_context_for_episode()) picks out the stretch of
   conversation around each ELEVATED/HIGH tension episode, bounded by
   natural pauses in speech rather than a fixed +-N-second clock window -
   so what's returned is "the question/answer exchange this spike happened
   during", not an arbitrary slice that might cut a sentence in half.

CHUNKING: audio is fed to Whisper on REAL speech/pause boundaries (a
lightweight RMS-based voice-activity split done right here, not fixed
clock-time windows) - see feed()/_maybe_flush_pending() below. This matters
for two separate reasons:
  - Accuracy: handing Whisper a clock-chopped 5-second window routinely
    slices a word or sentence in half at the boundary, which is a well-
    known cause of garbled/hallucinated text. Handing it one coherent
    utterance at a time (ending at an actual pause) avoids that.
  - Correct pause detection: the post-session context heuristic finds
    natural breaks by looking for gaps between recognized segments. If
    those segments came from arbitrary fixed-size windows, the gap between
    one window's end and the next window's start is close to 0 even when
    there was no real pause there at all, which silently merges unrelated
    stretches of conversation into one giant "exchange". Chunking on real
    silence makes those inter-chunk gaps meaningful pause boundaries.

No LLM involved anywhere in this file - purely local ASR (Whisper) plus a
simple energy-based pause/question-marker heuristic.
"""

import base64
import json
import os
import queue
import re
import threading
import time
from collections import deque

import numpy as np

SAMPLE_RATE = 16000  # must match voice_analyzer.SAMPLE_RATE - same mic stream feeds both

TRANSCRIBER_BACKEND = "assemblyai_async"
# "assemblyai_async" - NO live text. The session's recorded audio is sent to AssemblyAI
#                      when you press Stop; the transcript (used for the post-session context
#                      list) arrives typically within a minute. Supports Indonesian (via
#                      the universal-2 model); AssemblyAI's REALTIME models do not.
# "openai"           - live, via OpenAI gpt-live-transcribe (needs paid OpenAI credits).
# "local"            - live, offline faster-whisper.

# --- AssemblyAI async (pre-recorded) settings ---
ASSEMBLYAI_API_KEY_ENV = "ASSEMBLYAI_API_KEY"
ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com"
ASSEMBLYAI_SPEECH_MODELS = ["universal-2"]   # the model that lists Indonesian
ASSEMBLYAI_LANGUAGE = "id"
ASSEMBLYAI_POLL_SECONDS = 3
ASSEMBLYAI_TIMEOUT_SECONDS = 900
ASYNC_SEGMENT_SPLIT_SECONDS = 0.5   # word-timestamp gaps this long start a new segment
ASYNC_PAUSE_GAP_SECONDS = 2.0       # context expands until a pause this long (keeps a question with its answer)
MAX_CONTEXT_SECONDS = 30.0          # never let one episode's context grow beyond this span

# --- OpenAI gpt-live-transcribe settings (only used when backend == "openai") ---
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"   # read from the environment - never hard-code the key
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
OPENAI_MODEL = "gpt-live-transcribe"
OPENAI_LANGUAGES = ["id"]               # ISO 639-1 hint(s); Bahasa Indonesia
OPENAI_DELAY = None                     # optional: "minimal"/"low"/"medium"/"high"/"xhigh" - left off by default
                                         # until the basic connection is confirmed working on your machine
OPENAI_PROMPT = None                    # optional context hint, e.g. "Wawancara dalam Bahasa Indonesia."
OPENAI_SAMPLE_RATE = 24000              # this model wants 24 kHz mono PCM16; we capture at 16 kHz and upsample
OPENAI_SEND_CHUNK_SECONDS = 0.5

# ---------------------------------------------------------------------------
# Speech/pause (voice-activity) chunking - heuristic, not a trained VAD, but
# this is what decides where one Whisper job ends and the next begins, so
# it directly drives both transcription accuracy and pause-boundary
# correctness (see module docstring above).
# ---------------------------------------------------------------------------
FRAME_SECONDS = 0.10            # granularity of the speech/silence check
SILENCE_RMS_THRESHOLD = 0.012   # raw float32 amplitude RMS below this counts as
                                 # silence. Mic gain and room noise vary a lot -
                                 # if the live panel rarely updates, lower this;
                                 # if it's transcribing room noise, raise it.
SILENCE_HANG_SECONDS = 0.7      # this much continuous silence after speech ends
                                 # an utterance and triggers transcription of it
MIN_UTTERANCE_SECONDS = 0.5     # shorter than this (a cough, a click) is dropped,
                                 # never sent to Whisper
MAX_UTTERANCE_SECONDS = 18.0    # hard cap - flush even mid-speech after this long,
                                 # so one person talking continuously doesn't stall
                                 # the live view or grow the buffer without bound

# A gap this long between two consecutive recognized segments is treated as a
# natural break in the conversation - the boundary the post-session context
# matching expands outward to. Kept just under SILENCE_HANG_SECONDS so every
# real utterance-to-utterance boundary above reliably counts as a pause here.
PAUSE_GAP_SECONDS = 0.5

# The live panel shows roughly the last this-many-seconds of recognized
# speech (not a fixed line/character count), so it reads as "what's being
# said right now" regardless of how fast or slow people are talking.
LIVE_DISPLAY_SECONDS = 20.0

# "small" is noticeably more accurate on Indonesian speech than tiny/base
# while still runnable near-real-time on a normal CPU with int8
# quantization. Bump to "medium" if accuracy still isn't good enough and
# your CPU can keep up (slower); drop to "base" if the live panel falls
# behind (printed a growing backlog in the console).
WHISPER_MODEL_SIZE = "small"
WHISPER_LANGUAGE = "id"  # Bahasa Indonesia - set to None to let Whisper auto-detect instead

# Cheap textual heuristic only (not NLP) - just flags a transcript segment
# as "likely a question" for the post-session context readout.
_QUESTION_MARKERS = (
    "apa", "apakah", "siapa", "kenapa", "mengapa", "kapan", "dimana",
    "di mana", "bagaimana", "gimana", "berapa", "mana", "kok",
)


def _looks_like_question(text):
    t = text.strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    first_word = re.split(r"[\s,]+", t, maxsplit=1)[0]
    return first_word in _QUESTION_MARKERS


def group_episode_contexts(transcriber, episodes):
    """For every ELEVATED/HIGH episode, finds its conversation context, then
    MERGES episodes that landed in the same context (same time span) into one
    group, so a stretch of speech that triggered several rises is shown once
    with all of its trigger times. `episodes` is {"ELEVATED": [{"start":..,
    "end":..}, ...], "HIGH": [...]}. Returns groups in chronological order:
    {"levels": ["ELEVATED", ...], "times": {"ELEVATED": [20.1, 26.4], ...},
     "ctx": context dict or None}."""
    items = []
    for lvl in ("ELEVATED", "HIGH"):
        for ep in episodes.get(lvl, []):
            items.append((ep["start"], lvl,
                          transcriber.get_context_for_episode(ep["start"], ep["end"])))
    items.sort(key=lambda x: x[0])

    groups = []
    for t0, lvl, ctx in items:
        key = None if ctx is None else (round(ctx["context_start"], 1), round(ctx["context_end"], 1))
        grp = next((g for g in groups if key is not None and g["key"] == key), None)
        if grp is None:
            grp = {"key": key, "ctx": ctx, "times": {}}
            groups.append(grp)
        grp["times"].setdefault(lvl, []).append(t0)
    for g in groups:
        g["levels"] = [l for l in ("ELEVATED", "HIGH") if l in g["times"]]
    return groups


def _get_api_key(name):
    """Reads an API key from the environment. If this terminal/IDE was
    started BEFORE the variable was saved with setx (so it never received
    it), falls back to reading the saved per-user value straight from the
    Windows registry - so no restart is needed."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import winreg  # Windows only
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            value, _type = winreg.QueryValueEx(k, name)
            return value or None
    except Exception:
        return None


class Transcriber:
    """
    Feed it raw mono float32 audio samples at SAMPLE_RATE (the same stream
    VoiceAnalyzer already captures - see voice_analyzer.set_transcriber)
    via feed(); it splits that stream into real speech utterances (silence-
    bounded, not clock-bounded), runs local Whisper transcription on each
    one on a background thread, and builds up a timestamped transcript
    (elapsed seconds since this Transcriber's own start()).
    """

    def __init__(self):
        self.running = False
        self._model = None
        self._worker_thread = None
        self._job_queue = queue.Queue()

        # Small holding buffer so feed() (which can be called with any
        # chunk size) always analyzes exact FRAME_SECONDS blocks.
        self._frame_holdover = np.zeros(0, dtype=np.float32)
        self._frame_cursor_elapsed = 0.0  # elapsed seconds of the next sample in _frame_holdover

        # The utterance currently being accumulated (empty/None when we're
        # between utterances, waiting for speech to start).
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_start = None
        self._silence_run = 0.0

        self._state_lock = threading.Lock()  # guards everything above

        self._segments = []  # [{"start": float, "end": float, "text": str}, ...] elapsed seconds, in order
        self._segments_lock = threading.Lock()

        self.session_start = None
        self._load_error = None

        # OpenAI backend bookkeeping. The model returns no timestamps, so
        # each committed utterance's (start, end) - known from our own
        # silence chunking - is remembered and matched back to the
        # transcript the server returns for it (by item_id).
        self._meta_lock = threading.Lock()
        self._commit_meta = deque()   # (start, end) per commit sent, awaiting the server's "committed" event
        self._item_meta = {}          # item_id -> (start, end)
        self._unmapped = deque()      # fallback if "committed" carries no item_id: match by order
        self._outstanding = 0         # commits sent whose transcript hasn't arrived yet
        self._last_server_error = None  # last error/close reason reported by the server, for diagnostics
        self._pause_gap = PAUSE_GAP_SECONDS  # boundary used by get_context_for_episode

    # -----------------------------------------------------------------
    def start(self):
        if TRANSCRIBER_BACKEND == "assemblyai_async":
            # Nothing to do live - the recording is transcribed after Stop
            # (see transcribe_file). feed()/stop() are no-ops in this mode.
            self.session_start = time.time()
            self._segments = []
            self._load_error = None
            self._pause_gap = ASYNC_PAUSE_GAP_SECONDS
            self.running = False
            return
        self.session_start = time.time()
        self._frame_holdover = np.zeros(0, dtype=np.float32)
        self._frame_cursor_elapsed = 0.0
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_start = None
        self._silence_run = 0.0
        self._segments = []
        self._load_error = None
        self.running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self, timeout=25.0):
        """Flushes whatever utterance is still in progress (as a final
        job, even if it never hit a silence boundary), then blocks (up to
        timeout) until the worker has processed everything queued, so the
        transcript is complete before the caller reads get_segments()/
        get_context_for_episode(). Intended to be called from a background
        thread (same pattern as VoiceAnalyzer.stop() already used
        elsewhere in this project) - it can take a while if several
        utterances are still queued up."""
        if TRANSCRIBER_BACKEND == "assemblyai_async":
            return
        with self._state_lock:
            self._flush_pending_locked(force=True)

        self.running = False
        self._job_queue.put(None)  # sentinel - worker stops after draining everything queued before it
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)

    # -----------------------------------------------------------------
    def feed(self, samples):
        """Called from the realtime audio-callback thread (via
        VoiceAnalyzer._audio_callback) with each new chunk of raw mono
        float32 samples. Splits the stream into FRAME_SECONDS blocks,
        classifies each as speech/silence by RMS, and accumulates/flushes
        utterances accordingly. Cheap per call - no Whisper inference ever
        happens on this thread, only on the worker."""
        if not self.running or self.session_start is None:
            return

        frame_samples = int(FRAME_SECONDS * SAMPLE_RATE)
        with self._state_lock:
            self._frame_holdover = np.concatenate([self._frame_holdover, samples])
            while len(self._frame_holdover) >= frame_samples:
                frame = self._frame_holdover[:frame_samples]
                frame_start_elapsed = self._frame_cursor_elapsed
                self._frame_holdover = self._frame_holdover[frame_samples:]
                self._frame_cursor_elapsed += FRAME_SECONDS

                rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
                is_speech = rms >= SILENCE_RMS_THRESHOLD

                if is_speech:
                    if self._pending_start is None:
                        self._pending_start = frame_start_elapsed
                        self._pending = frame.copy()
                    else:
                        self._pending = np.concatenate([self._pending, frame])
                    self._silence_run = 0.0
                elif self._pending_start is not None:
                    # mid-utterance silence - keep it (natural trailing
                    # breath/pause), but track how long it's run
                    self._pending = np.concatenate([self._pending, frame])
                    self._silence_run += FRAME_SECONDS
                    if self._silence_run >= SILENCE_HANG_SECONDS:
                        self._flush_pending_locked()
                # else: still waiting for speech to start - nothing to buffer

                if (self._pending_start is not None
                        and (frame_start_elapsed + FRAME_SECONDS - self._pending_start) >= MAX_UTTERANCE_SECONDS):
                    self._flush_pending_locked()

    def _flush_pending_locked(self, force=False):
        """Must be called with self._state_lock held. Queues the current
        _pending utterance for transcription (if it's long enough to be
        worth it) and resets the accumulator."""
        if self._pending_start is not None:
            # Trim the trailing silence that triggered this flush (keeping a
            # short 0.1s tail) so the utterance's end time is where speech
            # actually stopped - otherwise every real pause would look
            # ~SILENCE_HANG_SECONDS shorter than it was, and the
            # pause-boundary matching would miss real boundaries.
            trim = max(0, int((self._silence_run - 0.1) * SAMPLE_RATE))
            audio = self._pending[:len(self._pending) - trim] if trim else self._pending
            duration = len(audio) / SAMPLE_RATE
            if duration >= MIN_UTTERANCE_SECONDS or (force and duration > 0):
                self._job_queue.put((self._pending_start, audio.copy()))
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_start = None
        self._silence_run = 0.0

    # -----------------------------------------------------------------
    def _ensure_model_loaded(self):
        if self._model is not None or self._load_error is not None:
            return
        try:
            from faster_whisper import WhisperModel  # pip install faster-whisper
            print(f"Transcriber: loading Whisper model '{WHISPER_MODEL_SIZE}' "
                  f"(first use only, may take a moment and needs internet once "
                  f"to download - fully offline after that)...")
            self._model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            print("Transcriber: model loaded.")
        except Exception as e:
            self._load_error = str(e)
            print(f"Transcriber: could not load Whisper model, transcription "
                  f"unavailable this session ({e})")

    def _worker_loop(self):
        if TRANSCRIBER_BACKEND == "openai":
            self._openai_worker_loop()
            return
        self._ensure_model_loaded()
        while True:
            job = self._job_queue.get()
            if job is None:
                break
            window_start, chunk = job
            if self._model is None:
                continue  # load failed earlier - drain the queue without transcribing
            try:
                self._transcribe_chunk(window_start, chunk)
            except Exception as e:
                print(f"Transcriber: error transcribing chunk at {window_start:.1f}s: {e}")

    # ---------------- AssemblyAI async (post-session) backend ----------------
    def transcribe_file(self, wav_path, time_offset=0.0):
        """Blocking - run from a background thread. Uploads the session's
        recorded .wav to AssemblyAI, waits for the transcript, and fills
        self._segments from its word timestamps. time_offset (seconds) is
        how much later the tension timeline's t=0 started than the
        recording's t=0; it is subtracted so transcript times line up with
        the episode times."""
        try:
            import requests  # pip install requests
        except ImportError:
            self._load_error = "pip install requests"
            return
        key = _get_api_key(ASSEMBLYAI_API_KEY_ENV)
        if not key:
            self._load_error = f"variabel {ASSEMBLYAI_API_KEY_ENV} belum di-set"
            return
        if not wav_path or not os.path.exists(wav_path):
            self._load_error = "file rekaman suara tidak ditemukan (tidak ada audio tersimpan)"
            return

        headers = {"authorization": key}

        def check(resp, what):
            if resp.status_code >= 400:
                raise RuntimeError(f"{what}: HTTP {resp.status_code} {resp.text[:300]}")

        try:
            print(f"Transcriber: uploading {wav_path} to AssemblyAI...")
            with open(wav_path, "rb") as f:
                r = requests.post(ASSEMBLYAI_BASE_URL + "/v2/upload", headers=headers, data=f, timeout=600)
            check(r, "upload")
            audio_url = r.json()["upload_url"]

            r = requests.post(ASSEMBLYAI_BASE_URL + "/v2/transcript", headers=headers, timeout=60, json={
                "audio_url": audio_url,
                "speech_models": ASSEMBLYAI_SPEECH_MODELS,
                "language_code": ASSEMBLYAI_LANGUAGE,
            })
            check(r, "membuat transkrip")
            tid = r.json()["id"]

            deadline = time.time() + ASSEMBLYAI_TIMEOUT_SECONDS
            while True:
                r = requests.get(f"{ASSEMBLYAI_BASE_URL}/v2/transcript/{tid}", headers=headers, timeout=60)
                check(r, "cek status")
                t = r.json()
                if t.get("status") == "completed":
                    break
                if t.get("status") == "error":
                    raise RuntimeError(t.get("error", "error tidak diketahui"))
                if time.time() > deadline:
                    raise RuntimeError("waktu tunggu habis")
                time.sleep(ASSEMBLYAI_POLL_SECONDS)
        except Exception as e:
            self._load_error = str(e)
            print(f"Transcriber: AssemblyAI transcription failed: {e}")
            return

        # Build segments from word timestamps (milliseconds), splitting at pauses.
        segments, cur = [], None
        for w in t.get("words") or []:
            ws_, we_ = w["start"] / 1000.0 - time_offset, w["end"] / 1000.0 - time_offset
            if cur is not None and ws_ - cur["end"] < ASYNC_SEGMENT_SPLIT_SECONDS:
                cur["end"] = we_
                cur["text"] += " " + w["text"]
            else:
                if cur is not None:
                    segments.append(cur)
                cur = {"start": ws_, "end": we_, "text": w["text"]}
        if cur is not None:
            segments.append(cur)
        with self._segments_lock:
            self._segments = segments
        print(f"Transcriber: AssemblyAI transcript ready ({len(segments)} segments).")

    # ---------------- OpenAI gpt-live-transcribe backend ----------------
    @staticmethod
    def _to_pcm16_24k_b64_chunks(chunk):
        n_out = int(len(chunk) * OPENAI_SAMPLE_RATE / SAMPLE_RATE)
        if n_out < 2 or len(chunk) < 2:
            return []
        resampled = np.interp(np.linspace(0, len(chunk) - 1, n_out), np.arange(len(chunk)), chunk)
        pcm = (np.clip(resampled, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        step = int(OPENAI_SEND_CHUNK_SECONDS * OPENAI_SAMPLE_RATE) * 2  # bytes, always even
        return [base64.b64encode(pcm[i:i + step]).decode("ascii") for i in range(0, len(pcm), step)]

    def _openai_worker_loop(self):
        def drain():
            while self._job_queue.get() is not None:
                pass

        try:
            import websocket  # pip install websocket-client
        except ImportError:
            self._load_error = "pip install websocket-client"
            print("Transcriber: websocket-client is not installed - run: pip install websocket-client")
            drain()
            return

        key = _get_api_key(OPENAI_API_KEY_ENV)
        if not key:
            self._load_error = f"variabel {OPENAI_API_KEY_ENV} belum di-set"
            print(f"Transcriber: environment variable {OPENAI_API_KEY_ENV} is not set.")
            drain()
            return

        try:
            ws = websocket.create_connection(
                OPENAI_REALTIME_URL, header=[f"Authorization: Bearer {key}"], timeout=20)
            ws.settimeout(None)
            transcription_cfg = {"model": OPENAI_MODEL}
            if OPENAI_LANGUAGES:
                transcription_cfg["languages"] = OPENAI_LANGUAGES
            if OPENAI_DELAY:
                transcription_cfg["delay"] = OPENAI_DELAY
            if OPENAI_PROMPT:
                transcription_cfg["prompt"] = OPENAI_PROMPT
            ws.send(json.dumps({
                "type": "session.update",
                "session": {"type": "transcription", "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": OPENAI_SAMPLE_RATE},
                    "transcription": transcription_cfg,
                    "turn_detection": None,
                }}},
            }))
            print("Transcriber: connected to OpenAI gpt-live-transcribe.")
        except Exception as e:
            self._load_error = f"koneksi OpenAI gagal: {e}"
            print(f"Transcriber: could not connect to OpenAI ({e})")
            drain()
            return

        receiver = threading.Thread(target=self._openai_receiver, args=(ws,), daemon=True)
        receiver.start()

        while True:
            job = self._job_queue.get()
            if job is None:
                break
            start, chunk = job
            if self._load_error:
                continue
            try:
                pieces = self._to_pcm16_24k_b64_chunks(chunk)
                if not pieces:
                    continue
                for piece in pieces:
                    ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": piece}))
                with self._meta_lock:
                    self._commit_meta.append((start, start + len(chunk) / SAMPLE_RATE))
                    self._outstanding += 1
                ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            except Exception as e:
                time.sleep(0.7)  # give the receiver a moment to capture the server's own reason
                reason = self._last_server_error or str(e)
                self._load_error = f"OpenAI menutup koneksi: {reason}"
                print(f"Transcriber: OpenAI connection lost - server said: {reason}")

        # stop(): wait (bounded) for transcripts of everything already committed
        deadline = time.time() + 20.0
        while time.time() < deadline:
            with self._meta_lock:
                if self._outstanding <= 0:
                    break
            time.sleep(0.2)
        try:
            ws.close()
        except Exception:
            pass

    def _openai_receiver(self, ws):
        while True:
            try:
                opcode, data = ws.recv_data()
            except Exception as e:
                if self._last_server_error is None:
                    self._last_server_error = f"koneksi terputus ({e})"
                break
            if opcode == 8:  # close frame - carries a status code + reason from the server
                code = int.from_bytes(data[:2], "big") if len(data) >= 2 else None
                reason = data[2:].decode("utf-8", "replace") if len(data) > 2 else ""
                self._last_server_error = f"close code {code}: {reason}".strip()
                print(f"Transcriber: server closed the connection - {self._last_server_error}")
                break
            if opcode != 1:
                continue
            raw = data
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            etype = ev.get("type")

            if etype == "input_audio_buffer.committed":
                with self._meta_lock:
                    meta = self._commit_meta.popleft() if self._commit_meta else None
                    if meta is not None:
                        if ev.get("item_id"):
                            self._item_meta[ev["item_id"]] = meta
                        else:
                            self._unmapped.append(meta)

            elif etype == "conversation.item.input_audio_transcription.completed":
                text = (ev.get("transcript") or "").strip()
                with self._meta_lock:
                    meta = self._item_meta.pop(ev.get("item_id"), None)
                    if meta is None and self._unmapped:
                        meta = self._unmapped.popleft()
                    self._outstanding = max(0, self._outstanding - 1)
                if meta is not None and text:
                    with self._segments_lock:
                        self._segments.append({"start": meta[0], "end": meta[1], "text": text})

            elif etype == "error":
                err = ev.get("error", ev)
                msg = err.get("message") if isinstance(err, dict) else None
                self._last_server_error = msg or str(err)
                print(f"Transcriber: OpenAI error event: {err}")
                with self._meta_lock:  # don't let a failed commit stall stop() forever
                    self._outstanding = max(0, self._outstanding - 1)

    def _transcribe_chunk(self, window_start, chunk):
        segments, _info = self._model.transcribe(
            chunk, language=WHISPER_LANGUAGE,
            beam_size=5,                      # faster-whisper's own default - noticeably
                                               # more accurate than greedy (beam_size=1);
                                               # affordable now that each job is one real
                                               # utterance instead of a fixed 5s window
            vad_filter=True,                  # extra safety net inside the chunk itself
                                               # (trims any residual leading/trailing
                                               # silence our own RMS split let through)
            condition_on_previous_text=False, # avoids one bad/hallucinated segment
                                               # dragging the rest of the same chunk
                                               # down with it
        )
        new_segments = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            new_segments.append({
                "start": window_start + seg.start,
                "end": window_start + seg.end,
                "text": text,
            })
        if not new_segments:
            return
        with self._segments_lock:
            self._segments.extend(new_segments)

    # -----------------------------------------------------------------
    def get_live_text(self):
        """Last ~LIVE_DISPLAY_SECONDS of recognized speech, as one block of
        text - what the live scrolling panel shows. Empty string if
        nothing's been recognized yet (still buffering, model still
        loading, or genuine silence)."""
        if self._load_error:
            return f"(Transkripsi tidak tersedia: {self._load_error})"
        with self._segments_lock:
            if not self._segments:
                return ""
            cutoff = self._segments[-1]["end"] - LIVE_DISPLAY_SECONDS
            recent = [s["text"] for s in self._segments if s["end"] >= cutoff]
        return " ".join(recent)

    def get_segments(self):
        with self._segments_lock:
            return list(self._segments)

    # -----------------------------------------------------------------
    def get_context_for_episode(self, ep_start, ep_end, pad_seconds=1.0):
        """
        Heuristic (no LLM) "what was being said" context for one
        ELEVATED/HIGH episode spanning [ep_start, ep_end] elapsed seconds.

        Rather than a fixed time window, this walks outward from whichever
        recognized segments overlap the episode to the nearest natural
        pauses in speech on each side (gaps >= PAUSE_GAP_SECONDS between
        consecutive segments - reliable now that chunking itself happens
        on real silence, see the module docstring) - so what's returned is
        the whole question-and-answer exchange the spike happened during,
        not an arbitrary N-seconds-before/after slice.

        Returns None if nothing was transcribed anywhere near this episode
        (transcription was off, or it was a silent/non-verbal moment).
        """
        segments = self.get_segments()
        if not segments:
            return None

        overlapping = [
            i for i, s in enumerate(segments)
            if s["end"] >= ep_start - pad_seconds and s["start"] <= ep_end + pad_seconds
        ]
        if not overlapping:
            # no segment literally overlaps the episode window (e.g. the spike
            # happened in a brief silence just after the relevant remark) -
            # fall back to whichever segment is closest in time
            nearest_i = min(
                range(len(segments)),
                key=lambda i: min(abs(segments[i]["start"] - ep_start), abs(segments[i]["end"] - ep_end)),
            )
            overlapping = [nearest_i]

        lo, hi = overlapping[0], overlapping[-1]

        gap = self._pause_gap
        while (lo > 0 and (segments[lo]["start"] - segments[lo - 1]["end"]) < gap
               and segments[hi]["end"] - segments[lo - 1]["start"] <= MAX_CONTEXT_SECONDS):
            lo -= 1
        while (hi < len(segments) - 1 and (segments[hi + 1]["start"] - segments[hi]["end"]) < gap
               and segments[hi + 1]["end"] - segments[lo]["start"] <= MAX_CONTEXT_SECONDS):
            hi += 1

        exchange = segments[lo:hi + 1]
        text = " ".join(s["text"] for s in exchange)
        has_question = any(_looks_like_question(s["text"]) for s in exchange)

        return {
            "context_start": exchange[0]["start"],
            "context_end": exchange[-1]["end"],
            "text": text,
            "has_question": has_question,
        }
