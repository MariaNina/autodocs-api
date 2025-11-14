# server.py
from __future__ import annotations

import os
import uuid
import threading
import requests
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from flask import Flask, request, jsonify
from waitress import serve

from pydub import AudioSegment

# OpenAI (ChatGPT) API
from openai import OpenAI

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# =========================
# Config (ENV overrides)
# =========================

FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS", "service-account.json")
FIREBASE_DB_URL = os.getenv(
    "FIREBASE_DB_URL",
    "https://autodocs-web-default-rtdb.asia-southeast1.firebasedatabase.app/",
)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# ---- OpenAI models ----
# NOTE: OPENAI_API_KEY is loaded later from ENV or Firebase RTDB (config/open_api)
# Transcription + diarization model
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-transcribe-diarize")
# Text model for summarization / translation / language detection
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4o-mini")

# Summarization tuning (for GLOBAL summary only)
SUMMARY_MAX_LEN = int(os.getenv("SUMMARY_MAX_LEN", "180"))
SUMMARY_MAX_INPUT_CHARS = int(os.getenv("SUMMARY_MAX_INPUT_CHARS", "16000"))

# Translation (EN <-> TL)
TRANSLATE_EN_TL = os.getenv("TRANSLATE_EN_TL", "1") not in ("0", "false", "False")
CONTENT_TRANSLATE_MAX_CHARS = int(os.getenv("CONTENT_TRANSLATE_MAX_CHARS", "24000"))
TRANSLATE_CHUNK_CHARS = int(os.getenv("TRANSLATE_CHUNK_CHARS", "3500"))

# Per-turn translation & full-content translation flags
ENABLE_TURN_TRANSLATION = os.getenv("ENABLE_TURN_TRANSLATION", "1") not in ("0", "false", "False")
ENABLE_CONTENT_TRANSLATION = os.getenv("ENABLE_CONTENT_TRANSLATION", "0") not in ("0", "false", "False")

# Upload limits
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "1500"))  # /transcribe max upload size
DOWNLOAD_TIMEOUT_SEC = int(os.getenv("DOWNLOAD_TIMEOUT_SEC", "600"))

# Turn merging
SAME_SPEAKER_GAP = float(os.getenv("SAME_SPEAKER_GAP", "0.6"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# =========================
# Firebase
# =========================

cred = credentials.Certificate(FIREBASE_CREDENTIALS)
firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})


def load_openai_api_key() -> str:
    """
    Priority:
    1) Environment variable OPENAI_API_KEY
    2) Firebase RTDB at /config/open_api
    """
    # 1) ENV first
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        logging.info("Using OPENAI_API_KEY from environment")
        return env_key.strip()

    # 2) Fallback: Firebase Realtime Database
    try:
        logging.info("Fetching OPENAI_API_KEY from Firebase RTDB at /config/open_api…")
        val = db.reference("config/open_api").get()
        if isinstance(val, str) and val.strip():
            return val.strip()
        logging.warning("No OpenAI key found at /config/open_api in RTDB")
    except Exception as e:
        logging.warning(f"Failed to load OpenAI key from RTDB: {e}")

    raise RuntimeError(
        "OPENAI_API_KEY is not set in ENV and not found in RTDB (/config/open_api)."
    )


# =========================
# OpenAI client
# =========================

OPENAI_API_KEY = load_openai_api_key()
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Flask app
# =========================

app = Flask(__name__)

# =========================
# Helpers
# =========================

ALLOWED_EXT = {
    ".m4a",
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".aac",
    ".wma",
    ".mp4",
    ".webm",
}


def safe_filename(name: str) -> str:
    base = re.sub(r"[^\w\-.]+", "_", (name or "").strip())[:120]
    return base or f"{uuid.uuid4().hex}.bin"


def download_to_file(url: str, dest_path: str, timeout: int = DOWNLOAD_TIMEOUT_SEC) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)


def hhmmss(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def time_label(start: float, end: float) -> str:
    dur = max(0.0, end - start)
    dur_str = f"{dur:.1f}s" if dur < 10 else f"{int(round(dur))}s"
    return f"{hhmmss(start)}–{hhmmss(end)} ({dur_str})"


def update_status(ref, status: str, extra: Optional[Dict[str, Any]] = None):
    payload = {"status": status}
    if extra:
        payload.update(extra)
    try:
        ref.update(payload)
    except Exception as e:
        logging.warning(f"status update failed: {e}")


def convert_to_wav(src_path: str, dst_path: str) -> Tuple[float, int]:
    """
    Convert input to WAV 16-bit PCM 16 kHz mono.
    You *could* skip this and send original file to OpenAI,
    but this keeps behaviour similar to your old server.
    """
    audio = AudioSegment.from_file(src_path)
    duration_sec = len(audio) / 1000.0
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio.export(dst_path, format="wav", parameters=["-acodec", "pcm_s16le"])
    return duration_sec, 1


def split_into_chunks_text(text: str, max_chars: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    buf: List[str] = []
    cur = 0

    for sent in re.split(r"(?<=[\.\!\?\n])\s+", text):
        snt = sent.strip()
        if not snt:
            continue
        if cur + len(snt) + 1 > max_chars and buf:
            parts.append(" ".join(buf))
            buf = [snt]
            cur = len(snt) + 1
        else:
            buf.append(snt)
            cur += len(snt) + 1

    if buf:
        parts.append(" ".join(buf))
    return parts


# =========================
# OpenAI-based NLP helpers
# =========================

def detect_language_code(text: str) -> str:
    """
    Detect whether the transcript is mainly English or Tagalog/Filipino.
    Returns: "en", "tl", or "unknown".
    """
    snippet = (text or "").strip()
    if not snippet:
        return "unknown"
    snippet = snippet[:4000]

    try:
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            temperature=0.0,
            max_tokens=4,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a language detector. "
                        "Determine if the user text is primarily in English or Tagalog/Filipino.\n"
                        "Reply with ONLY one of these codes:\n"
                        "en\n"
                        "tl\n"
                        "unknown"
                    ),
                },
                {"role": "user", "content": snippet},
            ],
        )
        code = (resp.choices[0].message.content or "").strip().lower()
        if "en" == code:
            return "en"
        if "tl" == code or "fil" == code or "tagalog" in code:
            return "tl"
        return "unknown"
    except Exception as e:
        logging.warning(f"Language detection failed: {e}")
        return "unknown"


def summarize_short(text: str, max_tokens: int = 160) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    try:
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            temperature=0.2,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that summarizes meeting transcripts. "
                        "Write a concise summary (1–3 sentences). "
                        "Use the same language as the input text."
                    ),
                },
                {"role": "user", "content": text[:4000]},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logging.warning(f"summarize_short error: {e}")
        return ""


def summarize_whole_transcript(text: str) -> str:
    """
    Single-shot global summary (no map-reduce) for speed.
    """
    text = (text or "").strip()
    if not text:
        return ""
    return summarize_short(text[:SUMMARY_MAX_INPUT_CHARS], max_tokens=SUMMARY_MAX_LEN)


def translate_with_gpt(text: str, src_lang: str, dst_lang: str, max_tokens: int = 512) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if src_lang == dst_lang:
        return text

    src_label = "English" if src_lang == "en" else "Tagalog/Filipino" if src_lang == "tl" else "auto-detect"
    dst_label = "English" if dst_lang == "en" else "Tagalog/Filipino" if dst_lang == "tl" else dst_lang

    try:
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            temperature=0.2,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator. "
                        f"Translate the user's text from {src_label} to {dst_label}. "
                        "Preserve meaning and tone. "
                        "Reply with ONLY the translated text."
                    ),
                },
                {"role": "user", "content": text[:4000]},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logging.warning(f"translate_with_gpt error: {e}")
        return f"(translation failed: {e})"


def translate_long_text_gpt(
    txt: str,
    src_lang: str,
    dst_lang: str,
    chunk_chars: int = TRANSLATE_CHUNK_CHARS,
    hard_cap: int = CONTENT_TRANSLATE_MAX_CHARS,
) -> str:
    txt = (txt or "").strip()
    if not txt:
        return ""
    clipped = txt[:hard_cap]
    parts = split_into_chunks_text(clipped, chunk_chars)
    out: List[str] = []
    for ch in parts:
        out.append(translate_with_gpt(ch, src_lang, dst_lang, max_tokens=600))
    return " ".join([p for p in out if p]).strip()


# =========================
# Core pipeline using GPT-4o Transcribe Diarize
# =========================

def build_turns_from_diarized_segments(segments: List[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, str], int]:
    """
    Take diarized segments from gpt-4o-transcribe-diarize and convert them into
    the 'turns' list that your frontend already expects.
    """
    # segments have: .speaker, .start, .end, .text
    raw_segments: List[Dict[str, Any]] = []
    for seg in segments:
        try:
            text = (seg.text or "").strip()
        except AttributeError:
            text = (seg.get("text") or "").strip()  # just in case
        if not text:
            continue
        try:
            s = float(seg.start)
            e = float(seg.end)
            spk = str(seg.speaker) if getattr(seg, "speaker", None) is not None else "A"
        except AttributeError:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
            spk = str(seg.get("speaker") or "A")
        raw_segments.append({"speaker": spk, "start": s, "end": e, "text": text})

    raw_segments.sort(key=lambda x: (x["start"], x["end"]))

    # Build stable speaker order by first appearance
    first_seen: Dict[str, float] = {}
    for s in raw_segments:
        if s["speaker"] not in first_seen:
            first_seen[s["speaker"]] = s["start"]
    ordered_labels = sorted(first_seen.keys(), key=lambda k: first_seen[k])
    label_map = {lab: f"Speaker {i+1}" for i, lab in enumerate(ordered_labels)}
    num_speakers = max(1, len(ordered_labels))

    # Merge neighbouring segments of same speaker with small gaps
    turns: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    def push_cur():
        nonlocal cur
        if cur and cur.get("text"):
            cur["duration_s"] = max(0.0, cur["end"] - cur["start"])
            cur["time_label"] = time_label(cur["start"], cur["end"])
            # friendly label
            cur["speakerLabel"] = label_map.get(cur["speaker"], "Speaker")
            turns.append(cur)
        cur = None

    for seg in raw_segments:
        spk = seg["speaker"]
        s = seg["start"]
        e = seg["end"]
        txt = seg["text"]

        if cur is None:
            cur = {"speaker": spk, "start": s, "end": e, "text": txt}
            continue

        if cur["speaker"] == spk and (s - cur["end"] <= SAME_SPEAKER_GAP):
            # extend current turn
            cur["end"] = max(cur["end"], e)
            cur["text"] = (cur["text"] + " " + txt).strip()
        else:
            push_cur()
            cur = {"speaker": spk, "start": s, "end": e, "text": txt}

    push_cur()
    return turns, label_map, num_speakers


def run_transcription_pipeline(original_path: str, firebase_id: str):
    wav_filename = f"{uuid.uuid4()}.wav"
    wav_path = os.path.join(UPLOAD_DIR, wav_filename)
    ref = db.reference(f"meetings/{firebase_id}")

    try:
        # ---- Convert to wav ----
        update_status(ref, "converting", {"message": "Converting to WAV…"})
        duration_sec, channels = convert_to_wav(original_path, wav_path)
        update_status(ref, "converting", {"durationSec": duration_sec, "channels": channels})

        # ---- Transcribe + diarize via OpenAI ----
        update_status(ref, "transcribing", {"message": "Transcribing audio with GPT-4o…"})
        logging.info(f"Sending audio to OpenAI model={TRANSCRIBE_MODEL}")

        with open(wav_path, "rb") as f:
            transcribe_kwargs: Dict[str, Any] = {
                "file": f,
                "model": TRANSCRIBE_MODEL,
            }
            # diarization model uses diarized_json and usually chunking_strategy="auto"
            if "diarize" in TRANSCRIBE_MODEL:
                transcribe_kwargs["response_format"] = "diarized_json"
                transcribe_kwargs["chunking_strategy"] = "auto"
            else:
                # fallback: plain json (no diarization)
                transcribe_kwargs["response_format"] = "json"

            transcript = client.audio.transcriptions.create(**transcribe_kwargs)

        full_text = getattr(transcript, "text", "") or ""
        segments = getattr(transcript, "segments", None)
        api_duration = getattr(transcript, "duration", None)

        total_duration = float(api_duration) if api_duration is not None else float(duration_sec)

        # ---- Build turns ----
        if segments:
            turns, label_map, num_speakers = build_turns_from_diarized_segments(list(segments))
        else:
            # No diarization available; just one big "Speaker 1".
            logging.info("No segments in transcription; creating single-turn transcript.")
            label_map = {"A": "Speaker 1"}
            num_speakers = 1
            turns = [
                {
                    "speaker": "A",
                    "speakerLabel": "Speaker 1",
                    "start": 0.0,
                    "end": total_duration,
                    "duration_s": total_duration,
                    "time_label": time_label(0.0, total_duration),
                    "text": full_text,
                }
            ]

        participants_label = f"{num_speakers} participant" + ("" if num_speakers == 1 else "s")

        # ---- Language detection ----
        language = detect_language_code(full_text)
        logging.info(f"Detected language: {language}")

        # ---- Summarization + translations (GLOBAL + optional per-turn translation) ----
        update_status(ref, "summarizing", {"message": "Summarizing content…"})

        # Decide target translation language for EN<->TL
        target_lang: Optional[str] = None
        if TRANSLATE_EN_TL:
            if language == "en":
                target_lang = "tl"
            elif language == "tl":
                target_lang = "en"

        # Per-turn: NO per-turn summary; translation optional
        for t in turns:
            raw = t.get("text", "") or ""

            # No per-turn summary
            t["summary"] = ""

            # Optional per-turn translation
            if ENABLE_TURN_TRANSLATION and TRANSLATE_EN_TL and target_lang and raw:
                t["translation"] = translate_with_gpt(raw, language, target_lang, max_tokens=256)
            else:
                t["translation"] = ""

        # ---- Global summary (single-shot) ----
        global_summary = summarize_whole_transcript(full_text) if full_text else ""

        # ---- Translations (summary + full content) ----
        summary_translation = ""
        content_translation = ""

        if TRANSLATE_EN_TL and target_lang:
            if global_summary:
                summary_translation = translate_with_gpt(global_summary, language, target_lang, max_tokens=256)
            if ENABLE_CONTENT_TRANSLATION and full_text:
                content_translation = translate_long_text_gpt(
                    full_text, language, target_lang, chunk_chars=TRANSLATE_CHUNK_CHARS
                )

        # ---- Save to Firebase ----
        payload = {
            "status": "done",
            "language": language,
            "numberOfSpeakers": num_speakers,
            "participantsLabel": participants_label,
            "speakers": label_map,
            "transcription": full_text,
            "summary": global_summary,
            "translation": summary_translation,
            "summaryTranslation": summary_translation,
            "contentTranslation": content_translation,
            "turns": turns,
        }
        ref.update(payload)

    except Exception as e:
        logging.exception("❌ Pipeline error")
        try:
            ref.update({"status": "error", "error_message": str(e)})
        except Exception:
            pass
    finally:
        for p in (original_path, wav_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# =========================
# Endpoints
# =========================

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files or "firebase_id" not in request.form:
        return jsonify({"error": "Missing audio or firebase_id"}), 400

    audio_file = request.files["audio"]
    firebase_id = request.form["firebase_id"].strip()
    if not firebase_id:
        return jsonify({"error": "Invalid firebase_id"}), 400

    if request.content_length and request.content_length > MAX_UPLOAD_MB * 1024 * 1024:
        return jsonify({"error": f"File too large (>{MAX_UPLOAD_MB} MB)"}), 413

    filename = safe_filename(audio_file.filename or f"{uuid.uuid4().hex}.bin")
    ext = os.path.splitext(filename)[1].lower()
    if ext and ext not in ALLOWED_EXT:
        filename = os.path.splitext(filename)[0] + ".bin"

    original_path = os.path.join(UPLOAD_DIR, filename)
    audio_file.save(original_path)

    ref = db.reference(f"meetings/{firebase_id}")
    update_status(ref, "queued", {"message": "Job queued"})

    threading.Thread(
        target=run_transcription_pipeline, args=(original_path, firebase_id), daemon=True
    ).start()
    return jsonify({"status": "queued"}), 200


@app.route("/transcribe_by_url", methods=["POST"])
def transcribe_by_url():
    data = request.get_json(silent=True) or {}
    firebase_id = (data.get("firebase_id") or "").strip()
    url = (data.get("url") or "").strip()
    if not firebase_id or not url:
        return jsonify({"error": "Missing firebase_id or url"}), 400
    try:
        guess = url.split("?")[0].split("/")[-1]
        filename = safe_filename(guess or f"{uuid.uuid4().hex}.m4a")
        if os.path.splitext(filename)[1].lower() not in ALLOWED_EXT:
            filename += ".m4a"
        original_path = os.path.join(UPLOAD_DIR, filename)
        ref = db.reference(f"meetings/{firebase_id}")
        update_status(ref, "queued", {"message": "Downloading audio…"})
        download_to_file(url, original_path)
    except Exception as e:
        logging.warning(f"Download error: {e}")
        return jsonify({"error": f"Failed to download: {e}"}), 400

    threading.Thread(
        target=run_transcription_pipeline, args=(original_path, firebase_id), daemon=True
    ).start()
    return jsonify({"status": "queued"}), 200


@app.route("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "transcribe_model": TRANSCRIBE_MODEL,
            "text_model": TEXT_MODEL,
            "enable_turn_translation": ENABLE_TURN_TRANSLATION,
            "enable_content_translation": ENABLE_CONTENT_TRANSLATION,
        }
    ), 200


@app.route("/")
def index():
    return "OK", 200


# =========================
# Serve
# =========================

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
