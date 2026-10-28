"""
Evidence-Based Speech Recognition Service (Kannada and English)
==============================================================
Provides robust, genuine speech transcription supporting Kannada (kn) and English (en)
using the installed Whisper ASR architecture (vasista22/whisper-kannada-small).

Core Features:
  1. Strict Kannada Unicode preservation (NFC normalization, UTF-8 integrity).
  2. English speech recognition support via model decoder prompt configuration.
  3. Strict anti-hallucination scoring and repetition/degeneracy filtering.
  4. Audio silence and noise pre-checks (zero-energy rejection).
  5. Unsupported language rejection (Devanagari, Cyrillic, Arabic, etc.).
  6. Chunk-level timestamp preservation when supported.
  7. Recoverable component failure handling (never crashes host pipeline).
"""

import os
import sys
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    GenerationConfig,
    pipeline
)

from services.audio_preprocessor import (
    extract_audio_waveform,
    AUDIO_SAMPLE_RATE
)

# Constants
ASR_MODEL_ID = "vasista22/whisper-kannada-small"
ASR_CHUNK_SECONDS = 30
ASR_MAX_LENGTH = 448
SUPPORTED_LANGUAGES = ["kn", "en"]

# Known Whisper hallucination phrases (often produced on near-silent or background audio)
HALLUCINATION_BLACKLIST = [
    "thank you for watching",
    "thanks for watching",
    "see you in the next video",
    "god bless you",
    "subtitles by",
    "amara.org",
    "subscribe to my channel",
    "please like and subscribe",
    "ಧನ್ಯವಾದಗಳು ವೀಕ್ಷಿಸಿದಕ್ಕಾಗಿ"  # Synthetic YouTube closing
]

# Global cache for loaded ASR pipeline
_ASR_PIPELINE = None


def get_or_load_asr_pipeline(device: Optional[int] = None):
    """
    Safely retrieves or initializes the Whisper ASR pipeline with
    proper generation configuration for timestamps and language support.
    """
    global _ASR_PIPELINE
    if _ASR_PIPELINE is not None:
        return _ASR_PIPELINE

    gpu_available = torch.cuda.is_available()
    dev = 0 if gpu_available else -1 if device is None else device
    dtype = torch.float16 if gpu_available else torch.float32

    try:
        processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            ASR_MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )

        # Configure GenerationConfig for timestamp generation
        try:
            gen_cfg = GenerationConfig.from_pretrained("openai/whisper-small")
            model.generation_config = gen_cfg
        except Exception:
            # Fallback generation config if remote HF call is disabled
            model.generation_config.no_timestamps_token_id = 50363

        asr_pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=ASR_CHUNK_SECONDS,
            batch_size=1,
            device=dev,
            torch_dtype=dtype
        )
        _ASR_PIPELINE = asr_pipe
        return _ASR_PIPELINE
    except Exception as exc:
        print(f"Warning: Failed to load ASR pipeline: {exc}")
        return None


def clean_and_normalize_unicode(text: str) -> str:
    """
    Normalizes text to Unicode NFC form, strips excessive whitespace,
    and preserves Kannada script and ASCII English without corruption.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", str(text))
    cleaned = re.sub(r"\s+", " ", normalized).strip()
    return cleaned


def analyze_transcript_language(text: str) -> Tuple[str, float, float]:
    """
    Analyzes the script composition of the text.
    Returns:
      (dominant_lang, kannada_ratio, english_ratio)
      dominant_lang in {'kannada', 'english', 'unsupported', 'none'}
    """
    if not text:
        return "none", 0.0, 0.0

    alphabetic = [c for c in text if c.isalpha()]
    if not alphabetic:
        return "none", 0.0, 0.0

    kannada_chars = [c for c in alphabetic if "\u0C80" <= c <= "\u0CFF"]
    english_chars = [c for c in alphabetic if c.isascii()]

    kannada_ratio = len(kannada_chars) / len(alphabetic)
    english_ratio = len(english_chars) / len(alphabetic)

    # Check for unsupported scripts (e.g. Devanagari \u0900-\u097F, Telugu \u0C00-\u0C7F, etc.)
    unsupported_chars = [
        c for c in alphabetic
        if not ("\u0C80" <= c <= "\u0CFF" or c.isascii())
    ]
    if len(unsupported_chars) / len(alphabetic) > 0.30:
        return "unsupported", round(kannada_ratio, 3), round(english_ratio, 3)

    if kannada_ratio >= 0.50:
        return "kannada", round(kannada_ratio, 3), round(english_ratio, 3)
    elif english_ratio >= 0.60:
        return "english", round(kannada_ratio, 3), round(english_ratio, 3)
    else:
        return "unsupported", round(kannada_ratio, 3), round(english_ratio, 3)


def score_and_validate_transcript(
    raw_text: str,
    duration_seconds: float,
    target_language: str = "auto"
) -> Dict[str, Any]:
    """
    Evaluates transcript credibility, checks for autoregressive hallucination loops,
    and enforces script conformity.
    """
    cleaned = clean_and_normalize_unicode(raw_text)
    if not cleaned:
        return {
            "transcript": "",
            "accepted": False,
            "language": "none",
            "reason": "empty_text",
            "kannada_ratio": 0.0,
            "english_ratio": 0.0,
            "word_count": 0
        }

    # Check against known hallucination phrases
    cleaned_lower = cleaned.lower()
    for phrase in HALLUCINATION_BLACKLIST:
        if phrase in cleaned_lower:
            return {
                "transcript": "",
                "accepted": False,
                "language": "none",
                "reason": f"known_hallucination_pattern ('{phrase}')",
                "kannada_ratio": 0.0,
                "english_ratio": 0.0,
                "word_count": 0
            }

    words = cleaned.split()
    word_count = len(words)
    if word_count < 1:
        return {
            "transcript": "",
            "accepted": False,
            "language": "none",
            "reason": "zero_words",
            "kannada_ratio": 0.0,
            "english_ratio": 0.0,
            "word_count": 0
        }

    # Repetition and Degeneracy Checks
    unique_ratio = len(set(words)) / word_count
    dominant_ratio = Counter(words).most_common(1)[0][1] / word_count
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)] if word_count >= 4 else []
    repeated_bigram_ratio = (
        sum(c for c in Counter(bigrams).values() if c > 1) / len(bigrams)
    ) if bigrams else 0.0

    words_per_sec = word_count / duration_seconds if duration_seconds > 0 else 0.0

    dom_lang, kn_ratio, en_ratio = analyze_transcript_language(cleaned)

    # Language suitability
    if target_language == "kn" and dom_lang != "kannada":
        return {
            "transcript": "",
            "accepted": False,
            "language": dom_lang,
            "reason": f"expected_kannada_got_{dom_lang}",
            "kannada_ratio": kn_ratio,
            "english_ratio": en_ratio,
            "word_count": word_count
        }
    elif target_language == "en" and dom_lang != "english":
        return {
            "transcript": "",
            "accepted": False,
            "language": dom_lang,
            "reason": f"expected_english_got_{dom_lang}",
            "kannada_ratio": kn_ratio,
            "english_ratio": en_ratio,
            "word_count": word_count
        }
    elif dom_lang == "unsupported":
        return {
            "transcript": "",
            "accepted": False,
            "language": "unsupported",
            "reason": "unsupported_language_script",
            "kannada_ratio": kn_ratio,
            "english_ratio": en_ratio,
            "word_count": word_count
        }

    # Threshold acceptance
    accepted = (
        word_count >= 1
        and unique_ratio >= 0.25
        and dominant_ratio <= 0.65
        and repeated_bigram_ratio <= 0.55
        and words_per_sec <= 6.0
        and (kn_ratio >= 0.50 or en_ratio >= 0.60)
    )

    reason = "accepted" if accepted else "failed_quality_metrics"
    if not accepted:
        if words_per_sec > 6.0:
            reason = "unrealistic_speech_rate"
        elif dominant_ratio > 0.65 or repeated_bigram_ratio > 0.55:
            reason = "repetitive_loop_detected"

    return {
        "transcript": cleaned if accepted else "",
        "accepted": accepted,
        "language": dom_lang if accepted else "none",
        "reason": reason,
        "kannada_ratio": kn_ratio,
        "english_ratio": en_ratio,
        "word_count": word_count if accepted else 0,
        "unique_ratio": round(unique_ratio, 3),
        "dominant_ratio": round(dominant_ratio, 3)
    }


def transcribe_audio(
    waveform: np.ndarray,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    language: str = "auto",
    asr_pipe: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Transcribes audio waveform with language handling and timestamp extraction.
    Returns:
      Dict with transcript, language, accepted, chunks, word_count, and processing metadata.
    """
    t_start = time.time()

    # 1. Audio pre-validation
    if waveform is None or len(waveform) == 0:
        return {
            "transcript": "",
            "accepted": False,
            "language": "none",
            "status": "missing_audio",
            "chunks": [],
            "word_count": 0,
            "duration_seconds": 0.0,
            "timings": {"duration_seconds": 0.0}
        }

    duration = float(len(waveform) / sample_rate)
    if duration < 0.50:
        return {
            "transcript": "",
            "accepted": False,
            "language": "none",
            "status": "audio_too_short",
            "chunks": [],
            "word_count": 0,
            "duration_seconds": round(duration, 3),
            "timings": {"duration_seconds": round(time.time() - t_start, 3)}
        }

    # Check for silence (RMS energy)
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    if rms < 0.001:
        return {
            "transcript": "",
            "accepted": False,
            "language": "none",
            "status": "silent_audio",
            "chunks": [],
            "word_count": 0,
            "duration_seconds": round(duration, 3),
            "timings": {"duration_seconds": round(time.time() - t_start, 3)}
        }

    # 2. Get ASR pipeline
    pipe = asr_pipe or get_or_load_asr_pipeline()
    if pipe is None:
        return {
            "transcript": "",
            "accepted": False,
            "language": "none",
            "status": "asr_model_unavailable",
            "chunks": [],
            "word_count": 0,
            "duration_seconds": round(duration, 3),
            "timings": {"duration_seconds": round(time.time() - t_start, 3)}
        }

    # 3. Transcribe with error recovery
    try:
        # Determine language parameter for Whisper
        gen_kwargs = {
            "max_length": ASR_MAX_LENGTH,
            "no_repeat_ngram_size": 3,
            "num_beams": 1,
            "do_sample": False
        }
        if language in ["kn", "en"]:
            gen_kwargs["language"] = language
            gen_kwargs["task"] = "transcribe"
        else:
            # Default to Kannada for vasista22 model, fallback to auto
            gen_kwargs["language"] = "kn"
            gen_kwargs["task"] = "transcribe"

        try:
            raw_result = pipe(
                {"raw": waveform.astype(np.float32), "sampling_rate": sample_rate},
                return_timestamps=True,
                generate_kwargs=gen_kwargs
            )
        except Exception:
            # Fallback for models without timestamp generation configs (e.g. fine-tuned whisper-kannada-small)
            raw_result = pipe(
                {"raw": waveform.astype(np.float32), "sampling_rate": sample_rate},
                return_timestamps=False,
                generate_kwargs=gen_kwargs
            )

        raw_text = raw_result.get("text", "")
        chunks_data = raw_result.get("chunks", [])

        # Format preserved chunk timestamps
        parsed_chunks = []
        for ch in chunks_data:
            ts = ch.get("timestamp")
            if ts and isinstance(ts, (list, tuple)):
                start_sec = round(float(ts[0]), 3) if ts[0] is not None else 0.0
                end_sec = round(float(ts[1]), 3) if ts[1] is not None else round(duration, 3)
                ch_text = clean_and_normalize_unicode(ch.get("text", ""))
                if ch_text:
                    parsed_chunks.append({
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "text": ch_text
                    })

        # Score and validate transcript
        quality = score_and_validate_transcript(raw_text, duration, target_language=language)

        if not quality["accepted"]:
            parsed_chunks = []

        # If chunks were empty but transcript accepted, synthesize overall chunk
        if quality["accepted"] and not parsed_chunks and quality["transcript"]:
            parsed_chunks.append({
                "start_sec": 0.0,
                "end_sec": round(duration, 3),
                "text": quality["transcript"]
            })

        duration_sec = round(time.time() - t_start, 3)

        return {
            "transcript": quality["transcript"],
            "accepted": quality["accepted"],
            "language": quality["language"],
            "status": "completed" if quality["accepted"] else quality["reason"],
            "chunks": parsed_chunks,
            "word_count": quality["word_count"],
            "kannada_ratio": quality["kannada_ratio"],
            "english_ratio": quality["english_ratio"],
            "duration_seconds": round(duration, 3),
            "timings": {
                "transcription_seconds": duration_sec
            }
        }

    except Exception as exc:
        print(f"Warning: ASR execution failed (recoverable): {exc}")
        return {
            "transcript": "",
            "accepted": False,
            "language": "none",
            "status": f"asr_execution_error: {str(exc)}",
            "chunks": [],
            "word_count": 0,
            "duration_seconds": round(duration, 3),
            "timings": {"duration_seconds": round(time.time() - t_start, 3)}
        }


def transcribe_video_speech(
    video_path: Path,
    language: str = "auto",
    asr_pipe: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Extracts audio from video file and executes genuine speech recognition.
    """
    audio_res = extract_audio_waveform(video_path, sample_rate=AUDIO_SAMPLE_RATE)
    if not audio_res["has_audio_stream"] or audio_res["is_silent"]:
        return {
            "transcript": "",
            "accepted": False,
            "language": "none",
            "status": "silent_or_missing_audio",
            "chunks": [],
            "word_count": 0,
            "duration_seconds": audio_res["duration_seconds"],
            "timings": {"duration_seconds": 0.0}
        }

    return transcribe_audio(
        waveform=audio_res["waveform"],
        sample_rate=audio_res["sample_rate"],
        language=language,
        asr_pipe=asr_pipe
    )
