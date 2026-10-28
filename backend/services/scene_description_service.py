"""
Evidence-Grounded Scene Description Service
==========================================
Generates factual, structured scene descriptions grounded strictly in verified
video observations, actual audio analysis, speech transcripts, and model prediction outputs.

Enforces the 6 mandatory sections:
  1. Visual observations.
  2. Audio observations.
  3. Speech transcript and summary.
  4. Key timestamps.
  5. Model prediction and class scores.
  6. Uncertainty and limitations.

Anti-Hallucination Guarantees:
  - Never asserts a fire, collision, siren, or emergency call occurred without verified evidence.
  - Treats missing ASR or description generation as a recoverable component failure.
  - Distinguishes model sensitivity from real-world physical causation.
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional, List

CAUSATION_DISCLAIMER = (
    "Perturbation results indicate model sensitivity to individual temporal frame removals "
    "and do not constitute proof of real-world physical causation."
)


def generate_evidence_grounded_scene_description(
    video_evidence: Dict[str, Any],
    audio_evidence: Dict[str, Any],
    speech_evidence: Dict[str, Any],
    prediction_details: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Constructs a structured, verified multi-section description solely from
    measured evidence across visual, audio, speech, and fusion model outputs.
    """
    t_start = time.time()

    # -------------------------------------------------------------------------
    # 1. Visual Observations
    # -------------------------------------------------------------------------
    sampled_count = video_evidence.get("sampled_frame_count", 12)
    resolution = video_evidence.get("frame_resolution", [224, 224])
    duration_sec = video_evidence.get("duration_seconds", 0.0)
    timestamps = video_evidence.get("frame_timestamps_sec", [])
    influential_data = video_evidence.get("influential_frames") or {}
    ranked_frames = influential_data.get("ranked_frames", [])

    keyframe_indices = []
    keyframe_timestamps = []
    rank_1_frame = None
    rank_1_delta = None

    if ranked_frames:
        for rf in ranked_frames:
            if rf.get("is_keyframe"):
                keyframe_indices.append(rf.get("frame_index"))
                keyframe_timestamps.append(rf.get("timestamp_sec"))
            if rf.get("rank") == 1:
                rank_1_frame = rf.get("frame_index")
                rank_1_delta = rf.get("score_difference")

    visual_summary_parts = [
        f"Visual stream evaluated across {sampled_count} uniformly sampled frames at {resolution[0]}x{resolution[1]} resolution."
    ]
    if duration_sec > 0:
        visual_summary_parts.append(f"Recorded scene duration spans {duration_sec:.2f} seconds.")
    if rank_1_frame is not None:
        visual_summary_parts.append(
            f"Temporal ablation identified frame {rank_1_frame} as the primary sensitivity keyframe (rank 1, Δ score: {rank_1_delta:+.4f})."
        )
    if keyframe_indices:
        visual_summary_parts.append(f"Top influential keyframe indices: {keyframe_indices} at timestamps {keyframe_timestamps}s.")

    visual_observations = {
        "summary": " ".join(visual_summary_parts),
        "sampled_frame_count": sampled_count,
        "resolution": resolution,
        "duration_seconds": round(duration_sec, 3),
        "sampled_timestamps_sec": timestamps,
        "keyframe_indices": keyframe_indices,
        "keyframe_timestamps_sec": keyframe_timestamps,
        "influential_rank_1_frame": rank_1_frame,
        "influential_rank_1_score_difference": rank_1_delta
    }

    # -------------------------------------------------------------------------
    # 2. Audio Observations
    # -------------------------------------------------------------------------
    has_audio = audio_evidence.get("has_audio_stream", False)
    is_silent = audio_evidence.get("is_silent", True)
    audio_dur = audio_evidence.get("duration_seconds", 0.0)
    waveform_stats = audio_evidence.get("waveform_stats") or {}
    mel_bins = audio_evidence.get("mel_bins", 64)
    rms_val = waveform_stats.get("rms_energy", 0.0)
    peak_val = waveform_stats.get("peak_amplitude", 0.0)

    audio_summary_parts = []
    if not has_audio or audio_dur <= 0.0:
        audio_state = "missing"
        audio_summary_parts.append(
            "Audio stream is absent from the media container. Acoustic gating was set to 0.0 (inactive)."
        )
    elif is_silent or rms_val < 0.001:
        audio_state = "silent"
        audio_summary_parts.append(
            f"Audio stream present ({audio_dur:.2f}s) but detected as near-silent (RMS: {rms_val:.5f}). "
            "No active acoustic events or emergency sirens detected."
        )
    else:
        audio_state = "active"
        audio_summary_parts.append(
            f"Acoustic stream active across {audio_dur:.2f}s with mean RMS energy {rms_val:.4f} and peak {peak_val:.4f}. "
            f"Spectrogram representation computed across {mel_bins} mel frequency bins. "
            "No ungrounded sirens, horns, or alarms are claimed unless indicated by acoustic feature representations."
        )

    audio_observations = {
        "summary": " ".join(audio_summary_parts),
        "has_audio_stream": has_audio,
        "is_silent": is_silent,
        "acoustic_state": audio_state,
        "duration_seconds": round(audio_dur, 3),
        "mel_bins": mel_bins,
        "rms_energy": round(rms_val, 6) if rms_val is not None else 0.0,
        "peak_amplitude": round(peak_val, 6) if peak_val is not None else 0.0
    }

    # -------------------------------------------------------------------------
    # 3. Speech Transcript and Summary
    # -------------------------------------------------------------------------
    transcript_text = speech_evidence.get("transcript") or ""
    is_accepted = speech_evidence.get("accepted", False)
    lang = speech_evidence.get("language") or "none"
    chunks = speech_evidence.get("chunks", [])
    word_count = speech_evidence.get("word_count") or (len(transcript_text.split()) if transcript_text else 0)

    speech_summary_parts = []
    if is_accepted and transcript_text.strip():
        lang_label = "Kannada" if lang == "kannada" else "English" if lang == "english" else lang.capitalize()
        speech_summary_parts.append(
            f"Intelligible speech recognized in {lang_label} ({word_count} words): \"{transcript_text}\"."
        )
        if chunks:
            chunk_intervals = [f"[{c.get('start_sec', 0.0):.1f}s-{c.get('end_sec', 0.0):.1f}s]" for c in chunks[:3]]
            speech_summary_parts.append(f"Speech activity spans intervals: {', '.join(chunk_intervals)}.")
    else:
        speech_summary_parts.append(
            "No intelligible Kannada or English speech detected in the audio track. "
            "Speech embedding defaulted to neutral zero vector representation (384 dimensions)."
        )

    speech_transcript_and_summary = {
        "summary": " ".join(speech_summary_parts),
        "detected_language": lang if is_accepted else "none",
        "raw_transcript": transcript_text if is_accepted else "",
        "is_accepted": is_accepted,
        "word_count": word_count if is_accepted else 0,
        "speech_present": bool(is_accepted and transcript_text.strip()),
        "chunks": chunks if is_accepted else []
    }

    # -------------------------------------------------------------------------
    # 4. Key Timestamps
    # -------------------------------------------------------------------------
    key_timestamps = [
        {"timestamp_sec": 0.0, "event_type": "scene_start", "description": "Scene analysis interval begins."}
    ]

    # Add keyframe timestamps
    if ranked_frames:
        for rf in ranked_frames:
            if rf.get("is_keyframe"):
                r_idx = rf.get("rank")
                f_idx = rf.get("frame_index")
                f_time = rf.get("timestamp_sec")
                delta_s = rf.get("score_difference")
                key_timestamps.append({
                    "timestamp_sec": f_time,
                    "event_type": f"keyframe_rank_{r_idx}",
                    "description": f"Visual keyframe {f_idx} (Rank {r_idx}, Δ score: {delta_s:+.4f})."
                })

    # Add speech chunk intervals
    if is_accepted and chunks:
        for c in chunks:
            key_timestamps.append({
                "timestamp_sec": c.get("start_sec", 0.0),
                "event_type": "speech_onset",
                "description": f"Spoken phrase start: \"{c.get('text', '')}\""
            })

    if duration_sec > 0:
        key_timestamps.append({
            "timestamp_sec": round(duration_sec, 3),
            "event_type": "scene_end",
            "description": "Scene analysis interval concludes."
        })

    # Sort key timestamps chronologically
    key_timestamps = sorted(key_timestamps, key=lambda x: x["timestamp_sec"])

    # -------------------------------------------------------------------------
    # 5. Model Prediction and Class Scores
    # -------------------------------------------------------------------------
    pred_class = prediction_details.get("predicted_class", "Normal")
    confidence = prediction_details.get("confidence", 0.0)
    class_scores = prediction_details.get("class_scores") or {}
    score_semantics = prediction_details.get("score_semantics", "uncalibrated_softmax_activations")
    modalities_used = prediction_details.get("modality_availability") or {
        "visual": True,
        "audio": bool(has_audio and not is_silent),
        "speech": bool(is_accepted and transcript_text.strip())
    }

    model_prediction_and_scores = {
        "predicted_class": pred_class,
        "confidence": round(float(confidence), 4),
        "class_scores": {k: round(float(v), 4) for k, v in class_scores.items()},
        "score_semantics": score_semantics,
        "modalities_used": modalities_used
    }

    # -------------------------------------------------------------------------
    # 6. Uncertainty and Limitations
    # -------------------------------------------------------------------------
    uncertainty_items = [
        CAUSATION_DISCLAIMER,
        "Softmax confidence scores represent internal multimodal activations and are not calibrated real-world probabilities.",
        "Speech recognition is strictly constrained to genuine Kannada and English acoustic phonetics; background ambient noise may affect transcription fidelity.",
        "All visual evidence is derived from 12 uniformly sampled temporal frames; unsampled events between frames cannot be verified."
    ]

    if not has_audio:
        uncertainty_items.append("Audio modality was unavailable; acoustic features were gated to zero.")
    elif is_silent:
        uncertainty_items.append("Audio stream was near-silent; acoustic features did not contribute distinctive cues.")

    if not is_accepted or not transcript_text.strip():
        uncertainty_items.append("No speech transcript was available; speech embedding defaulted to zero-vector fallback.")

    uncertainty_items.append(
        "Strict Evidence Boundary: No fire outbreak, vehicular collision, acoustic siren, or emergency dispatch "
        "is claimed without verified supporting sensor evidence."
    )

    uncertainty_and_limitations = {
        "causation_disclaimer": CAUSATION_DISCLAIMER,
        "limitations": uncertainty_items,
        "evidence_boundaries": "Constrained strictly to measured visual, acoustic, and speech tokens."
    }

    # -------------------------------------------------------------------------
    # Markdown Formatted Text Representation
    # -------------------------------------------------------------------------
    lines = [
        "# Evidence-Grounded Scene Description",
        "",
        "### 1. Visual Observations",
        visual_observations["summary"],
        "",
        "### 2. Audio Observations",
        audio_observations["summary"],
        "",
        "### 3. Speech Transcript and Summary",
        speech_transcript_and_summary["summary"],
        "",
        "### 4. Key Timestamps"
    ]
    for kt in key_timestamps:
        lines.append(f"- **{kt['timestamp_sec']:.2f}s** [{kt['event_type']}]: {kt['description']}")

    lines.extend([
        "",
        "### 5. Model Prediction and Class Scores",
        f"- **Predicted Class**: {pred_class} (Confidence: {confidence:.2%})",
        f"- **Class Scores**: Fire: {class_scores.get('Fire', 0.0):.4f}, Accident: {class_scores.get('Accident', 0.0):.4f}, Normal: {class_scores.get('Normal', 0.0):.4f}",
        f"- **Active Modalities**: Video: {modalities_used.get('visual', True)}, Audio: {modalities_used.get('audio', False)}, Speech: {modalities_used.get('speech', False)}",
        "",
        "### 6. Uncertainty and Limitations"
    ])
    for item in uncertainty_items:
        lines.append(f"- {item}")

    formatted_text = "\n".join(lines)
    duration_gen_sec = round(time.time() - t_start, 4)

    return {
        "structured": {
            "visual_observations": visual_observations,
            "audio_observations": audio_observations,
            "speech_transcript_and_summary": speech_transcript_and_summary,
            "key_timestamps": key_timestamps,
            "model_prediction_and_class_scores": model_prediction_and_scores,
            "uncertainty_and_limitations": uncertainty_and_limitations
        },
        "formatted_text": formatted_text,
        "timings": {
            "description_generation_seconds": duration_gen_sec
        }
    }
