import os
import json
import time
import shutil
import hashlib
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import cv2
import librosa
import torch
from PIL import Image

import database
from services.model_metadata_service import compute_sha256
from services.storage_manager import (
    UPLOADS_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    save_frame_as_artifact,
    save_spectrogram_as_artifact,
    save_report_as_artifact,
    MIME_TYPES
)
from services.video_preprocessor import (
    validate_video_file_detailed,
    extract_video_frames_with_metadata,
    compute_visual_embeddings,
    generate_video_thumbnails_and_contact_sheet
)
from services.audio_preprocessor import (
    compute_audio_logmel_features,
    extract_audio_waveform,
    save_audio_spectrogram_artifact,
    TIME_STEPS
)
from services.model_inference_service import (
    run_direct_multimodal_inference,
    get_production_model_metadata,
    locate_model_file,
    compute_sha256 as compute_model_sha256,
    CLASS_NAMES,
    CLASS_INDEX_MAP,
    MODEL_FILENAME
)
from services.frame_attribution_service import (
    compute_evidence_based_frame_attributions,
    CAUSATION_DISCLAIMER
)
from services.speech_recognition_service import (
    transcribe_video_speech,
    clean_and_normalize_unicode
)
from services.scene_description_service import (
    generate_evidence_grounded_scene_description
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BACKEND_DIR.parent / MODEL_FILENAME
if not MODEL_PATH.exists():
    try:
        MODEL_PATH = locate_model_file()
    except Exception:
        MODEL_PATH = BACKEND_DIR.parent / "kannada_av_scene_model.keras"

# Configuration constants
NUM_FRAMES = 12
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
AUDIO_SAMPLE_RATE = 16000
AUDIO_FIXED_SECONDS = 5.0
MEL_BINS = 64
N_FFT = 1024
HOP_LENGTH = 256
ASR_MAX_LENGTH = 448
TEXT_EMBEDDING_DIM = 384

# Concurrency Manager
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="av_scene_worker")
_ACTIVE_FUTURES: Dict[str, Any] = {}
_CANCEL_REQUESTED = set()

def get_job_queue_pending_count() -> int:
    return sum(1 for f in _ACTIVE_FUTURES.values() if not f.done())

def validate_video_file(filename: str, file_size: int) -> Tuple[bool, Optional[str]]:
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(filename).suffix.lower()
    if ext not in valid_extensions:
        return False, f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(valid_extensions))}"
    
    max_size = 100 * 1024 * 1024 # 100 MB
    if file_size > max_size:
        return False, f"File size ({round(file_size/(1024*1024), 1)} MB) exceeds maximum limit of 100 MB."
    if file_size < 1024:
        return False, "File is too small or empty."
    
    return True, None

def submit_analysis_job(
    filename: str,
    temp_source_path: Path,
    file_size_bytes: int,
    models_dict: Dict[str, Any],
    force_reanalysis: bool = False,
    description: Optional[str] = None
) -> Tuple[Dict[str, Any], bool]:
    """
    Submits a video for asynchronous analysis with content-hash duplicate prevention.
    Returns (job_record, was_duplicate).
    """
    sha256_hash = compute_sha256(temp_source_path)

    # 1. Duplicate Prevention: Check if identical video file was already analyzed
    if not force_reanalysis:
        existing_job = database.find_job_by_file_hash(sha256_hash, status="completed")
        if existing_job:
            existing_job["is_duplicate"] = True
            existing_job["duplicate_of"] = existing_job["job_id"]
            existing_job["cached"] = True
            existing_job["message"] = f"Identical video hash detected; returning cached historical analysis from {existing_job.get('completed_at')}."
            return existing_job, True

    # 2. Setup job record and managed storage
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    sanitized_name = Path(filename).name
    permanent_filename = f"{job_id}_{sanitized_name}"
    permanent_path = UPLOADS_DIR / permanent_filename
    
    shutil.copyfile(str(temp_source_path), str(permanent_path))
    
    job_record = database.create_job(
        job_id=job_id,
        filename=sanitized_name,
        file_path=str(permanent_path),
        file_size_bytes=file_size_bytes,
        file_hash_sha256=sha256_hash,
        description=description,
        model_identifier=MODEL_FILENAME
    )

    # 3. Register uploaded video as an artifact
    ext = Path(sanitized_name).suffix.lower()
    vid_mime = MIME_TYPES.get(ext, "video/mp4")
    database.add_artifact(
        artifact_id=f"art_vid_{uuid.uuid4().hex[:10]}",
        job_id=job_id,
        artifact_type="video",
        filename=sanitized_name,
        storage_relative_path=f"uploads/{permanent_filename}",
        file_size_bytes=file_size_bytes,
        mime_type=vid_mime,
        file_hash_sha256=sha256_hash,
        metadata={"original_filename": sanitized_name}
    )
    
    # 4. Dispatch background execution
    future = _EXECUTOR.submit(_execute_job_pipeline, job_id, permanent_path, models_dict, description)
    _ACTIVE_FUTURES[job_id] = future
    
    job_record["is_duplicate"] = False
    return job_record, False

def cancel_job(job_id: str) -> bool:
    job = database.get_job(job_id)
    if not job or job["status"] not in ("queued", "processing"):
        return False
    
    _CANCEL_REQUESTED.add(job_id)
    future = _ACTIVE_FUTURES.get(job_id)
    if future and not future.running():
        future.cancel()
    
    return database.cancel_job_record(job_id)

def _execute_job_pipeline(job_id: str, video_path: Path, models_dict: Dict[str, Any], description: Optional[str] = None):
    try:
        if job_id in _CANCEL_REQUESTED:
            database.cancel_job_record(job_id)
            return

        now_iso = datetime.now().isoformat()
        database.update_job_progress(job_id, "processing", 0.05, "Starting analysis pipeline", started_at=now_iso)

        # Verify all models are loaded
        required_models = ["multimodal_model", "visual_processor", "visual_model", "text_encoder", "asr_pipe"]
        for m in required_models:
            if m not in models_dict or models_dict[m] is None:
                raise RuntimeError(f"Required model '{m}' is not loaded or available.")

        t0 = time.time()
        timings = {}

        # ----------------------------------------------------
        # Step 1: Video Frames & Visual Embedding (25%)
        # ----------------------------------------------------
        if job_id in _CANCEL_REQUESTED:
            database.cancel_job_record(job_id); return
        database.update_job_progress(job_id, "processing", 0.15, "Extracting video frames and attributions")

        extract_result = extract_video_frames_with_metadata(video_path, num_frames=NUM_FRAMES)
        frames = extract_result["frames"]
        frame_indices = extract_result["sampled_indices"]
        frame_timestamps = extract_result["frame_timestamps_sec"]
        total_frames = extract_result["total_frames_in_video"]
        fps = extract_result["fps"]
        duration = extract_result["duration_seconds"]
        preprocessing_warnings = extract_result["warnings"]

        # Generate genuine thumbnails and 12-frame contact sheet filmstrip for frontend
        thumb_info = generate_video_thumbnails_and_contact_sheet(job_id, frames, frame_timestamps)

        database.update_job_progress(job_id, "processing", 0.30, "Computing ConvNeXt visual embeddings")
        torch_device = next(models_dict["visual_model"].parameters()).device
        visual_features, visual_proc_time = compute_visual_embeddings(
            frames_rgb=frames,
            visual_processor=models_dict["visual_processor"],
            visual_model=models_dict["visual_model"],
            device=torch_device
        )
        timings["visual_processing_seconds"] = visual_proc_time
        timings["frame_extraction_seconds"] = extract_result["timings"]["total_extraction_seconds"]

        # ----------------------------------------------------
        # Step 2: Audio & Log-Mel Spectrogram (50%)
        # ----------------------------------------------------
        if job_id in _CANCEL_REQUESTED:
            database.cancel_job_record(job_id); return
        database.update_job_progress(job_id, "processing", 0.50, "Extracting audio and log-mel spectrogram artifact")

        t1 = time.time()
        audio_result = compute_audio_logmel_features(video_path)
        mel = audio_result["mel_tensor"]
        audio_present = audio_result["audio_available"]
        audio_dur = audio_result["duration_seconds"]
        num_windows = audio_result["num_windows"]
        audio_features = np.expand_dims(mel, axis=0)
        timings["audio_processing_seconds"] = round(time.time() - t1, 3)

        # Generate and save genuine publication-quality spectrogram artifact with time/frequency axes
        spec_art = save_audio_spectrogram_artifact(
            job_id=job_id,
            mel_tensor=mel,
            audio_available=audio_present,
            axes_metadata=audio_result["axes_metadata"],
            warnings=audio_result["warnings"]
        )
        database.add_artifact(
            artifact_id=spec_art["artifact_id"],
            job_id=job_id,
            artifact_type=spec_art["artifact_type"],
            filename=spec_art["filename"],
            storage_relative_path=spec_art["storage_relative_path"],
            file_size_bytes=spec_art["file_size_bytes"],
            mime_type=spec_art["mime_type"],
            file_hash_sha256=spec_art["file_hash_sha256"],
            metadata=spec_art["metadata"]
        )

        # ----------------------------------------------------
        # Step 3: ASR Speech Transcription & Encoding (75%)
        # ----------------------------------------------------
        if job_id in _CANCEL_REQUESTED:
            database.cancel_job_record(job_id); return
        database.update_job_progress(job_id, "processing", 0.70, "Transcribing speech (Kannada and English)")

        t2 = time.time()
        asr_result = transcribe_video_speech(video_path, language="auto", asr_pipe=models_dict.get("asr_pipe"))
        transcript = asr_result["transcript"]

        # Persist speech transcript separately in speech_transcripts table
        try:
            database.add_speech_transcript(
                job_id=job_id,
                language=asr_result.get("language", "none"),
                transcript=transcript,
                confidence=1.0 if asr_result.get("accepted") else 0.0,
                is_accepted=asr_result.get("accepted", False),
                chunks=asr_result.get("chunks", [])
            )
        except Exception as st_err:
            print(f"Warning: Failed to persist speech transcript to database: {st_err}")

        database.update_job_progress(job_id, "processing", 0.85, "Encoding text embeddings")
        if transcript:
            speech_embeddings = models_dict["text_encoder"].encode(
                [transcript], convert_to_numpy=True, normalize_embeddings=True
            )
        else:
            speech_embeddings = np.zeros((1, TEXT_EMBEDDING_DIM), dtype=np.float32)
        timings["speech_processing_seconds"] = round(time.time() - t2, 3)

        # ----------------------------------------------------
        # Step 4: Multimodal Cross-Modal Fusion Inference (90%)
        # ----------------------------------------------------
        if job_id in _CANCEL_REQUESTED:
            database.cancel_job_record(job_id); return
        database.update_job_progress(job_id, "processing", 0.90, f"Executing multimodal fusion inference ({MODEL_FILENAME})")

        t3 = time.time()
        inference_res = run_direct_multimodal_inference(
            video_frames=frames,
            audio_spectrogram=mel,
            audio_available=audio_present,
            transcript=transcript
        )
        pred_class = inference_res["predicted_class"]
        pred_class_idx = inference_res["predicted_index"]
        confidence = inference_res["confidence"]
        class_scores = inference_res["class_scores"]
        timings["fusion_inference_seconds"] = inference_res["timings"]["inference_duration_seconds"]
        # ----------------------------------------------------
        # Step 4b: Evidence-Based Influential Frame Attribution (95%)
        # ----------------------------------------------------
        if job_id in _CANCEL_REQUESTED:
            database.cancel_job_record(job_id); return
        database.update_job_progress(job_id, "processing", 0.95, "Measuring evidence-based influential frame attributions")

        t_attr = time.time()
        attribution_res = compute_evidence_based_frame_attributions(
            video_frames=frames,
            frame_timestamps_sec=frame_timestamps,
            audio_spectrogram=mel,
            audio_available=audio_present,
            transcript=transcript,
            target_class=pred_class,
            perturbation_strategy="neutral_gray",
            top_k_keyframes=3
        )
        timings["frame_attribution_seconds"] = attribution_res["timings"]["attribution_duration_seconds"]

        # Link keyframe artifacts into frame_attributions table
        keyframe_set = set()
        for idx_in_sample, f_attr in enumerate(attribution_res["frames"]):
            is_key = f_attr["is_keyframe"]
            if is_key:
                keyframe_set.add(idx_in_sample)
            art_id = thumb_info["frames"][idx_in_sample]["artifact_id"] if (idx_in_sample < len(thumb_info["frames"])) else None

            database.add_frame_attribution(
                job_id=job_id,
                frame_index=int(frame_indices[idx_in_sample]),
                timestamp_sec=f_attr["timestamp_sec"],
                attribution_score=f_attr["attribution_score"],
                baseline_score=f_attr["baseline_score"],
                perturbed_score=f_attr["perturbed_score"],
                score_difference=f_attr["score_difference"],
                attribution_method=attribution_res["attribution_method"],
                rank=f_attr["rank"],
                is_keyframe=is_key,
                artifact_id=art_id,
                label=f_attr["label"]
            )

        timings["total_pipeline_seconds"] = round(time.time() - t0, 3)

        # ----------------------------------------------------
        # Step 5: Evidence Assembly, Reports & Completion (100%)
        # ----------------------------------------------------
        probabilities = class_scores

        evidence = {
            "prediction_details": {
                "predicted_class": pred_class,
                "predicted_index": pred_class_idx,
                "class_scores": class_scores,
                "confidence": confidence,
                "score_semantics": inference_res["score_semantics"],
                "is_calibrated": False,
                "calibration_note": inference_res["calibration_note"],
                "modality_availability": inference_res["modality_availability"],
                "model_identification": inference_res["model_identification"],
                "warnings": inference_res["warnings"]
            },
            "visual_evidence": {
                "total_video_frames": total_frames,
                "fps": round(fps, 2) if fps else None,
                "duration_seconds": round(duration, 3),
                "sampled_frame_count": NUM_FRAMES,
                "sampled_indices": [int(idx) for idx in frame_indices],
                "frame_timestamps_sec": frame_timestamps,
                "frame_resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
                "visual_feature_dim": visual_features.shape[-1],
                "thumbnail_url": thumb_info["thumbnail_url"],
                "filmstrip_url": thumb_info["filmstrip_url"],
                "keyframes_saved": len(keyframe_set),
                "influential_frames": {
                    "attribution_method": attribution_res["attribution_method"],
                    "target_class": attribution_res["target_class"],
                    "baseline_score": attribution_res["baseline_score"],
                    "causation_disclaimer": attribution_res["causation_disclaimer"],
                    "reliability": attribution_res["reliability"],
                    "score_range": attribution_res["score_range"],
                    "ranked_frames": attribution_res["frames"],
                    "warnings": attribution_res["warnings"]
                },
                "warnings": preprocessing_warnings
            },
            "audio_evidence": {
                "audio_present": audio_present,
                "audio_available": audio_result["audio_available"],
                "has_audio_stream": audio_result["has_audio_stream"],
                "is_silent": audio_result["is_silent"],
                "duration_seconds": round(audio_dur, 2),
                "sample_rate_hz": AUDIO_SAMPLE_RATE,
                "windows_averaged": num_windows,
                "window_starts_sec": audio_result.get("window_starts_sec", []),
                "mel_bins": MEL_BINS,
                "time_steps": TIME_STEPS,
                "spectrogram_shape": list(mel.shape),
                "waveform_stats": audio_result["waveform_stats"],
                "axes_metadata": audio_result["axes_metadata"],
                "warnings": audio_result["warnings"],
                "status": audio_result["status"]
            },
            "speech_evidence": {
                "transcript": transcript,
                "transcript_accepted": asr_result["accepted"],
                "language": asr_result.get("language", "none"),
                "word_count": asr_result.get("word_count", 0),
                "chunks": asr_result.get("chunks", []),
                "kannada_char_ratio": asr_result.get("kannada_ratio", 0.0),
                "english_char_ratio": asr_result.get("english_ratio", 0.0),
                "embedding_dim": TEXT_EMBEDDING_DIM,
                "embedding_norm": float(np.linalg.norm(speech_embeddings)),
                "used_fallback_zero_vector": bool(not transcript)
            },
            "timings": timings
        }

        # Generate evidence-grounded structured scene description
        desc_res = generate_evidence_grounded_scene_description(
            video_evidence=evidence["visual_evidence"],
            audio_evidence=evidence["audio_evidence"],
            speech_evidence=evidence["speech_evidence"],
            prediction_details=evidence["prediction_details"]
        )
        evidence["scene_description"] = desc_res["structured"]
        evidence["formatted_description"] = desc_res["formatted_text"]
        final_description = description or desc_res["formatted_text"]

        model_checksum = inference_res["model_identification"]["sha256"]

        # Complete in Database
        database.complete_job(
            job_id=job_id,
            prediction=pred_class,
            confidence=confidence,
            probabilities=class_scores,
            evidence=evidence,
            model_identifier=inference_res["model_identification"]["model_file"],
            model_checksum=model_checksum,
            video_modality=True,
            audio_modality=inference_res["modality_availability"]["audio"],
            speech_modality=inference_res["modality_availability"]["speech"],
            transcript=transcript,
            description=final_description,
            report_reference=f"REP_{job_id}",
            structured_description=desc_res["structured"]
        )

        # Trigger Automated Emergency Warning Asynchronously (Non-blocking, rule-governed)
        if pred_class in ["Fire", "Accident"]:
            try:
                from services.whatsapp_service import send_alert_async
                job_record = database.get_job(job_id)
                video_identifier = (job_record.get("filename") if job_record else None) or video_path.name
                send_alert_async(
                    prediction=pred_class,
                    confidence=confidence,
                    video_identifier=video_identifier,
                    job_id=job_id
                )
            except Exception as notify_err:
                print(f"Warning: Non-critical emergency notification failed to spawn for {job_id}: {notify_err}")

        # Pre-generate and store reports as managed files
        for fmt in ["json", "markdown", "html", "text"]:
            try:
                content, media_type, download_name = generate_job_report(job_id, fmt=fmt)
                if isinstance(content, str):
                    rep_art = save_report_as_artifact(job_id, fmt, content, download_name)
                    database.add_artifact(
                        artifact_id=rep_art["artifact_id"],
                        job_id=job_id,
                        artifact_type="report",
                        filename=rep_art["filename"],
                        storage_relative_path=rep_art["storage_relative_path"],
                        file_size_bytes=rep_art["file_size_bytes"],
                        mime_type=rep_art["mime_type"],
                        file_hash_sha256=rep_art["file_hash_sha256"],
                        metadata=rep_art["metadata"]
                    )
            except Exception as rep_err:
                print(f"Warning: Failed to pre-generate {fmt} report for {job_id}: {rep_err}")

    except Exception as e:
        database.fail_job(job_id, str(e))
    finally:
        _ACTIVE_FUTURES.pop(job_id, None)
        _CANCEL_REQUESTED.discard(job_id)

def _extract_frames_and_metadata(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0:
        capture.release()
        raise ValueError("Video contains no readable frames.")
        
    if total_frames == 1:
        indices = np.zeros(NUM_FRAMES, dtype=np.int64)
    else:
        indices = np.round(np.linspace(0, total_frames - 1, NUM_FRAMES)).astype(np.int64)
        
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()
        if not success:
            capture.release()
            raise RuntimeError(f"Could not read frame {index} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    capture.release()
    return np.asarray(frames, dtype=np.uint8), indices, total_frames, fps

def _extract_full_audio(video_path: Path, sample_rate: int = AUDIO_SAMPLE_RATE):
    import subprocess
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "pipe:1"
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        return np.zeros(0, dtype=np.float32), sample_rate
    return np.frombuffer(process.stdout, dtype=np.float32), sample_rate

def _waveform_to_logmel(waveform, sample_rate=AUDIO_SAMPLE_RATE, duration_seconds=AUDIO_FIXED_SECONDS):
    target_length = int(sample_rate * duration_seconds)
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)))
    else:
        waveform = waveform[:target_length]
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=MEL_BINS, power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel[..., np.newaxis].astype(np.float32)

def _extract_fixed_audio_logmel(video_path: Path):
    waveform, sr = _extract_full_audio(video_path)
    dur = len(waveform) / sr if sr > 0 else 0.0
    if len(waveform) == 0:
        return np.zeros((MEL_BINS, 313, 1), dtype=np.float32), False, 0.0, 0
    window_length = int(sr * AUDIO_FIXED_SECONDS)
    if len(waveform) <= window_length:
        windows = [waveform]
    else:
        max_start = len(waveform) - window_length
        number_of_windows = min(4, max(1, int(np.ceil(len(waveform) / window_length))))
        starts = np.linspace(0, max_start, number_of_windows).astype(int)
        windows = [waveform[start:start + window_length] for start in starts]
    mel_windows = [_waveform_to_logmel(w, sample_rate=sr) for w in windows]
    averaged = np.mean(np.stack(mel_windows, axis=0), axis=0).astype(np.float32)
    return averaged, True, dur, len(windows)

def _transcribe_video_kannada(video_path: Path, asr_pipe):
    import unicodedata, re
    extract_res = extract_audio_waveform(video_path, sample_rate=AUDIO_SAMPLE_RATE)
    waveform = extract_res["waveform"]
    sample_rate = extract_res["sample_rate"]
    duration = extract_res["duration_seconds"]
    if duration < 0.50 or not extract_res["has_audio_stream"] or extract_res["is_silent"]:
        return {"transcript": "", "accepted": False, "kannada_ratio": 0.0}
        
    result = asr_pipe(
        {"raw": waveform, "sampling_rate": sample_rate},
        return_timestamps=False,
        generate_kwargs={"max_length": ASR_MAX_LENGTH, "no_repeat_ngram_size": 3, "num_beams": 1, "do_sample": False}
    )
    raw_text = result.get("text", "")
    normalized = unicodedata.normalize("NFC", str(raw_text))
    cleaned = re.sub(r"\s+", " ", normalized).strip()
    
    words = cleaned.split() if cleaned else []
    word_count = len(words)
    alphabetic = [c for c in cleaned if c.isalpha()]
    kannada_chars = [c for c in alphabetic if "\u0C80" <= c <= "\u0CFF"]
    kannada_ratio = (len(kannada_chars) / len(alphabetic)) if alphabetic else 0.0
    words_per_second = word_count / duration if duration > 0 else 0.0
    
    accepted = all([
        duration >= 0.50,
        word_count >= 2,
        kannada_ratio >= 0.70,
        words_per_second <= 5.0
    ])
    
    return {
        "transcript": cleaned if accepted else "",
        "accepted": accepted,
        "kannada_ratio": round(kannada_ratio, 3)
    }

def generate_job_report(job_id: str, fmt: str = "json") -> Tuple[Any, str, str]:
    """Generates an authoritative analysis report using reporting_service."""
    from services.reporting_service import generate_detailed_analysis_report
    return generate_detailed_analysis_report(job_id=job_id, fmt=fmt)
