import os
import shutil
import tempfile
from pathlib import Path
import unicodedata
import re
import subprocess
from contextlib import asynccontextmanager
from urllib.parse import quote
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import cv2
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Response, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
import database
from services.whatsapp_service import send_alert, send_alert_async, get_whatsapp_config
from services.twilio_service import trigger_twilio_call, trigger_twilio_message
from services.model_metadata_service import build_model_metadata, check_model_readiness, compute_sha256
from services.reporting_service import (
    generate_detailed_analysis_report,
    generate_project_level_report,
    compare_analysis_jobs
)
from services.job_engine import (
    submit_analysis_job,
    validate_video_file,
    cancel_job,
    generate_job_report,
    get_job_queue_pending_count,
    UPLOADS_DIR
)
from services.storage_manager import (
    get_storage_stats,
    run_storage_cleanup,
    resolve_safe_path
)
from services.evaluation_service import get_evaluation_and_dataset_metadata
from services.system_service import get_system_operational_stats
from services.video_preprocessor import extract_video_frames_with_metadata
from services.audio_preprocessor import compute_audio_logmel_features
from services.model_inference_service import (
    load_multimodal_production_model,
    run_direct_multimodal_inference,
    get_production_model_metadata,
    CLASS_INDEX_MAP,
    CLASS_NAMES as PROD_CLASS_NAMES,
    MODEL_FILENAME
)
from services.frame_attribution_service import (
    compute_evidence_based_frame_attributions,
    compute_gradient_frame_attributions,
    CAUSATION_DISCLAIMER
)
from services.speech_recognition_service import (
    transcribe_video_speech,
    transcribe_audio,
    clean_and_normalize_unicode
)
from services.scene_description_service import (
    generate_evidence_grounded_scene_description
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
import tensorflow as tf
from transformers import AutoProcessor, AutoImageProcessor, AutoModel, AutoModelForSpeechSeq2Seq, pipeline
from sentence_transformers import SentenceTransformer

# Verified Class Mapping for best_multimodal_model.keras: 0=Accident, 1=Fire, 2=Normal
CLASS_NAMES = PROD_CLASS_NAMES
NUM_FRAMES = 12
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
AUDIO_SAMPLE_RATE = 16000
AUDIO_FIXED_SECONDS = 5.0
MEL_BINS = 64
N_FFT = 1024
HOP_LENGTH = 256
ASR_MODEL_ID = "vasista22/whisper-kannada-small"
ASR_LANGUAGE = "kn"
ASR_CHUNK_SECONDS = 30
ASR_MAX_LENGTH = 448
VISUAL_MODEL_ID = "facebook/convnextv2-tiny-1k-224"
VISUAL_FEATURE_DIM = 768
TEXT_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_EMBEDDING_DIM = 384

GPU_AVAILABLE = torch.cuda.is_available()
ASR_DEVICE = 0 if GPU_AVAILABLE else -1
TORCH_DEVICE = torch.device("cuda" if GPU_AVAILABLE else "cpu")

if GPU_AVAILABLE:
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:
        print(f"TensorFlow GPU memory-growth setup warning: {exc}")

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading multimodal production keras model...")
    multimodal_model = load_multimodal_production_model()
    models["multimodal_model"] = multimodal_model
    
    print("Loading Visual Model...")
    visual_processor = AutoImageProcessor.from_pretrained(VISUAL_MODEL_ID)
    visual_model = AutoModel.from_pretrained(
        VISUAL_MODEL_ID,
        dtype=torch.float16 if GPU_AVAILABLE else torch.float32
    ).to(TORCH_DEVICE).eval()
    for param in visual_model.parameters():
        param.requires_grad = False
    models["visual_processor"] = visual_processor
    models["visual_model"] = visual_model

    print("Loading Text Embedding Model...")
    text_encoder = SentenceTransformer(TEXT_MODEL_ID, device="cuda" if GPU_AVAILABLE else "cpu")
    models["text_encoder"] = text_encoder
    
    print("Loading ASR Model...")
    asr_dtype = torch.float16 if GPU_AVAILABLE else torch.float32
    asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_MODEL_ID, dtype=asr_dtype, low_cpu_mem_usage=True
    )
    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=asr_model,
        tokenizer=asr_processor.tokenizer,
        feature_extractor=asr_processor.feature_extractor,
        chunk_length_s=ASR_CHUNK_SECONDS,
        batch_size=1,
        device=ASR_DEVICE,
        dtype=asr_dtype
    )
    forced_decoder_ids = asr_processor.get_decoder_prompt_ids(language=ASR_LANGUAGE, task="transcribe")
    asr_pipe.model.config.forced_decoder_ids = forced_decoder_ids
    if hasattr(asr_pipe.model, "generation_config"):
        asr_pipe.model.generation_config.forced_decoder_ids = forced_decoder_ids
        asr_pipe.model.generation_config.max_length = ASR_MAX_LENGTH
    models["asr_pipe"] = asr_pipe
    print("All models loaded.")
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_uniform_frame_indices(total_frames, num_frames=NUM_FRAMES):
    if total_frames <= 0:
        raise ValueError("Video contains no readable frames.")
    if total_frames == 1:
        return np.zeros(num_frames, dtype=np.int64)
    indices = np.linspace(0, total_frames - 1, num_frames)
    return np.round(indices).astype(np.int64)

def extract_video_frames(video_path, num_frames=NUM_FRAMES, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = get_uniform_frame_indices(total_frames, num_frames)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()
        if not success:
            capture.release()
            raise RuntimeError(f"Could not read frame {index} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    capture.release()
    return np.asarray(frames, dtype=np.uint8)

def extract_full_audio(video_path, sample_rate=AUDIO_SAMPLE_RATE):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1"]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        return np.zeros(0, dtype=np.float32), sample_rate
    return np.frombuffer(process.stdout, dtype=np.float32), sample_rate

def waveform_to_logmel(waveform, sample_rate=AUDIO_SAMPLE_RATE, duration_seconds=AUDIO_FIXED_SECONDS):
    target_length = int(sample_rate * duration_seconds)
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)))
    else:
        waveform = waveform[:target_length]
    mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=MEL_BINS, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel[..., np.newaxis].astype(np.float32)

def extract_fixed_audio_logmel(video_path):
    waveform, sr = extract_full_audio(video_path)
    if len(waveform) == 0:
        return np.zeros((MEL_BINS, 313, 1), dtype=np.float32), False
    window_length = int(sr * AUDIO_FIXED_SECONDS)
    if len(waveform) <= window_length:
        windows = [waveform]
    else:
        max_start = len(waveform) - window_length
        number_of_windows = min(4, max(1, int(np.ceil(len(waveform) / window_length))))
        starts = np.linspace(0, max_start, number_of_windows).astype(int)
        windows = [waveform[start:start + window_length] for start in starts]
    mel_windows = [waveform_to_logmel(w, sample_rate=sr) for w in windows]
    return np.mean(np.stack(mel_windows, axis=0), axis=0).astype(np.float32), True

def normalize_asr_text(text):
    if text is None: return ""
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", text).strip()

def score_asr_transcript(transcript, audio_duration):
    transcript = normalize_asr_text(transcript)
    words = transcript.split() if transcript else []
    word_count = len(words)
    alphabetic = [c for c in transcript if c.isalpha()]
    kannada_ratio = (len([c for c in alphabetic if "\u0C80" <= c <= "\u0CFF"]) / len(alphabetic)) if alphabetic else 0.0
    unique_ratio = (len(set(words)) / len(words)) if words else 0.0
    from collections import Counter
    dominant_ratio = (Counter(words).most_common(1)[0][1] / len(words)) if words else 0.0
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)] if len(words) >= 4 else []
    repeated_ratio = (sum(c for c in Counter(bigrams).values() if c > 1) / len(bigrams)) if bigrams else 0.0
    words_per_second = word_count / audio_duration if audio_duration > 0 else 0.0
    accepted = all([audio_duration >= 0.50, word_count >= 2, kannada_ratio >= 0.70, unique_ratio >= 0.30, dominant_ratio <= 0.60, repeated_ratio <= 0.60, words_per_second <= 5.0])
    return {"transcript": transcript, "accepted": accepted}

def transcribe_video_kannada(video_path):
    waveform, sample_rate = extract_full_audio(video_path, sample_rate=AUDIO_SAMPLE_RATE)
    duration = len(waveform) / sample_rate
    if duration < 0.50:
        return {"transcript": "", "accepted": False}
    result = models["asr_pipe"]({"raw": waveform, "sampling_rate": sample_rate}, return_timestamps=False, generate_kwargs={"max_length": ASR_MAX_LENGTH, "no_repeat_ngram_size": 3, "num_beams": 1, "do_sample": False})
    raw_transcript = normalize_asr_text(result.get("text", ""))
    quality = score_asr_transcript(raw_transcript, duration)
    return quality if quality["accepted"] else {"transcript": "", "accepted": False}

@app.get("/")
def root():
    return {
        "service": "Kannada Audio-Visual Scene Understanding API",
        "status": "online",
        "docs": "/docs",
        "health": "/api/health",
        "version": "2.0.0"
    }

@app.get("/health")
@app.get("/api/health")
def health():
    db_ok = True
    try:
        conn = database.get_connection()
        conn.execute("SELECT 1")
        conn.close()
    except Exception:
        db_ok = False

    models_ready = bool(models.get("multimodal_model"))
    is_healthy = db_ok and models_ready

    return {
        "status": "healthy" if is_healthy else "degraded",
        "healthy": is_healthy,
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "database": "ok" if db_ok else "error",
            "multimodal_model": "loaded" if models.get("multimodal_model") else "unloaded",
            "visual_backbone": "loaded" if models.get("visual_model") else "unloaded",
            "text_encoder": "loaded" if models.get("text_encoder") else "unloaded",
            "asr_pipeline": "loaded" if models.get("asr_pipe") else "unloaded",
            "storage": "writable" if os.access(str(database.BACKEND_DIR), os.W_OK) else "readonly"
        }
    }

@app.get("/api/readiness")
def readiness():
    return check_model_readiness(models)

@app.get("/api/model/metadata")
def get_model_metadata_endpoint():
    return build_model_metadata()

@app.get("/api/model/readiness")
def get_model_readiness_endpoint():
    return check_model_readiness(models)

class JobUpdateRequest(BaseModel):
    description: Optional[str] = None
    notification_status: Optional[str] = None

@app.post("/api/jobs/analyze", status_code=status.HTTP_202_ACCEPTED)
async def create_analysis_job_endpoint(
    response: Response,
    video: UploadFile = File(...),
    force_reanalysis: bool = Query(False, description="Bypass duplicate cache and re-run full analysis"),
    description: Optional[str] = Query(None, description="Optional incident description or analyst notes")
):
    valid, reason = validate_video_file(video.filename, 1024 * 1024)
    if not valid:
        raise HTTPException(status_code=400, detail=reason)

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / video.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        file_size = temp_path.stat().st_size
        valid, reason = validate_video_file(video.filename, file_size)
        if not valid:
            raise HTTPException(status_code=400, detail=reason)

        job_record, was_duplicate = submit_analysis_job(
            filename=video.filename,
            temp_source_path=temp_path,
            file_size_bytes=file_size,
            models_dict=models,
            force_reanalysis=force_reanalysis,
            description=description
        )
        if was_duplicate:
            response.status_code = status.HTTP_200_OK
        return job_record
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/api/jobs/compare")
def compare_jobs_endpoint(
    job_1: str = Query(..., description="First job ID to compare"),
    job_2: str = Query(..., description="Second job ID to compare")
):
    try:
        return compare_analysis_jobs(job_id_1=job_1, job_id_2=job_2)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}")
def get_job_endpoint(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job

@app.patch("/api/jobs/{job_id}")
def update_job_endpoint(job_id: str, payload: JobUpdateRequest):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if payload.description is not None:
        database.update_job_description(job_id, payload.description)
    if payload.notification_status is not None:
        database.update_job_notification_status(job_id, payload.notification_status)
    return database.get_job(job_id)

@app.get("/api/jobs/{job_id}/artifacts")
def list_job_artifacts_endpoint(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    artifacts = database.get_job_artifacts(job_id)
    return {
        "job_id": job_id,
        "total_artifacts": len(artifacts),
        "artifacts": artifacts
    }

@app.get("/api/jobs/{job_id}/artifacts/{artifact_id}")
def download_job_artifact_endpoint(job_id: str, artifact_id: str):
    artifact = database.get_artifact(artifact_id)
    if not artifact or artifact["job_id"] != job_id:
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found for job '{job_id}'.")
    
    storage_rel = artifact.get("_storage_path")
    if not storage_rel:
        raise HTTPException(status_code=404, detail="Artifact storage path is unavailable.")
    
    try:
        safe_file_path = resolve_safe_path(storage_rel)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
        
    if not safe_file_path.exists() or not safe_file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file does not exist on disk.")
        
    return FileResponse(
        path=str(safe_file_path),
        media_type=artifact.get("mime_type", "application/octet-stream"),
        filename=artifact.get("filename", f"artifact_{artifact_id}")
    )

@app.get("/api/jobs/{job_id}/attributions")
def get_job_attributions_endpoint(
    job_id: str,
    sort_by: str = Query("rank", pattern="^(rank|time)$")
):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    
    order_by_rank = (sort_by == "rank")
    attributions = database.get_frame_attributions(job_id, order_by_rank=order_by_rank)
    
    influential_meta = job.get("evidence", {}).get("visual_evidence", {}).get("influential_frames", {})
    
    target_class = influential_meta.get("target_class") or job.get("prediction")
    baseline_score = influential_meta.get("baseline_score") or job.get("confidence")
    attribution_method = influential_meta.get("attribution_method", "temporal_frame_ablation (neutral_gray)")
    perturbation_strategy = influential_meta.get("perturbation_strategy", "neutral_gray")
    reliability = influential_meta.get("reliability", "reliable_sensitivity" if len(attributions) > 0 else "unavailable")
    causation_disclaimer = influential_meta.get("causation_disclaimer", CAUSATION_DISCLAIMER)
    score_range = influential_meta.get("score_range")
    warnings = influential_meta.get("warnings", [])

    return {
        "job_id": job_id,
        "target_class": target_class,
        "baseline_score": baseline_score,
        "attribution_method": attribution_method,
        "perturbation_strategy": perturbation_strategy,
        "causation_disclaimer": causation_disclaimer,
        "reliability": reliability,
        "score_range": score_range,
        "sort_by": sort_by,
        "total_frames_attributed": len(attributions),
        "warnings": warnings,
        "attributions": attributions
    }

@app.get("/api/jobs/{job_id}/transcripts")
def get_job_transcripts_endpoint(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    transcripts = database.get_speech_transcripts(job_id)
    return {
        "job_id": job_id,
        "total_transcripts": len(transcripts),
        "transcripts": transcripts
    }

@app.get("/api/jobs/{job_id}/description")
def get_job_description_endpoint(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    struct_desc = job.get("structured_description")
    formatted_desc = job.get("description")
    evidence_desc = job.get("evidence", {}).get("scene_description")
    return {
        "job_id": job_id,
        "prediction": job.get("prediction"),
        "confidence": job.get("confidence"),
        "structured": struct_desc or evidence_desc or {},
        "formatted_text": formatted_desc or ""
    }

@app.get("/api/jobs/{job_id}/thumbnail")
def get_job_thumbnail_endpoint(job_id: str):
    artifacts = database.get_job_artifacts(job_id)
    thumb = next((a for a in artifacts if a["artifact_type"] == "thumbnail"), None)
    if not thumb:
        raise HTTPException(status_code=404, detail=f"Thumbnail not available for job '{job_id}'.")
    return download_job_artifact_endpoint(job_id, thumb["artifact_id"])

@app.get("/api/jobs/{job_id}/filmstrip")
def get_job_filmstrip_endpoint(job_id: str):
    artifacts = database.get_job_artifacts(job_id)
    filmstrip = next((a for a in artifacts if a["artifact_type"] == "filmstrip"), None)
    if not filmstrip:
        raise HTTPException(status_code=404, detail=f"Filmstrip not available for job '{job_id}'.")
    return download_job_artifact_endpoint(job_id, filmstrip["artifact_id"])

@app.get("/api/jobs/{job_id}/frames")
def get_job_frames_endpoint(job_id: str):
    artifacts = database.get_job_artifacts(job_id)
    frames = [a for a in artifacts if a["artifact_type"] in ("frame", "keyframe")]
    return {
        "job_id": job_id,
        "total_extracted_frames": len(frames),
        "frames": frames
    }

@app.get("/api/jobs/{job_id}/spectrogram")
def get_job_spectrogram_endpoint(job_id: str):
    artifacts = database.get_job_artifacts(job_id)
    spectrogram = next((a for a in artifacts if a["artifact_type"] == "spectrogram"), None)
    if not spectrogram:
        raise HTTPException(status_code=404, detail=f"Spectrogram not available for job '{job_id}'.")
    return download_job_artifact_endpoint(job_id, spectrogram["artifact_id"])

@app.get("/api/jobs/{job_id}/audio")
def get_job_audio_metadata_endpoint(job_id: str):
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    
    evidence = job.get("evidence", {}) or {}
    audio_ev = evidence.get("audio_evidence", {}) or {}
    
    artifacts = database.get_job_artifacts(job_id)
    spectrogram = next((a for a in artifacts if a["artifact_type"] == "spectrogram"), None)
    spectrogram_url = f"/api/jobs/{job_id}/spectrogram" if spectrogram else None
    
    return {
        "job_id": job_id,
        "audio_available": audio_ev.get("audio_available", False),
        "has_audio_stream": audio_ev.get("has_audio_stream", False),
        "is_silent": audio_ev.get("is_silent", False),
        "status": audio_ev.get("status", "unknown"),
        "duration_seconds": audio_ev.get("duration_seconds", 0.0),
        "sample_rate_hz": audio_ev.get("sample_rate_hz", 16000),
        "mel_bins": audio_ev.get("mel_bins", 64),
        "time_steps": audio_ev.get("time_steps", 313),
        "windows_averaged": audio_ev.get("windows_averaged", 0),
        "window_starts_sec": audio_ev.get("window_starts_sec", []),
        "spectrogram_shape": audio_ev.get("spectrogram_shape", [64, 313, 1]),
        "waveform_stats": audio_ev.get("waveform_stats", {}),
        "axes_metadata": audio_ev.get("axes_metadata", {}),
        "spectrogram_url": spectrogram_url,
        "warnings": audio_ev.get("warnings", [])
    }

@app.post("/api/jobs/{job_id}/cancel")
def cancel_job_endpoint(job_id: str):
    success = cancel_job(job_id)
    if not success:
        job = database.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' cannot be cancelled (current status: {job['status']})."
        )
    return {"job_id": job_id, "status": "cancelled", "message": "Job cancellation initiated."}

@app.get("/api/jobs")
def list_jobs_endpoint(
    query: Optional[str] = Query(None, description="Search by filename or job ID"),
    status: Optional[str] = Query(None, description="Filter by status (queued, processing, completed, failed, cancelled)"),
    prediction: Optional[str] = Query(None, description="Filter by prediction (Fire, Accident, Normal)"),
    start_date: Optional[str] = Query(None, description="Filter by start date ISO"),
    end_date: Optional[str] = Query(None, description="Filter by end date ISO"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort_desc: bool = Query(True, description="Sort descending by creation date")
):
    jobs, total, pagination = database.list_jobs(
        query=query, status=status, prediction=prediction,
        start_date=start_date, end_date=end_date, limit=limit, offset=offset, sort_desc=sort_desc
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "pagination": pagination,
        "jobs": jobs
    }

@app.get("/api/storage/policy")
def get_storage_policy_endpoint():
    return get_storage_stats()

@app.post("/api/storage/cleanup")
def cleanup_storage_endpoint(
    retention_days: Optional[int] = Query(None, description="Purge artifacts older than N days"),
    max_storage_mb: Optional[int] = Query(None, description="Storage ceiling in MB"),
    clean_orphans: bool = Query(True, description="Remove files missing from database registry"),
    dry_run: bool = Query(False, description="Preview actions without deleting files")
):
    conn = database.get_connection()
    try:
        return run_storage_cleanup(
            conn,
            retention_days=retention_days,
            max_storage_mb=max_storage_mb,
            clean_orphans=clean_orphans,
            dry_run=dry_run
        )
    finally:
        conn.close()

@app.get("/api/jobs/{job_id}/report")
def get_job_report_endpoint(
    job_id: str,
    format: str = Query("json", pattern="^(json|markdown|html|text|csv|pdf)$")
):
    try:
        content, media_type, download_name = generate_detailed_analysis_report(job_id=job_id, fmt=format)
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reports/project")
def get_project_report_endpoint(
    format: str = Query("json", pattern="^(json|markdown|html|csv|pdf)$")
):
    try:
        content, media_type, download_name = generate_project_level_report(fmt=format)
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/notifications/dispatches")
def get_notifications_dispatches_endpoint(
    job_id: Optional[str] = Query(None, description="Filter dispatches by specific job ID"),
    limit: int = Query(50, ge=1, le=200)
):
    if job_id:
        job = database.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        dispatches = database.get_job_dispatches(job_id)
    else:
        dispatches = database.get_recent_dispatches(limit=limit)
    return {
        "total": len(dispatches),
        "limit": limit,
        "dispatches": dispatches
    }

@app.get("/api/notifications/config")
def get_notifications_config_endpoint():
    cfg = get_whatsapp_config()
    return {
        "test_mode": cfg["test_mode"],
        "confidence_threshold": cfg["confidence_threshold"],
        "cooldown_minutes": cfg["cooldown_minutes"],
        "allowed_classes": cfg["allowed_classes"],
        "has_test_number": bool(cfg["test_number"]),
        "has_fire_number": bool(cfg["fire_number"]),
        "has_accident_number": bool(cfg["accident_number"])
    }

class NotificationTestRequest(BaseModel):
    event: str = "Fire"
    confidence: float = 0.95
    video_identifier: str = "simulated_test_sample.mp4"

@app.post("/api/notifications/test")
def test_notification_endpoint(req: NotificationTestRequest):
    if req.event not in ["Fire", "Accident"]:
        raise HTTPException(status_code=400, detail="Only Fire and Accident events trigger notifications.")
    return send_alert(
        prediction=req.event,
        confidence=req.confidence,
        video_identifier=req.video_identifier,
        force=True
    )

@app.get("/api/evaluation/metadata")
def get_evaluation_metadata_endpoint():
    return get_evaluation_and_dataset_metadata()

@app.get("/api/system/stats")
def get_system_stats_endpoint():
    return get_system_operational_stats(models, job_queue_size=get_job_queue_pending_count())

class WhatsAppAlertRequest(BaseModel):
    prediction: str
    confidence: float
    video_identifier: str

@app.post("/send-whatsapp-alert")
async def send_whatsapp_alert_endpoint(request: WhatsAppAlertRequest, background_tasks: BackgroundTasks):
    if request.prediction not in ["Fire", "Accident"]:
        return {"status": "skipped", "message": "Normal predictions do not trigger alerts."}
    
    background_tasks.add_task(send_alert, request.prediction, request.confidence, request.video_identifier)
    return {"status": "success", "message": "WhatsApp alert queued successfully."}

@app.get("/api/whatsapp/link")
def get_whatsapp_link(event: str, confidence: float, video_identifier: str):
    if event not in ["Fire", "Accident"]:
        raise HTTPException(status_code=400, detail="Invalid event type for WhatsApp")

    test_mode = os.getenv("WHATSAPP_TEST_MODE", "false").lower() == "true"
    if test_mode:
        phone = os.getenv("WHATSAPP_TEST_NUMBER")
    else:
        if event == "Fire":
            phone = os.getenv("FIRE_WHATSAPP_NUMBER")
        elif event == "Accident":
            phone = os.getenv("ACCIDENT_WHATSAPP_NUMBER")
    
    if not phone:
        raise HTTPException(status_code=500, detail="WhatsApp recipient number not configured.")
        
    clean_phone = "".join(filter(str.isdigit, phone))
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conf_str = f"{(confidence * 100):.2f}%"
    
    message = (
        f"🚨 AI VIDEO ALERT\n\n"
        f"Event: {event.upper()}\n\n"
        f"Confidence: {conf_str}\n\n"
        f"Video: {video_identifier}\n\n"
        f"Time: {timestamp}\n\n"
        f"The video analysis system has detected a POSSIBLE {event.upper()} INCIDENT.\n\n"
        f"Please verify the situation and take appropriate action.\n\n"
        f"This is an AI-generated detection and should be independently verified."
    )
    
    encoded_message = quote(message)
    url = f"https://web.whatsapp.com/send?phone={clean_phone}&text={encoded_message}"
    try:
        database.record_dispatch(job_id=None, dispatch_type="whatsapp_link", event_type=event, recipient=clean_phone, status="success")
    except Exception:
        pass
    return {"url": url}

class TwilioAlertRequest(BaseModel):
    event: str
    confidence: float
    video_identifier: str
    event_id: str

@app.post("/api/emergency/call")
def emergency_call(request: TwilioAlertRequest):
    if request.event not in ["Fire", "Accident"]:
        raise HTTPException(status_code=400, detail="Invalid event type")
    result = trigger_twilio_call(request.event, request.confidence, request.video_identifier, request.event_id)
    try:
        database.record_dispatch(
            job_id=request.event_id,
            dispatch_type="twilio_call",
            event_type=request.event,
            recipient="configured_recipient",
            status="success" if result.get("success") else "failed",
            external_id=result.get("call_sid")
        )
    except Exception:
        pass
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to initiate call"))
    return result

@app.post("/api/emergency/message")
def emergency_message(request: TwilioAlertRequest):
    if request.event not in ["Fire", "Accident"]:
        raise HTTPException(status_code=400, detail="Invalid event type")
    result = trigger_twilio_message(request.event, request.confidence, request.video_identifier, request.event_id)
    try:
        database.record_dispatch(
            job_id=request.event_id,
            dispatch_type="twilio_sms",
            event_type=request.event,
            recipient="configured_recipient",
            status="success" if result.get("success") else "failed",
            external_id=result.get("message_sid")
        )
    except Exception:
        pass
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to send message"))
    return result

@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported video format")
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / video.filename
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        video_res = extract_video_frames_with_metadata(temp_video_path)
        frames = video_res["frames"]
        frame_meta = video_res
        audio_res = compute_audio_logmel_features(temp_video_path)
        mel = audio_res.get("mel_tensor", audio_res.get("logmel_spectrogram"))
        audio_available = audio_res["audio_available"]

        asr_result = transcribe_video_speech(temp_video_path, language="auto", asr_pipe=models.get("asr_pipe"))
        transcript = asr_result.get("transcript")

        inference_res = run_direct_multimodal_inference(
            video_frames=frames,
            audio_spectrogram=mel,
            audio_available=audio_available,
            transcript=transcript
        )

        pred_class = inference_res["predicted_class"]
        confidence = inference_res["confidence"]
        probabilities = inference_res["class_scores"]

        # Evidence-based influential frame analysis (temporal frame ablation)
        frame_timestamps = frame_meta.get("frame_timestamps_sec")
        if not frame_timestamps or len(frame_timestamps) != NUM_FRAMES:
            frame_timestamps = [float(i * 0.5) for i in range(NUM_FRAMES)]

        attribution_res = compute_evidence_based_frame_attributions(
            video_frames=frames,
            frame_timestamps_sec=frame_timestamps,
            audio_spectrogram=mel,
            audio_available=audio_available,
            transcript=transcript,
            target_class=pred_class
        )

        # Log into persistent database for unified history & auditability
        file_size = temp_video_path.stat().st_size
        sync_job_id = f"job_sync_{temp_video_path.stem[:8]}_{int(datetime.now().timestamp())}"
        try:
            permanent_save_path = UPLOADS_DIR / f"{sync_job_id}_{video.filename}"
            shutil.copyfile(str(temp_video_path), str(permanent_save_path))
            file_hash = compute_sha256(permanent_save_path)
            
            database.create_job(
                job_id=sync_job_id,
                filename=video.filename,
                file_path=str(permanent_save_path),
                file_size_bytes=file_size,
                file_hash_sha256=file_hash
            )
            evidence = {
                "prediction_details": inference_res,
                "visual_evidence": {
                    "sampled_frame_count": NUM_FRAMES,
                    "resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
                    "frame_timestamps_sec": frame_timestamps,
                    "influential_frames": {
                        "attribution_method": attribution_res["attribution_method"],
                        "target_class": attribution_res["target_class"],
                        "baseline_score": attribution_res["baseline_score"],
                        "causation_disclaimer": attribution_res["causation_disclaimer"],
                        "reliability": attribution_res["reliability"],
                        "score_range": attribution_res["score_range"],
                        "ranked_frames": attribution_res["frames"],
                        "warnings": attribution_res["warnings"]
                    }
                },
                "audio_evidence": {
                    "mel_bins": MEL_BINS,
                    "sample_rate_hz": AUDIO_SAMPLE_RATE,
                    "audio_available": audio_available,
                    "has_audio_stream": audio_res.get("has_audio_stream", True),
                    "is_silent": audio_res.get("is_silent", False),
                    "duration_seconds": audio_res.get("duration_seconds", 0.0),
                    "waveform_stats": audio_res.get("waveform_stats")
                },
                "speech_evidence": {
                    "transcript": transcript,
                    "transcript_accepted": asr_result.get("accepted"),
                    "language": asr_result.get("language", "none"),
                    "word_count": asr_result.get("word_count", 0),
                    "chunks": asr_result.get("chunks", []),
                    "kannada_char_ratio": asr_result.get("kannada_ratio", 0.0),
                    "english_char_ratio": asr_result.get("english_ratio", 0.0)
                }
            }

            # Evidence-grounded scene description generation
            desc_res = generate_evidence_grounded_scene_description(
                video_evidence=evidence["visual_evidence"],
                audio_evidence=evidence["audio_evidence"],
                speech_evidence=evidence["speech_evidence"],
                prediction_details=evidence["prediction_details"]
            )
            evidence["scene_description"] = desc_res["structured"]
            evidence["formatted_description"] = desc_res["formatted_text"]

            model_chk = inference_res["model_identification"]["sha256"]

            database.complete_job(
                job_id=sync_job_id,
                prediction=pred_class,
                confidence=confidence,
                probabilities=probabilities,
                evidence=evidence,
                model_identifier=inference_res["model_identification"]["model_file"],
                model_checksum=model_chk,
                video_modality=True,
                audio_modality=inference_res["modality_availability"]["audio"],
                speech_modality=inference_res["modality_availability"]["speech"],
                transcript=transcript,
                description=desc_res["formatted_text"],
                report_reference=f"REP_{sync_job_id}",
                structured_description=desc_res["structured"]
            )

            # Persist speech transcript separately in speech_transcripts table
            if transcript or asr_result.get("status") != "silent_or_missing_audio":
                database.add_speech_transcript(
                    job_id=sync_job_id,
                    language=asr_result.get("language", "none"),
                    transcript=transcript or "",
                    confidence=1.0 if asr_result.get("accepted") else 0.0,
                    is_accepted=asr_result.get("accepted", False),
                    chunks=asr_result.get("chunks", [])
                )

            for f_attr in attribution_res["frames"]:
                database.add_frame_attribution(
                    job_id=sync_job_id,
                    frame_index=f_attr["frame_index"],
                    timestamp_sec=f_attr["timestamp_sec"],
                    attribution_score=f_attr["attribution_score"],
                    baseline_score=f_attr["baseline_score"],
                    perturbed_score=f_attr["perturbed_score"],
                    score_difference=f_attr["score_difference"],
                    attribution_method=attribution_res["attribution_method"],
                    rank=f_attr["rank"],
                    is_keyframe=f_attr["is_keyframe"],
                    label=f_attr["label"]
                )

            # Trigger Non-blocking Emergency Warning (Rule-gated)
            if pred_class in ["Fire", "Accident"]:
                try:
                    send_alert_async(
                        prediction=pred_class,
                        confidence=confidence,
                        video_identifier=video.filename,
                        job_id=sync_job_id
                    )
                except Exception as notify_err:
                    print(f"Warning: Non-critical WhatsApp alert trigger failed for sync job: {notify_err}")
        except Exception as db_err:
            print(f"Warning: Failed to persist sync prediction to DB: {db_err}")
            sync_job_id = None
            desc_res = generate_evidence_grounded_scene_description(
                video_evidence={
                    "sampled_frame_count": NUM_FRAMES,
                    "resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
                    "frame_timestamps_sec": frame_timestamps,
                    "influential_frames": attribution_res
                },
                audio_evidence=audio_res,
                speech_evidence=asr_result,
                prediction_details=inference_res
            )

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "video_filename": video.filename,
            "probabilities": probabilities,
            "score_semantics": inference_res["score_semantics"],
            "is_calibrated": inference_res["is_calibrated"],
            "calibration_note": inference_res["calibration_note"],
            "modality_availability": inference_res["modality_availability"],
            "speech_recognition": asr_result,
            "scene_description": desc_res["structured"],
            "formatted_description": desc_res["formatted_text"],
            "influential_frames": {
                "target_class": attribution_res["target_class"],
                "baseline_score": attribution_res["baseline_score"],
                "attribution_method": attribution_res["attribution_method"],
                "causation_disclaimer": attribution_res["causation_disclaimer"],
                "reliability": attribution_res["reliability"],
                "score_range": attribution_res["score_range"],
                "ranked_frames": attribution_res["frames"],
                "warnings": attribution_res["warnings"]
            },
            "timings": inference_res["timings"],
            "model_identification": inference_res["model_identification"],
            "warnings": inference_res["warnings"],
            "job_id": sync_job_id
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
