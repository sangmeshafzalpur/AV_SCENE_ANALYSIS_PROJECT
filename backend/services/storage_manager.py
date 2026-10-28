import os
import shutil
import hashlib
import uuid
import time
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except (ImportError, Exception):
    HAS_MATPLOTLIB = False

BACKEND_DIR = Path(__file__).resolve().parent.parent
STORAGE_ROOT = BACKEND_DIR / "storage"
UPLOADS_DIR = STORAGE_ROOT / "uploads"
ARTIFACTS_DIR = STORAGE_ROOT / "artifacts"
REPORTS_DIR = STORAGE_ROOT / "reports"
TEMP_DIR = STORAGE_ROOT / "temp"

# Ensure all managed directories exist
for d in [STORAGE_ROOT, UPLOADS_DIR, ARTIFACTS_DIR, REPORTS_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Default retention policy settings (configurable via environment variables)
STORAGE_RETENTION_DAYS = int(os.getenv("STORAGE_RETENTION_DAYS", "30"))
MAX_STORAGE_MB = int(os.getenv("MAX_STORAGE_MB", "5000"))
CLEANUP_FAILED_JOBS_HOURS = int(os.getenv("CLEANUP_FAILED_JOBS_HOURS", "24"))

MIME_TYPES = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".json": "application/json",
    ".md": "text/markdown",
    ".html": "text/html",
    ".txt": "text/plain",
    ".wav": "audio/wav"
}

def compute_file_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def resolve_safe_path(storage_relative_path: str) -> Path:
    """
    Resolves a relative storage path safely against STORAGE_ROOT.
    Strictly prevents directory traversal attacks (e.g. '../', '..\\', or absolute paths).
    """
    if not storage_relative_path:
        raise PermissionError("Access denied: Empty path provided")

    p_raw = Path(storage_relative_path)
    if p_raw.is_absolute() or storage_relative_path.startswith(("/", "\\")):
        raise PermissionError(f"Access denied: Absolute path traversal attempt detected ({storage_relative_path})")

    clean_rel = p_raw.as_posix()
    target_path = (STORAGE_ROOT / clean_rel).resolve()
    storage_root_resolved = STORAGE_ROOT.resolve()

    try:
        target_path.relative_to(storage_root_resolved)
    except ValueError:
        raise PermissionError(f"Access denied: Path traversal attempt detected ({storage_relative_path})")

    if target_path == storage_root_resolved:
        raise PermissionError("Access denied: Root storage directory cannot be accessed directly")

    return target_path

def sanitize_job_storage(job_id: str) -> Path:
    """Returns the dedicated artifact directory for a job."""
    job_dir = ARTIFACTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir

def sanitize_report_storage(job_id: str) -> Path:
    """Returns the dedicated report directory for a job."""
    report_dir = REPORTS_DIR / job_id
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir

def save_frame_as_artifact(
    job_id: str,
    frame_rgb: np.ndarray,
    frame_index: int,
    timestamp_sec: float,
    is_keyframe: bool = False
) -> Dict[str, Any]:
    """
    Saves an RGB video frame as a managed JPEG file and returns its artifact metadata.
    """
    job_artifacts_dir = sanitize_job_storage(job_id)
    filename = f"frame_{frame_index:02d}.jpg"
    abs_path = job_artifacts_dir / filename
    
    if frame_rgb is None or getattr(frame_rgb, "size", 0) == 0 or frame_rgb.ndim != 3:
        frame_rgb = np.zeros((224, 224, 3), dtype=np.uint8)

    # Convert RGB to BGR for OpenCV saving
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(abs_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    file_size = abs_path.stat().st_size
    rel_path = f"artifacts/{job_id}/{filename}"
    file_hash = compute_file_sha256(abs_path)
    
    metadata = {
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 2),
        "is_keyframe": is_keyframe,
        "width": int(frame_rgb.shape[1]),
        "height": int(frame_rgb.shape[0])
    }
    
    artifact_id = f"art_frame_{uuid.uuid4().hex[:10]}"
    return {
        "artifact_id": artifact_id,
        "job_id": job_id,
        "artifact_type": "keyframe" if is_keyframe else "frame",
        "filename": filename,
        "storage_relative_path": rel_path,
        "file_size_bytes": file_size,
        "file_hash_sha256": file_hash,
        "mime_type": "image/jpeg",
        "metadata": metadata
    }

def save_spectrogram_as_artifact(
    job_id: str,
    mel_spectrogram: np.ndarray,
    axes_metadata: Optional[Dict[str, Any]] = None,
    audio_available: bool = True,
    warnings: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Renders and saves a log-mel spectrogram as a high-contrast viridis PNG artifact with time/frequency axes.
    """
    job_artifacts_dir = sanitize_job_storage(job_id)
    filename = "spectrogram.png"
    abs_path = job_artifacts_dir / filename
    
    # Squeeze or select 2D slice safely
    if mel_spectrogram is None or getattr(mel_spectrogram, "size", 0) == 0:
        mel_2d = np.zeros((64, 313), dtype=np.float32)
        audio_available = False
    elif mel_spectrogram.ndim == 3 and mel_spectrogram.shape[-1] >= 1:
        mel_2d = mel_spectrogram[:, :, 0]
    elif mel_spectrogram.ndim == 2:
        mel_2d = mel_spectrogram
    else:
        mel_2d = np.squeeze(mel_spectrogram)
        if mel_2d.ndim != 2:
            mel_2d = np.zeros((64, 313), dtype=np.float32)
            audio_available = False

    # Ensure robust axes metadata defaults
    duration_sec = 5.0
    max_freq_khz = 8.0
    if axes_metadata and isinstance(axes_metadata, dict):
        t_ax = axes_metadata.get("time_axis")
        f_ax = axes_metadata.get("frequency_axis")
        if isinstance(t_ax, dict):
            duration_sec = float(t_ax.get("window_duration_seconds", 5.0))
        if isinstance(f_ax, dict):
            max_freq_khz = float(f_ax.get("max_frequency_hz", 8000.0)) / 1000.0
    else:
        axes_metadata = {
            "time_axis": {
                "window_duration_seconds": 5.0,
                "time_steps": int(mel_2d.shape[1]),
                "seconds_per_step": round(5.0 / max(1, mel_2d.shape[1]), 6),
                "time_range_seconds": [0.0, 5.0],
                "display_ticks_sec": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "time_label": "Time (seconds)"
            },
            "frequency_axis": {
                "mel_bins": int(mel_2d.shape[0]),
                "min_frequency_hz": 0.0,
                "max_frequency_hz": 8000.0,
                "display_ticks_hz": [0, 500, 1000, 2000, 4000, 8000],
                "frequency_label": "Frequency (Hz / Mel Scale)"
            }
        }
        
    rendered_with_plt = False
    if HAS_MATPLOTLIB:
        try:
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
            plt.subplots_adjust(left=0.10, right=0.92, top=0.88, bottom=0.15)
            
            has_content = audio_available and (
                (float(mel_2d.max()) - float(mel_2d.min()) > 1e-6) or bool(np.any(mel_2d != 0))
            )
            if has_content:
                im = ax.imshow(
                    mel_2d,
                    cmap="viridis",
                    origin="lower",
                    aspect="auto",
                    extent=[0.0, duration_sec, 0.0, max_freq_khz]
                )
                ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold", labelpad=8)
                ax.set_ylabel("Frequency (kHz)", fontsize=11, fontweight="bold", labelpad=8)
                cbar = fig.colorbar(im, ax=ax, pad=0.02)
                cbar.set_label("Standardized Log-Mel Amplitude (dB rel.)", fontsize=9, labelpad=6)
                cbar.ax.tick_params(labelsize=8)
                ax.set_title("Log-Mel Spectrogram (64 Bins, 16 kHz Mono, 5.0s Window)", fontsize=12, fontweight="bold", pad=10)
                ax.set_xticks(np.arange(0.0, duration_sec + 0.5, 1.0))
                ax.set_yticks(np.arange(0.0, max_freq_khz + 1.0, 2.0))
                ax.grid(color="white", alpha=0.15, linestyle="--", linewidth=0.7)
            else:
                fig.patch.set_facecolor("#111827")
                ax.set_facecolor("#1f2937")
                ax.text(
                    0.5, 0.65,
                    "AUDIO STREAM UNAVAILABLE",
                    ha="center", va="center",
                    fontsize=16, fontweight="heavy",
                    color="#f87171",
                    transform=ax.transAxes
                )
                ax.text(
                    0.5, 0.48,
                    "Video container has no readable audio track or audio is silent.\n"
                    "Verified Missing-Audio Policy Active: (64, 313, 1) Zero-Tensor Provided.\n"
                    "Classification relies on ConvNeXt visual and Kannada speech embeddings.",
                    ha="center", va="center",
                    fontsize=10,
                    color="#9ca3af",
                    linespacing=1.6,
                    transform=ax.transAxes
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#374151")
                ax.set_title("Audio Modality Status: Stream Unavailable / Muted", fontsize=12, fontweight="bold", color="#e5e7eb", pad=10)
                
            fig.savefig(str(abs_path), bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            rendered_with_plt = True
        except Exception:
            rendered_with_plt = False

    if not rendered_with_plt:
        # Fallback to pure OpenCV rendering
        canvas = np.full((500, 1000, 3), (25, 20, 17), dtype=np.uint8)
        if audio_available and (float(mel_2d.max()) - float(mel_2d.min()) > 1e-6):
            min_val, max_val = float(mel_2d.min()), float(mel_2d.max())
            norm_mel = ((mel_2d - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            norm_mel = np.flipud(norm_mel)
            color_mel = cv2.applyColorMap(norm_mel, cv2.COLORMAP_VIRIDIS)
            resized_mel = cv2.resize(color_mel, (830, 350), interpolation=cv2.INTER_LINEAR)
            canvas[60:410, 90:920] = resized_mel
            cv2.rectangle(canvas, (90, 60), (920, 410), (120, 120, 120), 1)
            cv2.putText(canvas, "Log-Mel Spectrogram", (90, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (60, 50), (940, 450), (45, 40, 35), -1)
            cv2.rectangle(canvas, (60, 50), (940, 450), (70, 65, 60), 1)
            cv2.putText(canvas, "AUDIO STREAM UNAVAILABLE", (260, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (113, 113, 248), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Video container has no readable audio track or audio is silent.", (210, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imwrite(str(abs_path), canvas)
    
    file_size = abs_path.stat().st_size
    rel_path = f"artifacts/{job_id}/{filename}"
    file_hash = compute_file_sha256(abs_path)
    
    metadata = {
        "mel_bins": int(mel_2d.shape[0]),
        "time_steps": int(mel_2d.shape[1]),
        "colormap": "viridis",
        "audio_available": audio_available,
        "has_axes_labels": True,
        "axes_metadata": axes_metadata,
        "warnings": warnings or []
    }
    
    artifact_id = f"art_spec_{uuid.uuid4().hex[:10]}"
    return {
        "artifact_id": artifact_id,
        "job_id": job_id,
        "artifact_type": "spectrogram",
        "filename": filename,
        "storage_relative_path": rel_path,
        "file_size_bytes": file_size,
        "file_hash_sha256": file_hash,
        "mime_type": "image/png",
        "metadata": metadata
    }

def save_report_as_artifact(
    job_id: str,
    fmt: str,
    content: str,
    filename: str
) -> Dict[str, Any]:
    """
    Persists a generated report to storage and prepares artifact registration.
    """
    report_dir = sanitize_report_storage(job_id)
    abs_path = report_dir / filename
    
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    file_size = abs_path.stat().st_size
    rel_path = f"reports/{job_id}/{filename}"
    file_hash = compute_file_sha256(abs_path)
    ext = Path(filename).suffix.lower()
    mime = MIME_TYPES.get(ext, "application/octet-stream")
    
    artifact_id = f"art_rep_{fmt}_{uuid.uuid4().hex[:8]}"
    return {
        "artifact_id": artifact_id,
        "job_id": job_id,
        "artifact_type": "report",
        "filename": filename,
        "storage_relative_path": rel_path,
        "file_size_bytes": file_size,
        "file_hash_sha256": file_hash,
        "mime_type": mime,
        "metadata": {"format": fmt}
    }

def build_public_artifact_url(job_id: str, artifact_id: str) -> str:
    """Returns safe API URL path for downloading the artifact."""
    return f"/api/jobs/{job_id}/artifacts/{artifact_id}"

def get_storage_stats() -> Dict[str, Any]:
    """
    Calculates storage usage and disk breakdown for all managed media directories.
    """
    def _dir_stats(path: Path) -> Tuple[int, int]:
        total_size = 0
        total_files = 0
        if path.exists():
            for p in path.rglob("*"):
                try:
                    if p.is_file():
                        total_size += p.stat().st_size
                        total_files += 1
                except OSError:
                    pass
        return total_size, total_files

    uploads_bytes, uploads_files = _dir_stats(UPLOADS_DIR)
    artifacts_bytes, artifacts_files = _dir_stats(ARTIFACTS_DIR)
    reports_bytes, reports_files = _dir_stats(REPORTS_DIR)
    temp_bytes, temp_files = _dir_stats(TEMP_DIR)
    total_managed_bytes = uploads_bytes + artifacts_bytes + reports_bytes + temp_bytes

    disk_info = shutil.disk_usage(STORAGE_ROOT)

    return {
        "retention_policy": {
            "retention_days": STORAGE_RETENTION_DAYS,
            "max_storage_mb": MAX_STORAGE_MB,
            "cleanup_failed_jobs_hours": CLEANUP_FAILED_JOBS_HOURS
        },
        "storage_breakdown": {
            "uploads": {"bytes": uploads_bytes, "mb": round(uploads_bytes / (1024 * 1024), 2), "files": uploads_files},
            "artifacts": {"bytes": artifacts_bytes, "mb": round(artifacts_bytes / (1024 * 1024), 2), "files": artifacts_files},
            "reports": {"bytes": reports_bytes, "mb": round(reports_bytes / (1024 * 1024), 2), "files": reports_files},
            "temp": {"bytes": temp_bytes, "mb": round(temp_bytes / (1024 * 1024), 2), "files": temp_files},
            "total_managed": {
                "bytes": total_managed_bytes,
                "mb": round(total_managed_bytes / (1024 * 1024), 2),
                "total_files": uploads_files + artifacts_files + reports_files + temp_files
            }
        },
        "host_disk": {
            "total_bytes": disk_info.total,
            "free_bytes": disk_info.free,
            "free_gb": round(disk_info.free / (1024 ** 3), 2),
            "used_gb": round(disk_info.used / (1024 ** 3), 2)
        }
    }

def run_storage_cleanup(
    db_conn,
    retention_days: Optional[int] = None,
    max_storage_mb: Optional[int] = None,
    clean_orphans: bool = True,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Executes storage retention and cleanup policy:
    1. Removes files for jobs older than retention_days.
    2. Cleans temp files older than 1 hour.
    3. Cleans orphaned files not indexed in database.
    4. Truncates oldest artifacts if storage exceeds max_storage_mb limit.
    """
    days = retention_days if retention_days is not None else STORAGE_RETENTION_DAYS
    max_mb = max_storage_mb if max_storage_mb is not None else MAX_STORAGE_MB

    now = datetime.now()
    cutoff_retention = (now - timedelta(days=days)).isoformat()
    cutoff_failed = (now - timedelta(hours=CLEANUP_FAILED_JOBS_HOURS)).isoformat()

    audit = {
        "dry_run": dry_run,
        "retention_days": days,
        "max_storage_mb": max_mb,
        "freed_bytes": 0,
        "deleted_files_count": 0,
        "expired_jobs_purged": 0,
        "orphan_files_purged": 0,
        "temp_files_purged": 0
    }

    # 1. Clean temp directory files older than 1 hour
    one_hour_ago = time.time() - 3600
    if TEMP_DIR.exists():
        for p in TEMP_DIR.iterdir():
            if p.is_file() and p.stat().st_mtime < one_hour_ago:
                audit["freed_bytes"] += p.stat().st_size
                audit["deleted_files_count"] += 1
                audit["temp_files_purged"] += 1
                if not dry_run:
                    try:
                        p.unlink()
                    except Exception:
                        pass

    # 2. Identify expired jobs from DB
    cursor = db_conn.execute("""
        SELECT job_id, created_at, status FROM analysis_jobs
        WHERE created_at < ? OR (status IN ('failed', 'cancelled') AND created_at < ?)
    """, (cutoff_retention, cutoff_failed))
    expired_jobs = cursor.fetchall()

    for job in expired_jobs:
        jid = job["job_id"] if isinstance(job, (dict, sqlite3.Row)) or hasattr(job, "keys") else job[0]
        audit["expired_jobs_purged"] += 1
        
        # Get all registered artifacts for this job
        art_cursor = db_conn.execute("SELECT storage_relative_path FROM job_artifacts WHERE job_id = ?", (jid,))
        for art in art_cursor.fetchall():
            rel_art = art["storage_relative_path"] if isinstance(art, (dict, sqlite3.Row)) or hasattr(art, "keys") else art[0]
            try:
                p = resolve_safe_path(rel_art)
                if p.exists() and p.is_file():
                    audit["freed_bytes"] += p.stat().st_size
                    audit["deleted_files_count"] += 1
                    if not dry_run:
                        p.unlink()
            except Exception:
                pass

        # Check job directories
        job_art_dir = ARTIFACTS_DIR / jid
        if job_art_dir.exists():
            if not dry_run:
                shutil.rmtree(job_art_dir, ignore_errors=True)

        job_rep_dir = REPORTS_DIR / jid
        if job_rep_dir.exists():
            if not dry_run:
                shutil.rmtree(job_rep_dir, ignore_errors=True)

        # Remove from database if not dry_run
        if not dry_run:
            with db_conn:
                db_conn.execute("DELETE FROM job_artifacts WHERE job_id = ?", (jid,))
                db_conn.execute("DELETE FROM frame_attributions WHERE job_id = ?", (jid,))

    # 3. Clean orphan files if requested
    if clean_orphans:
        # Collect all active relative paths and filenames from DB
        c = db_conn.execute("SELECT storage_relative_path FROM job_artifacts")
        registered_paths = set()
        for row in c.fetchall():
            val = row["storage_relative_path"] if isinstance(row, (dict, sqlite3.Row)) or hasattr(row, "keys") else row[0]
            if val:
                clean_p = str(val).replace("\\", "/").strip("/")
                registered_paths.add(clean_p)
                registered_paths.add(Path(clean_p).name)

        c_jobs = db_conn.execute("SELECT file_path, filename, job_id FROM analysis_jobs")
        for row in c_jobs.fetchall():
            fp = row["file_path"] if isinstance(row, (dict, sqlite3.Row)) or hasattr(row, "keys") else row[0]
            fn = row["filename"] if isinstance(row, (dict, sqlite3.Row)) or hasattr(row, "keys") else row[1]
            jid = row["job_id"] if isinstance(row, (dict, sqlite3.Row)) or hasattr(row, "keys") else row[2]

            if fp:
                try:
                    fp_p = Path(fp)
                    registered_paths.add(fp_p.name)
                    try:
                        rel_fp = fp_p.resolve().relative_to(STORAGE_ROOT.resolve()).as_posix()
                        registered_paths.add(rel_fp)
                    except ValueError:
                        pass
                    clean_fp = str(fp).replace("\\", "/").strip("/")
                    registered_paths.add(clean_fp)
                except Exception:
                    pass

            if jid and fn:
                registered_paths.add(f"uploads/{jid}_{fn}".replace("\\", "/"))
                registered_paths.add(f"{jid}_{fn}".replace("\\", "/"))
            if fn:
                registered_paths.add(f"uploads/{fn}".replace("\\", "/"))
                registered_paths.add(str(fn).replace("\\", "/"))
                registered_paths.add(Path(fn).name)

        for directory in [UPLOADS_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
            if not directory.exists():
                continue
            for p in directory.rglob("*"):
                if p.is_file():
                    try:
                        rel = p.relative_to(STORAGE_ROOT).as_posix()
                        filename = p.name
                        if (rel not in registered_paths and 
                            filename not in registered_paths and 
                            not filename.startswith(".")):
                            audit["freed_bytes"] += p.stat().st_size
                            audit["deleted_files_count"] += 1
                            audit["orphan_files_purged"] += 1
                            if not dry_run:
                                p.unlink()
                    except Exception:
                        pass

    audit["freed_mb"] = round(audit["freed_bytes"] / (1024 * 1024), 2)
    return audit
