import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from migrations import run_migrations

BACKEND_DIR = Path(__file__).resolve().parent
DB_PATH = BACKEND_DIR / "av_scene_data.db"

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db():
    """Initializes and runs versioned migrations on the database."""
    conn = get_connection()
    try:
        run_migrations(conn)
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Job Operations
# ---------------------------------------------------------------------------

def create_job(
    job_id: str,
    filename: str,
    file_path: str,
    file_size_bytes: int,
    file_hash_sha256: Optional[str] = None,
    description: Optional[str] = None,
    model_identifier: str = "kannada_av_scene_model.keras"
) -> Dict[str, Any]:
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        with conn:
            conn.execute("""
                INSERT INTO analysis_jobs (
                    job_id, filename, file_path, file_size_bytes, file_hash_sha256,
                    status, progress, current_step, created_at, updated_at,
                    description, model_identifier, notification_status
                ) VALUES (?, ?, ?, ?, ?, 'queued', 0.0, 'Job queued for processing', ?, ?, ?, ?, 'none')
            """, (
                job_id, filename, file_path, file_size_bytes, file_hash_sha256,
                now, now, description, model_identifier
            ))
    finally:
        conn.close()
    return get_job(job_id)

def update_job_progress(
    job_id: str,
    status: str,
    progress: float,
    current_step: str,
    started_at: Optional[str] = None
):
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        with conn:
            if started_at:
                conn.execute("""
                    UPDATE analysis_jobs
                    SET status = ?, progress = ?, current_step = ?, started_at = ?, updated_at = ?
                    WHERE job_id = ?
                """, (status, progress, current_step, started_at, now, job_id))
            else:
                conn.execute("""
                    UPDATE analysis_jobs
                    SET status = ?, progress = ?, current_step = ?, updated_at = ?
                    WHERE job_id = ?
                """, (status, progress, current_step, now, job_id))
    finally:
        conn.close()

def complete_job(
    job_id: str,
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float],
    evidence: Dict[str, Any],
    model_identifier: Optional[str] = "kannada_av_scene_model.keras",
    model_checksum: Optional[str] = None,
    video_modality: bool = True,
    audio_modality: bool = False,
    speech_modality: bool = False,
    transcript: Optional[str] = None,
    description: Optional[str] = None,
    notification_status: Optional[str] = "none",
    report_reference: Optional[str] = None,
    structured_description: Optional[Dict[str, Any]] = None
):
    now = datetime.now().isoformat()
    evidence_str = json.dumps(evidence, ensure_ascii=False) if evidence is not None else None
    struct_desc_str = json.dumps(structured_description, ensure_ascii=False) if structured_description is not None else None
    conn = get_connection()
    try:
        with conn:
            conn.execute("""
                UPDATE analysis_jobs
                SET status = 'completed',
                    progress = 1.0,
                    current_step = 'Analysis completed successfully',
                    completed_at = ?,
                    updated_at = ?,
                    prediction = ?,
                    confidence = ?,
                    prob_fire = ?,
                    prob_accident = ?,
                    prob_normal = ?,
                    evidence_json = ?,
                    model_identifier = COALESCE(?, model_identifier),
                    model_checksum = ?,
                    video_modality_available = ?,
                    audio_modality_available = ?,
                    speech_modality_available = ?,
                    transcript = ?,
                    description = COALESCE(?, description),
                    structured_description = COALESCE(?, structured_description),
                    notification_status = COALESCE(?, notification_status),
                    report_reference = ?
                WHERE job_id = ?
            """, (
                now,
                now,
                prediction,
                confidence,
                probabilities.get("Fire", 0.0),
                probabilities.get("Accident", 0.0),
                probabilities.get("Normal", 0.0),
                evidence_str,
                model_identifier,
                model_checksum,
                1 if video_modality else 0,
                1 if audio_modality else 0,
                1 if speech_modality else 0,
                transcript,
                description,
                struct_desc_str,
                notification_status,
                report_reference,
                job_id
            ))
    finally:
        conn.close()

def fail_job(job_id: str, error_message: str):
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        with conn:
            conn.execute("""
                UPDATE analysis_jobs
                SET status = 'failed',
                    current_step = 'Analysis failed',
                    completed_at = ?,
                    updated_at = ?,
                    error_message = ?
                WHERE job_id = ?
            """, (now, now, error_message, job_id))
    finally:
        conn.close()

def cancel_job_record(job_id: str) -> bool:
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        with conn:
            cursor = conn.execute("""
                UPDATE analysis_jobs
                SET status = 'cancelled',
                    current_step = 'Job cancelled by user request',
                    completed_at = ?,
                    updated_at = ?
                WHERE job_id = ? AND status IN ('queued', 'processing')
            """, (now, now, job_id))
            return cursor.rowcount > 0
    finally:
        conn.close()

def update_job_description(job_id: str, description: str) -> bool:
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        with conn:
            cursor = conn.execute("""
                UPDATE analysis_jobs
                SET description = ?, updated_at = ?
                WHERE job_id = ?
            """, (description, now, job_id))
            return cursor.rowcount > 0
    finally:
        conn.close()

def update_job_notification_status(job_id: str, notification_status: str) -> bool:
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        with conn:
            cursor = conn.execute("""
                UPDATE analysis_jobs
                SET notification_status = ?, updated_at = ?
                WHERE job_id = ?
            """, (notification_status, now, job_id))
            return cursor.rowcount > 0
    finally:
        conn.close()

def find_job_by_file_hash(file_hash_sha256: str, status: str = "completed") -> Optional[Dict[str, Any]]:
    if not file_hash_sha256:
        return None
    conn = get_connection()
    try:
        cursor = conn.execute("""
            SELECT * FROM analysis_jobs 
            WHERE file_hash_sha256 = ? AND status = ?
            ORDER BY completed_at DESC
            LIMIT 1
        """, (file_hash_sha256, status))
        row = cursor.fetchone()
        if not row:
            return None
        return _format_job_row(row)
    finally:
        conn.close()

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT * FROM analysis_jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return _format_job_row(row)
    finally:
        conn.close()

def list_jobs(
    query: Optional[str] = None,
    status: Optional[str] = None,
    prediction: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    sort_desc: bool = True
) -> Tuple[List[Dict[str, Any]], int, Dict[str, Any]]:
    conn = get_connection()
    try:
        conditions = []
        params = []

        if query:
            conditions.append("(filename LIKE ? OR job_id LIKE ? OR description LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])

        if status:
            conditions.append("status = ?")
            params.append(status)

        if prediction:
            conditions.append("prediction = ?")
            params.append(prediction)

        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        count_cursor = conn.execute(f"SELECT COUNT(*) as total FROM analysis_jobs {where_clause}", params)
        total = count_cursor.fetchone()["total"]

        order = "DESC" if sort_desc else "ASC"
        query_sql = f"""
            SELECT * FROM analysis_jobs
            {where_clause}
            ORDER BY created_at {order}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        cursor = conn.execute(query_sql, params)
        rows = cursor.fetchall()

        current_page = (offset // limit) + 1 if limit > 0 else 1
        total_pages = (total + limit - 1) // limit if limit > 0 else 1
        pagination = {
            "total": total,
            "limit": limit,
            "offset": offset,
            "page": current_page,
            "total_pages": total_pages,
            "has_next": (offset + limit) < total,
            "has_prev": offset > 0
        }

        return [_format_job_row(r) for r in rows], total, pagination
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Artifacts Operations
# ---------------------------------------------------------------------------

def add_artifact(
    artifact_id: str,
    job_id: str,
    artifact_type: str,
    filename: str,
    storage_relative_path: str,
    file_size_bytes: int,
    mime_type: str,
    file_hash_sha256: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    now = datetime.now().isoformat()
    meta_json = json.dumps(metadata) if metadata else None
    conn = get_connection()
    try:
        with conn:
            conn.execute("""
                INSERT INTO job_artifacts (
                    artifact_id, job_id, artifact_type, filename,
                    storage_relative_path, file_size_bytes, file_hash_sha256,
                    mime_type, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                artifact_id, job_id, artifact_type, filename,
                storage_relative_path, file_size_bytes, file_hash_sha256,
                mime_type, meta_json, now
            ))
            return artifact_id
    finally:
        conn.close()

def get_job_artifacts(job_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            SELECT * FROM job_artifacts 
            WHERE job_id = ? 
            ORDER BY created_at ASC
        """, (job_id,))
        return [_format_artifact_row(r) for r in cursor.fetchall()]
    finally:
        conn.close()

def get_artifact(artifact_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT * FROM job_artifacts WHERE artifact_id = ?", (artifact_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return _format_artifact_row(row)
    finally:
        conn.close()

def delete_artifact(artifact_id: str) -> bool:
    conn = get_connection()
    try:
        with conn:
            cursor = conn.execute("DELETE FROM job_artifacts WHERE artifact_id = ?", (artifact_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Frame Attributions Operations
# ---------------------------------------------------------------------------

def add_frame_attribution(
    job_id: str,
    frame_index: int,
    timestamp_sec: float,
    attribution_score: float,
    baseline_score: Optional[float] = None,
    perturbed_score: Optional[float] = None,
    score_difference: Optional[float] = None,
    attribution_method: str = "temporal_frame_ablation",
    rank: Optional[int] = None,
    is_keyframe: bool = False,
    artifact_id: Optional[str] = None,
    label: Optional[str] = None
) -> int:
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        with conn:
            cursor = conn.execute("""
                INSERT INTO frame_attributions (
                    job_id, frame_index, timestamp_sec, attribution_score,
                    baseline_score, perturbed_score, score_difference,
                    attribution_method, rank, is_keyframe, artifact_id, label, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id, frame_index, timestamp_sec, attribution_score,
                baseline_score, perturbed_score, score_difference,
                attribution_method, rank, 1 if is_keyframe else 0, artifact_id, label, now
            ))
            return cursor.lastrowid
    finally:
        conn.close()

def get_frame_attributions(job_id: str, order_by_rank: bool = False) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        order_clause = "rank ASC, frame_index ASC" if order_by_rank else "frame_index ASC"
        cursor = conn.execute(f"""
            SELECT * FROM frame_attributions 
            WHERE job_id = ? 
            ORDER BY {order_clause}
        """, (job_id,))
        results = []
        for r in cursor.fetchall():
            d = dict(r)
            d["is_keyframe"] = bool(d["is_keyframe"])
            d["attribution_url"] = f"/api/jobs/{job_id}/artifacts/{d['artifact_id']}" if d.get("artifact_id") else None
            results.append(d)
        return results
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Speech Transcripts Operations
# ---------------------------------------------------------------------------

def add_speech_transcript(
    job_id: str,
    language: str,
    transcript: str,
    confidence: float = 1.0,
    is_accepted: bool = True,
    chunks: Optional[List[Dict[str, Any]]] = None
) -> int:
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        with conn:
            cursor = conn.execute("""
                INSERT INTO speech_transcripts (
                    job_id, language, transcript, confidence, is_accepted, chunks_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                language,
                transcript,
                confidence,
                1 if is_accepted else 0,
                json.dumps(chunks, ensure_ascii=False) if chunks is not None else None,
                now
            ))
            return cursor.lastrowid
    finally:
        conn.close()

def get_speech_transcripts(job_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            SELECT * FROM speech_transcripts 
            WHERE job_id = ? 
            ORDER BY id ASC
        """, (job_id,))
        results = []
        for r in cursor.fetchall():
            d = dict(r)
            d["is_accepted"] = bool(d["is_accepted"])
            if d.get("chunks_json"):
                try:
                    d["chunks"] = json.loads(d["chunks_json"])
                except Exception:
                    d["chunks"] = []
            else:
                d["chunks"] = []
            results.append(d)
        return results
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Emergency Dispatches Operations
# ---------------------------------------------------------------------------

def record_dispatch(
    job_id: Optional[str],
    dispatch_type: str,
    event_type: str,
    recipient: str,
    status: str,
    external_id: Optional[str] = None,
    error_message: Optional[str] = None
):
    now = datetime.now().isoformat()
    conn = get_connection()
    try:
        with conn:
            # Check if job_id exists in analysis_jobs
            effective_job_id = job_id
            job_exists = False
            if job_id:
                job_row = conn.execute("SELECT 1 FROM analysis_jobs WHERE job_id = ?", (job_id,)).fetchone()
                if job_row:
                    job_exists = True
                else:
                    effective_job_id = None

            # Check if error_message column is in table
            cols = [r[1] for r in conn.execute("PRAGMA table_info(emergency_dispatches);").fetchall()]
            if "error_message" in cols:
                conn.execute("""
                    INSERT INTO emergency_dispatches (
                        job_id, dispatch_type, event_type, recipient, status, external_id, error_message, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (effective_job_id, dispatch_type, event_type, recipient, status, external_id, error_message, now))
            else:
                conn.execute("""
                    INSERT INTO emergency_dispatches (
                        job_id, dispatch_type, event_type, recipient, status, external_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (effective_job_id, dispatch_type, event_type, recipient, status, external_id, now))
            
            # Also update parent job's notification status if linked and existing
            if job_exists and job_id:
                if status == "success" or status == "sent":
                    notif_status = "dispatched"
                elif status.startswith("skipped"):
                    notif_status = status
                else:
                    notif_status = "failed"

                conn.execute("""
                    UPDATE analysis_jobs 
                    SET notification_status = ?, updated_at = ?
                    WHERE job_id = ?
                """, (notif_status, now, job_id))
    finally:
        conn.close()

def get_job_dispatches(job_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT * FROM emergency_dispatches WHERE job_id = ? ORDER BY created_at DESC", (job_id,))
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()

def has_recent_dispatch(
    event_type: str,
    recipient: Optional[str] = None,
    job_id: Optional[str] = None,
    window_minutes: int = 60
) -> bool:
    """
    Checks if a successful dispatch was already sent for the same job_id
    or the same (event_type, recipient) within the duplicate cooldown window.
    """
    conn = get_connection()
    try:
        threshold_time = (datetime.now() - timedelta(minutes=window_minutes)).isoformat()
        
        # 1. Job-level duplicate check
        if job_id:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM emergency_dispatches
                WHERE job_id = ? AND status IN ('success', 'sent', 'dispatched')
            """, (job_id,))
            if cursor.fetchone()[0] > 0:
                return True

        # 2. Recipient & Event level cooldown check
        if recipient:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM emergency_dispatches
                WHERE event_type = ? AND recipient = ? AND status IN ('success', 'sent', 'dispatched') AND created_at >= ?
            """, (event_type, recipient, threshold_time))
            if cursor.fetchone()[0] > 0:
                return True

        return False
    finally:
        conn.close()

def get_recent_dispatches(limit: int = 50) -> List[Dict[str, Any]]:
    """Returns recent emergency dispatches across the entire system."""
    conn = get_connection()
    try:
        cursor = conn.execute("""
            SELECT * FROM emergency_dispatches 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# System Metrics & Stats
# ---------------------------------------------------------------------------

def get_stats() -> Dict[str, Any]:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_jobs,
                SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) as queued_jobs,
                SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing_jobs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_jobs,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_jobs,
                SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled_jobs,
                SUM(CASE WHEN prediction = 'Fire' THEN 1 ELSE 0 END) as fire_predictions,
                SUM(CASE WHEN prediction = 'Accident' THEN 1 ELSE 0 END) as accident_predictions,
                SUM(CASE WHEN prediction = 'Normal' THEN 1 ELSE 0 END) as normal_predictions,
                SUM(file_size_bytes) as total_bytes_processed
            FROM analysis_jobs
        """)
        row = cursor.fetchone()
        
        dispatch_cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_dispatches,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_dispatches,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_dispatches
            FROM emergency_dispatches
        """)
        d_row = dispatch_cursor.fetchone()

        artifacts_cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_artifacts,
                SUM(file_size_bytes) as total_artifacts_bytes
            FROM job_artifacts
        """)
        a_row = artifacts_cursor.fetchone()

        db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0

        return {
            "total_jobs": row["total_jobs"] or 0,
            "jobs_by_status": {
                "queued": row["queued_jobs"] or 0,
                "processing": row["processing_jobs"] or 0,
                "completed": row["completed_jobs"] or 0,
                "failed": row["failed_jobs"] or 0,
                "cancelled": row["cancelled_jobs"] or 0
            },
            "predictions_by_class": {
                "Fire": row["fire_predictions"] or 0,
                "Accident": row["accident_predictions"] or 0,
                "Normal": row["normal_predictions"] or 0
            },
            "dispatches": {
                "total": d_row["total_dispatches"] or 0,
                "successful": d_row["successful_dispatches"] or 0,
                "failed": d_row["failed_dispatches"] or 0
            },
            "artifacts": {
                "total_count": a_row["total_artifacts"] or 0,
                "total_bytes": a_row["total_artifacts_bytes"] or 0
            },
            "total_bytes_processed": row["total_bytes_processed"] or 0,
            "database_size_bytes": db_size
        }
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Formatters (Sanitizes absolute server filesystem paths from public response)
# ---------------------------------------------------------------------------

def _format_job_row(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    
    # SECURITY: Strip internal host filesystem paths from public output
    d.pop("file_path", None)
    
    evidence = {}
    if d.get("evidence_json"):
        try:
            evidence = json.loads(d["evidence_json"])
        except Exception:
            evidence = {}
    d["evidence"] = evidence
    
    probabilities = None
    if d["status"] == "completed" and d["prediction"] is not None:
        probabilities = {
            "Fire": d.get("prob_fire") or 0.0,
            "Accident": d.get("prob_accident") or 0.0,
            "Normal": d.get("prob_normal") or 0.0
        }
    d["probabilities"] = probabilities

    # Convert modality flags to boolean dictionary
    d["modalities"] = {
        "video": bool(d.get("video_modality_available", 1)),
        "audio": bool(d.get("audio_modality_available", 0)),
        "speech": bool(d.get("speech_modality_available", 0))
    }

    # Parse structured_description if present
    if d.get("structured_description"):
        try:
            d["structured_description"] = json.loads(d["structured_description"])
        except Exception:
            pass

    # Safe public URLs
    jid = d["job_id"]
    d["artifacts_url"] = f"/api/jobs/{jid}/artifacts"
    d["attributions_url"] = f"/api/jobs/{jid}/attributions"
    d["transcripts_url"] = f"/api/jobs/{jid}/transcripts"
    d["report_download_url"] = f"/api/jobs/{jid}/report"

    return d

def _format_artifact_row(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    # SECURITY: Strip internal storage_relative_path from public serialization
    storage_path = d.pop("storage_relative_path", "")
    metadata = {}
    if d.get("metadata_json"):
        try:
            metadata = json.loads(d["metadata_json"])
        except Exception:
            metadata = {}
    d["metadata"] = metadata
    d.pop("metadata_json", None)

    # Safe public download URL
    jid = d["job_id"]
    aid = d["artifact_id"]
    d["download_url"] = f"/api/jobs/{jid}/artifacts/{aid}"
    # Keep internal storage path accessible only by explicitly inspecting row
    d["_storage_path"] = storage_path
    return d

# Auto-initialize database tables and run migrations on import
init_db()
