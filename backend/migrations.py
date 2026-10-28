import sqlite3
import hashlib
from datetime import datetime
from typing import List, Dict, Callable, Any, Optional, Set

"""
Database Migration Engine for Video Analysis System.
Provides versioned, sequential, non-destructive migrations.
"""

def _compute_checksum(script: str) -> str:
    return hashlib.sha256(script.strip().encode("utf-8")).hexdigest()[:16]

# Define migrations in strictly ascending version order
MIGRATIONS: List[Dict[str, Any]] = [
    {
        "version": 1,
        "name": "001_baseline_tables",
        "description": "Create baseline analysis_jobs, emergency_dispatches, and system_logs tables",
        "up": """
        CREATE TABLE IF NOT EXISTS analysis_jobs (
            job_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            file_hash_sha256 TEXT,
            status TEXT NOT NULL,
            progress REAL DEFAULT 0.0,
            current_step TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            prediction TEXT,
            confidence REAL,
            prob_fire REAL,
            prob_accident REAL,
            prob_normal REAL,
            evidence_json TEXT,
            error_message TEXT
        );

        CREATE TABLE IF NOT EXISTS emergency_dispatches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            dispatch_type TEXT NOT NULL,
            event_type TEXT NOT NULL,
            recipient TEXT NOT NULL,
            status TEXT NOT NULL,
            external_id TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (job_id) REFERENCES analysis_jobs(job_id)
        );

        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            details_json TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_status ON analysis_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON analysis_jobs(created_at);
        CREATE INDEX IF NOT EXISTS idx_jobs_prediction ON analysis_jobs(prediction);
        CREATE INDEX IF NOT EXISTS idx_dispatches_job_id ON emergency_dispatches(job_id);
        """
    },
    {
        "version": 2,
        "name": "002_add_model_and_modality_columns",
        "description": "Add model identifier, checksum, modality availability, transcript, and notes to analysis_jobs",
        "up": """
        -- SQLite requires adding columns one by one; existing data is preserved
        ALTER TABLE analysis_jobs ADD COLUMN model_identifier TEXT DEFAULT 'kannada_av_scene_model.keras';
        ALTER TABLE analysis_jobs ADD COLUMN model_checksum TEXT;
        ALTER TABLE analysis_jobs ADD COLUMN video_modality_available INTEGER DEFAULT 1;
        ALTER TABLE analysis_jobs ADD COLUMN audio_modality_available INTEGER DEFAULT 0;
        ALTER TABLE analysis_jobs ADD COLUMN speech_modality_available INTEGER DEFAULT 0;
        ALTER TABLE analysis_jobs ADD COLUMN transcript TEXT;
        ALTER TABLE analysis_jobs ADD COLUMN description TEXT;
        ALTER TABLE analysis_jobs ADD COLUMN notification_status TEXT DEFAULT 'none';
        ALTER TABLE analysis_jobs ADD COLUMN report_reference TEXT;
        ALTER TABLE analysis_jobs ADD COLUMN updated_at TEXT;
        """
    },
    {
        "version": 3,
        "name": "003_create_job_artifacts_table",
        "description": "Create job_artifacts table for managed file storage (frames, spectrograms, reports)",
        "up": """
        CREATE TABLE IF NOT EXISTS job_artifacts (
            artifact_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            artifact_type TEXT NOT NULL, -- video, frame, spectrogram, report, audio
            filename TEXT NOT NULL,
            storage_relative_path TEXT NOT NULL, -- relative to storage root, NEVER exposing local host absolute path
            file_size_bytes INTEGER NOT NULL,
            file_hash_sha256 TEXT,
            mime_type TEXT NOT NULL,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (job_id) REFERENCES analysis_jobs(job_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_artifacts_job_id ON job_artifacts(job_id);
        CREATE INDEX IF NOT EXISTS idx_artifacts_type ON job_artifacts(artifact_type);
        """
    },
    {
        "version": 4,
        "name": "004_create_frame_attributions_table",
        "description": "Create frame_attributions table for frame-level attention scores and keyframe highlights",
        "up": """
        CREATE TABLE IF NOT EXISTS frame_attributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            frame_index INTEGER NOT NULL,
            timestamp_sec REAL NOT NULL,
            attribution_score REAL NOT NULL,
            is_keyframe INTEGER DEFAULT 0,
            artifact_id TEXT,
            label TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (job_id) REFERENCES analysis_jobs(job_id) ON DELETE CASCADE,
            FOREIGN KEY (artifact_id) REFERENCES job_artifacts(artifact_id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_frame_attributions_job_id ON frame_attributions(job_id);
        CREATE INDEX IF NOT EXISTS idx_frame_attributions_score ON frame_attributions(attribution_score);
        """
    },
    {
        "version": 5,
        "name": "005_performance_and_audit_indexes",
        "description": "Add composite and deduplication indexes for fast searchable history and integrity checks",
        "up": """
        CREATE INDEX IF NOT EXISTS idx_jobs_file_hash ON analysis_jobs(file_hash_sha256);
        CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON analysis_jobs(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_dispatches_status ON emergency_dispatches(status);
        """
    },
    {
        "version": 6,
        "name": "006_evidence_based_frame_attributions",
        "description": "Add columns for baseline score, perturbed score, score diff, rank, and attribution method",
        "up": """
        ALTER TABLE frame_attributions ADD COLUMN baseline_score REAL;
        ALTER TABLE frame_attributions ADD COLUMN perturbed_score REAL;
        ALTER TABLE frame_attributions ADD COLUMN score_difference REAL;
        ALTER TABLE frame_attributions ADD COLUMN attribution_method TEXT;
        ALTER TABLE frame_attributions ADD COLUMN rank INTEGER;
        """
    },
    {
        "version": 7,
        "name": "007_speech_transcripts_and_scene_descriptions",
        "description": "Create speech_transcripts table with chunks/timestamps and add structured_description column",
        "up": """
        CREATE TABLE IF NOT EXISTS speech_transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            language TEXT NOT NULL,
            transcript TEXT NOT NULL,
            confidence REAL,
            is_accepted INTEGER DEFAULT 1,
            chunks_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES analysis_jobs(job_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_speech_transcripts_job ON speech_transcripts(job_id);
        ALTER TABLE analysis_jobs ADD COLUMN structured_description TEXT;
        """
    },
    {
        "version": 8,
        "name": "008_notification_rules_and_dispatch_errors",
        "description": "Add error_message column and audit indexes to emergency_dispatches table",
        "up": """
        ALTER TABLE emergency_dispatches ADD COLUMN error_message TEXT;
        CREATE INDEX IF NOT EXISTS idx_dispatches_type_job ON emergency_dispatches(dispatch_type, job_id);
        CREATE INDEX IF NOT EXISTS idx_dispatches_event_recip ON emergency_dispatches(event_type, recipient, created_at);
        """
    }
]

def init_migration_table(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        checksum TEXT NOT NULL,
        applied_at TEXT NOT NULL
    );
    """)

def get_applied_versions(conn: sqlite3.Connection) -> List[int]:
    init_migration_table(conn)
    cursor = conn.execute("SELECT version FROM schema_migrations ORDER BY version ASC;")
    return [row[0] for row in cursor.fetchall()]

def get_existing_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    cursor = conn.execute(f"PRAGMA table_info({table_name});")
    return [row[1] for row in cursor.fetchall()]

def run_migrations(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Applies all pending migrations in atomic transactions.
    Safely handles column addition checks to prevent duplicate column errors on existing DBs.
    """
    init_migration_table(conn)
    applied_versions = set(get_applied_versions(conn))
    applied_this_run = []

    for migration in MIGRATIONS:
        v = migration["version"]
        if v in applied_versions:
            continue

        name = migration["name"]
        up_script = migration["up"]
        checksum = _compute_checksum(up_script)

        with conn:
            # Special non-destructive handling for ALTER TABLE statements in migrations 2 and 6
            if v == 2:
                existing_cols = set(get_existing_table_columns(conn, "analysis_jobs"))
                for line in up_script.strip().splitlines():
                    line = line.strip()
                    if line.startswith("ALTER TABLE analysis_jobs ADD COLUMN"):
                        parts = line.split()
                        col_name = parts[5]
                        if col_name not in existing_cols:
                            conn.execute(line)
            elif v == 6:
                existing_cols = set(get_existing_table_columns(conn, "frame_attributions"))
                for line in up_script.strip().splitlines():
                    line = line.strip()
                    if line.startswith("ALTER TABLE frame_attributions ADD COLUMN"):
                        parts = line.split()
                        col_name = parts[5]
                        if col_name not in existing_cols:
                            conn.execute(line)
            elif v == 7:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS speech_transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    language TEXT NOT NULL,
                    transcript TEXT NOT NULL,
                    confidence REAL,
                    is_accepted INTEGER DEFAULT 1,
                    chunks_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES analysis_jobs(job_id) ON DELETE CASCADE
                );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_speech_transcripts_job ON speech_transcripts(job_id);")
                existing_cols = set(get_existing_table_columns(conn, "analysis_jobs"))
                if "structured_description" not in existing_cols:
                    conn.execute("ALTER TABLE analysis_jobs ADD COLUMN structured_description TEXT;")
            elif v == 8:
                existing_cols = set(get_existing_table_columns(conn, "emergency_dispatches"))
                if "error_message" not in existing_cols:
                    conn.execute("ALTER TABLE emergency_dispatches ADD COLUMN error_message TEXT;")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_dispatches_type_job ON emergency_dispatches(dispatch_type, job_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_dispatches_event_recip ON emergency_dispatches(event_type, recipient, created_at);")
            else:
                conn.executescript(up_script)

            now_iso = datetime.now().isoformat()
            conn.execute("""
                INSERT INTO schema_migrations (version, name, description, checksum, applied_at)
                VALUES (?, ?, ?, ?, ?)
            """, (v, name, migration.get("description", ""), checksum, now_iso))

        applied_this_run.append({
            "version": v,
            "name": name,
            "applied_at": now_iso
        })

    return applied_this_run
