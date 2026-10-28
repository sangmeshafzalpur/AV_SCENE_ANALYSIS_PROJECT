import os
import sys
import unittest
import tempfile
import shutil
import sqlite3
import hashlib
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import cv2
import requests

import database
from migrations import run_migrations, get_applied_versions, MIGRATIONS
from services.storage_manager import (
    STORAGE_ROOT,
    UPLOADS_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    TEMP_DIR,
    resolve_safe_path,
    save_frame_as_artifact,
    save_spectrogram_as_artifact,
    save_report_as_artifact,
    run_storage_cleanup,
    get_storage_stats
)
from services.job_engine import submit_analysis_job, validate_video_file

class TestDatabasePersistenceAndArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:8000"
        try:
            r = requests.get(f"{cls.base_url}/api/health", timeout=1)
            cls.server_online = (r.status_code == 200)
        except Exception:
            cls.server_online = False

    def test_01_versioned_migrations_and_idempotency(self):
        """Verify migrations execute in atomic version order and are strictly idempotent."""
        test_conn = sqlite3.connect(":memory:")
        test_conn.row_factory = sqlite3.Row

        # Initial run
        applied = run_migrations(test_conn)
        self.assertEqual([m["version"] for m in applied], list(range(1, len(MIGRATIONS) + 1)))

        # Second run (idempotency check)
        applied_second = run_migrations(test_conn)
        self.assertEqual(len(applied_second), 0, "Second migration run should apply 0 migrations")

        # Verify columns in analysis_jobs
        cols = [r[1] for r in test_conn.execute("PRAGMA table_info(analysis_jobs)").fetchall()]
        required_cols = [
            "job_id", "filename", "status", "prediction", "confidence",
            "model_identifier", "model_checksum", "video_modality_available",
            "audio_modality_available", "speech_modality_available",
            "transcript", "description", "notification_status", "report_reference"
        ]
        for rc in required_cols:
            self.assertIn(rc, cols, f"Column '{rc}' must exist in analysis_jobs")
        test_conn.close()

    def test_02_persistence_across_connection_restarts(self):
        """Verify records, artifacts, and attributions survive complete DB connection close & restart."""
        temp_dir = tempfile.mkdtemp()
        temp_db_path = Path(temp_dir) / "restart_test.db"
        try:
            # 1. Open connection, migrate, insert records
            conn1 = sqlite3.connect(str(temp_db_path))
            conn1.row_factory = sqlite3.Row
            run_migrations(conn1)

            # Insert job
            test_job_id = "job_restart_test_123"
            test_checksum = "2ec2af07c9ef1d061575e8a48f0fadd741abd0f6f35c10b7399aa9815ed4cb42"
            now = "2026-10-27T12:00:00"
            with conn1:
                conn1.execute("""
                    INSERT INTO analysis_jobs (
                        job_id, filename, file_path, file_size_bytes, file_hash_sha256,
                        status, progress, created_at, model_identifier, model_checksum,
                        video_modality_available, audio_modality_available, speech_modality_available,
                        transcript, description, notification_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    test_job_id, "traffic.mp4", "uploads/traffic.mp4", 500000, "hash123",
                    "completed", 1.0, now, "kannada_av_scene_model.keras", test_checksum,
                    1, 1, 1, "ನೋಡಿ ಇಲ್ಲಿ", "Accident near junction", "dispatched"
                ))

                conn1.execute("""
                    INSERT INTO job_artifacts (
                        artifact_id, job_id, artifact_type, filename, storage_relative_path,
                        file_size_bytes, mime_type, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, ("art_01", test_job_id, "spectrogram", "spec.png", "artifacts/spec.png", 1024, "image/png", now))

                conn1.execute("""
                    INSERT INTO frame_attributions (
                        job_id, frame_index, timestamp_sec, attribution_score, is_keyframe, label, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (test_job_id, 3, 1.25, 0.94, 1, "Keyframe Peak", now))

            # 2. Simulate complete restart by closing connection
            conn1.close()
            del conn1

            # 3. Re-open brand-new connection from scratch
            conn2 = sqlite3.connect(str(temp_db_path))
            conn2.row_factory = sqlite3.Row

            job_row = conn2.execute("SELECT * FROM analysis_jobs WHERE job_id = ?", (test_job_id,)).fetchone()
            self.assertIsNotNone(job_row)
            self.assertEqual(job_row["model_checksum"], test_checksum)
            self.assertEqual(job_row["transcript"], "ನೋಡಿ ಇಲ್ಲಿ")
            self.assertEqual(job_row["notification_status"], "dispatched")
            self.assertEqual(job_row["audio_modality_available"], 1)

            art_row = conn2.execute("SELECT * FROM job_artifacts WHERE job_id = ?", (test_job_id,)).fetchone()
            self.assertIsNotNone(art_row)
            self.assertEqual(art_row["artifact_type"], "spectrogram")

            attr_row = conn2.execute("SELECT * FROM frame_attributions WHERE job_id = ?", (test_job_id,)).fetchone()
            self.assertIsNotNone(attr_row)
            self.assertAlmostEqual(attr_row["attribution_score"], 0.94, places=2)
            self.assertEqual(attr_row["is_keyframe"], 1)

            conn2.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_03_transaction_safety_and_rollback_on_failure(self):
        """Verify that transactions are atomic and roll back cleanly if an error occurs midway."""
        test_conn = sqlite3.connect(":memory:")
        test_conn.row_factory = sqlite3.Row
        run_migrations(test_conn)

        job_id = "job_atomic_fail_test"
        try:
            with test_conn:
                test_conn.execute("""
                    INSERT INTO analysis_jobs (
                        job_id, filename, file_path, file_size_bytes, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (job_id, "corrupt.mp4", "uploads/corrupt.mp4", 100, "queued", "2026-10-27T12:00:00"))
                
                # Intentional error to trigger rollback
                raise RuntimeError("Simulated mid-pipeline pipeline crash!")
        except RuntimeError:
            pass

        # Verify job was rolled back and does not exist
        row = test_conn.execute("SELECT * FROM analysis_jobs WHERE job_id = ?", (job_id,)).fetchone()
        self.assertIsNone(row, "Uncommitted record must be rolled back on exception")
        test_conn.close()

    def test_04_duplicate_prevention_by_content_hash(self):
        """Verify content-hash deduplication detects identical videos and avoids redundant processing."""
        # Create a temporary dummy video file
        temp_dir = tempfile.mkdtemp()
        dummy_video = Path(temp_dir) / "dup_test.mp4"
        try:
            with open(dummy_video, "wb") as f:
                f.write(b"\x00\x00\x00 ftypisom" + b"\x00" * 4000)

            sha256 = hashlib.sha256()
            with open(dummy_video, "rb") as f:
                sha256.update(f.read())
            file_hash = sha256.hexdigest()

            # Insert an already-completed job with this hash
            job_existing_id = f"job_dup_origin_{uuid.uuid4().hex[:8]}"
            database.create_job(
                job_id=job_existing_id,
                filename="dup_test.mp4",
                file_path=str(dummy_video),
                file_size_bytes=dummy_video.stat().st_size,
                file_hash_sha256=file_hash
            )
            database.complete_job(
                job_id=job_existing_id,
                prediction="Fire",
                confidence=0.98,
                probabilities={"Fire": 0.98, "Accident": 0.01, "Normal": 0.01},
                evidence={},
                model_checksum="2ec2af07c9ef1d061575e8a48f0fadd741abd0f6f35c10b7399aa9815ed4cb42"
            )

            # Test duplicate lookup
            found = database.find_job_by_file_hash(file_hash, status="completed")
            self.assertIsNotNone(found)
            self.assertEqual(found["job_id"], job_existing_id)

            # Test submit_analysis_job with force_reanalysis=False
            job_record, was_duplicate = submit_analysis_job(
                filename="dup_test.mp4",
                temp_source_path=dummy_video,
                file_size_bytes=dummy_video.stat().st_size,
                models_dict={},
                force_reanalysis=False
            )
            self.assertTrue(was_duplicate)
            self.assertTrue(job_record.get("cached"))
            self.assertEqual(job_record["job_id"], job_existing_id)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_05_path_traversal_prevention_and_artifact_security(self):
        """Verify strict path traversal protection and secure artifact downloading."""
        # Test directory traversal attacks
        malicious_paths = [
            "../../../../Windows/System32/calc.exe",
            "..\\..\\boot.ini",
            "/etc/shadow",
            "artifacts/../../../secret.txt"
        ]
        for bad_path in malicious_paths:
            with self.assertRaises(PermissionError, msg=f"Path traversal '{bad_path}' should be blocked"):
                resolve_safe_path(bad_path)

        # Create parent job to satisfy foreign key constraint
        test_job_id = f"job_security_{uuid.uuid4().hex[:8]}"
        database.create_job(
            job_id=test_job_id,
            filename="security_test.mp4",
            file_path="uploads/security_test.mp4",
            file_size_bytes=2048,
            file_hash_sha256="dummyhash"
        )

        # Create a managed artifact and save it
        dummy_rgb = np.ones((224, 224, 3), dtype=np.uint8) * 128
        frame_art = save_frame_as_artifact(
            job_id=test_job_id,
            frame_rgb=dummy_rgb,
            frame_index=1,
            timestamp_sec=0.04,
            is_keyframe=True
        )
        self.assertIn("storage_relative_path", frame_art)
        self.assertFalse(frame_art["storage_relative_path"].startswith("C:"), "Must use relative storage path")

        # Register artifact
        art_id = database.add_artifact(
            artifact_id=frame_art["artifact_id"],
            job_id=test_job_id,
            artifact_type=frame_art["artifact_type"],
            filename=frame_art["filename"],
            storage_relative_path=frame_art["storage_relative_path"],
            file_size_bytes=frame_art["file_size_bytes"],
            mime_type=frame_art["mime_type"],
            file_hash_sha256=frame_art["file_hash_sha256"],
            metadata=frame_art["metadata"]
        )

        # Retrieve via database
        retrieved = database.get_artifact(art_id)
        self.assertIsNotNone(retrieved)
        self.assertNotIn("storage_relative_path", retrieved, "Relative storage path must not be exposed in public dict")
        self.assertIn("download_url", retrieved)
        self.assertEqual(retrieved["download_url"], f"/api/jobs/{test_job_id}/artifacts/{art_id}")

        # Live endpoint download
        if self.server_online:
            resp = requests.get(f"{self.base_url}/api/jobs/{test_job_id}/artifacts/{art_id}")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.headers["content-type"], "image/jpeg")
            self.assertGreater(len(resp.content), 0)

            # Test non-existent artifact -> 404
            resp404 = requests.get(f"{self.base_url}/api/jobs/{test_job_id}/artifacts/art_non_existent")
            self.assertEqual(resp404.status_code, 404)

    def test_06_spectrogram_and_report_artifacts_generation(self):
        """Verify spectrogram colormapping and multi-format report artifact persistence."""
        job_id = "job_report_artifact_test"
        mel_dummy = np.random.randn(64, 313).astype(np.float32)

        # Save spectrogram
        spec_art = save_spectrogram_as_artifact(job_id, mel_dummy)
        self.assertTrue(Path(STORAGE_ROOT / spec_art["storage_relative_path"]).exists())
        self.assertEqual(spec_art["mime_type"], "image/png")

        # Save report
        rep_art = save_report_as_artifact(job_id, "markdown", "# Test Report Content", "test_report.md")
        self.assertTrue(Path(STORAGE_ROOT / rep_art["storage_relative_path"]).exists())
        self.assertEqual(rep_art["mime_type"], "text/markdown")

    def test_07_storage_retention_and_cleanup_policies(self):
        """Verify dry-run and live execution of storage retention policies."""
        stats = get_storage_stats()
        self.assertIn("retention_policy", stats)
        self.assertIn("storage_breakdown", stats)
        self.assertIn("host_disk", stats)
        self.assertGreater(stats["retention_policy"]["retention_days"], 0)

        # Test dry-run cleanup
        conn = database.get_connection()
        try:
            audit_dry = run_storage_cleanup(conn, retention_days=365, clean_orphans=False, dry_run=True)
            self.assertTrue(audit_dry["dry_run"])
            self.assertEqual(audit_dry["retention_days"], 365)
        finally:
            conn.close()

        # Query policy endpoint live
        if self.server_online:
            resp = requests.get(f"{self.base_url}/api/storage/policy")
            self.assertEqual(resp.status_code, 200)
            policy_data = resp.json()
            self.assertIn("retention_policy", policy_data)
            self.assertIn("storage_breakdown", policy_data)

    def test_08_public_api_scrubs_host_filesystem_paths(self):
        """Verify that public API endpoints never leak private server filesystem paths."""
        if not self.server_online:
            raise unittest.SkipTest("Live API server is not running on 127.0.0.1:8000 (skipping live test)")
        resp = requests.get(f"{self.base_url}/api/jobs?limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for job in data.get("jobs", []):
            self.assertNotIn("file_path", job, "Host filesystem path must NOT be present in public job data")
            # Ensure no value contains raw Windows drive path
            for k, v in job.items():
                if isinstance(v, str):
                    self.assertNotIn("C:\\Users", v, f"Key '{k}' must not leak private host directory path")

    def test_09_spectrogram_artifact_robustness(self):
        """Verify save_spectrogram_as_artifact handles zero-size arrays and empty metadata without crashing."""
        test_job_id = f"job_spec_robust_{uuid.uuid4().hex[:8]}"
        database.create_job(
            job_id=test_job_id,
            filename="robust_spec.mp4",
            file_path=f"uploads/{test_job_id}_robust_spec.mp4",
            file_size_bytes=1024,
            file_hash_sha256="dummy_hash"
        )

        # Test empty 0-size mel
        res_empty = save_spectrogram_as_artifact(test_job_id, np.zeros((0,)))
        self.assertEqual(res_empty["artifact_type"], "spectrogram")
        self.assertFalse(res_empty["metadata"]["audio_available"])
        self.assertIn("axes_metadata", res_empty["metadata"])

        # Test valid mel with empty dict axes_metadata
        res_meta = save_spectrogram_as_artifact(
            test_job_id,
            np.ones((64, 313, 1), dtype=np.float32),
            axes_metadata={}
        )
        self.assertEqual(res_meta["artifact_type"], "spectrogram")

    def test_10_storage_cleanup_tuple_connection_and_orphan_protection(self):
        """Verify run_storage_cleanup works with standard tuple connections and never purges active uploads."""
        # 1. Standard SQLite connection (tuple rows)
        mem_conn = sqlite3.connect(":memory:")
        mem_conn.execute("CREATE TABLE analysis_jobs (job_id TEXT, created_at TEXT, status TEXT, file_path TEXT, filename TEXT)")
        mem_conn.execute("CREATE TABLE job_artifacts (artifact_id TEXT, job_id TEXT, storage_relative_path TEXT)")
        mem_conn.execute("CREATE TABLE frame_attributions (attribution_id TEXT, job_id TEXT)")

        # Create active test file in UPLOADS_DIR
        active_filename = f"protect_test_{uuid.uuid4().hex[:8]}.mp4"
        active_file = UPLOADS_DIR / active_filename
        active_file.write_bytes(b"test active video data")

        try:
            mem_conn.execute(
                "INSERT INTO analysis_jobs VALUES (?, datetime('now'), 'completed', ?, ?)",
                ("job_active", str(active_file), active_filename)
            )

            # Run cleanup with standard connection
            audit = run_storage_cleanup(mem_conn, clean_orphans=True, dry_run=False)
            self.assertEqual(audit["expired_jobs_purged"], 0)

            # Ensure active video was NOT purged
            self.assertTrue(active_file.exists(), "Active uploaded video must NOT be deleted as orphan!")
        finally:
            if active_file.exists():
                active_file.unlink()
            mem_conn.close()

if __name__ == "__main__":
    unittest.main()
