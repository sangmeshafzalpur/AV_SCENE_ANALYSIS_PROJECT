import os
import sys
import time
import unittest
import requests
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_DIR.parent
TEST_VIDEO_PATH = WORKSPACE_ROOT / "test_video.mp4"
BASE_URL = "http://127.0.0.1:8000"

class TestBackendApiAndEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Wait up to 30 seconds for server to be fully ready
        max_retries = 30
        for i in range(max_retries):
            try:
                r = requests.get(f"{BASE_URL}/api/health", timeout=2)
                if r.status_code == 200 and r.json().get("checks", {}).get("multimodal_model") == "loaded":
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("Backend server did not become ready in time.")

    def test_01_health_and_readiness(self):
        # 1. /health
        r_health = requests.get(f"{BASE_URL}/health", timeout=5)
        self.assertEqual(r_health.status_code, 200)
        h_data = r_health.json()
        self.assertTrue(h_data["healthy"])
        self.assertEqual(h_data["checks"]["database"], "ok")
        self.assertEqual(h_data["checks"]["multimodal_model"], "loaded")

        # 2. /api/readiness
        r_ready = requests.get(f"{BASE_URL}/api/readiness", timeout=5)
        self.assertEqual(r_ready.status_code, 200)
        ready_data = r_ready.json()
        self.assertTrue(ready_data["ready"])
        self.assertEqual(ready_data["components"]["model_file"]["status"], "ready")

    def test_02_invalid_file_uploads(self):
        # 1. Invalid extension (.txt)
        files = {"video": ("malicious.txt", b"This is not a video.", "text/plain")}
        r = requests.post(f"{BASE_URL}/api/jobs/analyze", files=files, timeout=5)
        self.assertEqual(r.status_code, 400)
        self.assertIn("Unsupported file extension", r.json()["detail"])

        # 2. Empty file (0 bytes)
        files_empty = {"video": ("empty.mp4", b"", "video/mp4")}
        r_empty = requests.post(f"{BASE_URL}/api/jobs/analyze", files=files_empty, timeout=5)
        self.assertEqual(r_empty.status_code, 400)
        self.assertIn("too small or empty", r_empty.json()["detail"])

    def test_03_asynchronous_job_lifecycle(self):
        self.assertTrue(TEST_VIDEO_PATH.exists(), f"Test video not found at {TEST_VIDEO_PATH}")

        # 1. Submit analysis job
        with open(TEST_VIDEO_PATH, "rb") as f:
            files = {"video": ("test_video.mp4", f, "video/mp4")}
            r_submit = requests.post(f"{BASE_URL}/api/jobs/analyze?force_reanalysis=true", files=files, timeout=10)
        
        self.assertEqual(r_submit.status_code, 202)
        job_data = r_submit.json()
        job_id = job_data["job_id"]
        self.assertTrue(job_id.startswith("job_"))
        self.assertIn(job_data["status"], ("queued", "processing"))

        # 2. Poll job status until completed
        max_wait = 180 # seconds (allows CPU-based 12-frame attribution pipeline to complete)
        start_wait = time.time()
        final_job = None
        while time.time() - start_wait < max_wait:
            r_status = requests.get(f"{BASE_URL}/api/jobs/{job_id}", timeout=5)
            self.assertEqual(r_status.status_code, 200)
            final_job = r_status.json()
            if final_job["status"] in ("completed", "failed"):
                break
            time.sleep(1)

        self.assertEqual(final_job["status"], "completed", f"Job failed or timed out: {final_job.get('error_message')}")
        self.assertEqual(final_job["progress"], 1.0)
        self.assertIn(final_job["prediction"], ["Fire", "Accident", "Normal"])
        self.assertGreater(final_job["confidence"], 0.0)
        self.assertLessEqual(final_job["confidence"], 1.0)
        
        # Verify evidence payload
        evidence = final_job["evidence"]
        self.assertIn("visual_evidence", evidence)
        self.assertEqual(evidence["visual_evidence"]["sampled_frame_count"], 12)
        self.assertIn("audio_evidence", evidence)
        self.assertIn("speech_evidence", evidence)
        self.assertIn("timings", evidence)

        # 3. Test report generation in all 4 formats
        for fmt, expected_mime in [
            ("json", "application/json"),
            ("markdown", "text/markdown"),
            ("html", "text/html"),
            ("text", "text/plain")
        ]:
            r_rep = requests.get(f"{BASE_URL}/api/jobs/{job_id}/report?format={fmt}", timeout=5)
            self.assertEqual(r_rep.status_code, 200, f"Report format {fmt} failed")
            self.assertIn(expected_mime, r_rep.headers.get("content-type", ""))
            self.assertIn("attachment", r_rep.headers.get("content-disposition", ""))

    def test_04_searchable_history_and_filters(self):
        r = requests.get(f"{BASE_URL}/api/jobs?limit=10&offset=0", timeout=5)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("total", data)
        self.assertIn("jobs", data)
        self.assertGreaterEqual(data["total"], 1)

        # Filter by status completed
        r_comp = requests.get(f"{BASE_URL}/api/jobs?status=completed", timeout=5)
        self.assertEqual(r_comp.status_code, 200)
        for j in r_comp.json()["jobs"]:
            self.assertEqual(j["status"], "completed")

        # Search query by filename
        r_search = requests.get(f"{BASE_URL}/api/jobs?query=test_video", timeout=5)
        self.assertEqual(r_search.status_code, 200)
        for j in r_search.json()["jobs"]:
            self.assertIn("test_video", j["filename"].lower())

    def test_05_job_cancellation_and_not_found(self):
        # 1. Nonexistent job
        r_notfound = requests.get(f"{BASE_URL}/api/jobs/job_nonexistent_xyz", timeout=5)
        self.assertEqual(r_notfound.status_code, 404)

        r_cancel_nf = requests.post(f"{BASE_URL}/api/jobs/job_nonexistent_xyz/cancel", timeout=5)
        self.assertEqual(r_cancel_nf.status_code, 404)

        # 2. Cancel an already completed job (should be rejected with 400)
        r_jobs = requests.get(f"{BASE_URL}/api/jobs?status=completed&limit=1", timeout=5)
        jobs = r_jobs.json()["jobs"]
        if jobs:
            c_job_id = jobs[0]["job_id"]
            r_cancel_completed = requests.post(f"{BASE_URL}/api/jobs/{c_job_id}/cancel", timeout=5)
            self.assertEqual(r_cancel_completed.status_code, 400)
            self.assertIn("cannot be cancelled", r_cancel_completed.json()["detail"])

    def test_06_evaluation_metadata(self):
        r_eval = requests.get(f"{BASE_URL}/api/evaluation/metadata", timeout=5)
        self.assertEqual(r_eval.status_code, 200)
        eval_data = r_eval.json()
        
        # Verify official dataset numbers
        self.assertEqual(eval_data["dataset_metadata"]["training_set"]["total"], 1080)
        self.assertEqual(eval_data["dataset_metadata"]["blind_test_set"]["total"], 270)
        self.assertAlmostEqual(eval_data["blind_test_evaluation"]["overall_accuracy"], 0.981481, places=4)
        self.assertEqual(len(eval_data["class_wise_performance"]), 3)
        self.assertIn("Actual_Fire", eval_data["confusion_matrix"])

    def test_07_system_statistics(self):
        r_stats = requests.get(f"{BASE_URL}/api/system/stats", timeout=5)
        self.assertEqual(r_stats.status_code, 200)
        stats = r_stats.json()
        
        self.assertEqual(stats["status"], "operational")
        self.assertGreater(stats["uptime_seconds"], 0)
        self.assertIn("process_memory_rss_mb", stats["resources"])
        self.assertGreater(stats["resources"]["process_memory_rss_mb"], 100) # Models loaded
        self.assertIn("database_metrics", stats["job_engine"])
        self.assertGreaterEqual(stats["job_engine"]["database_metrics"]["total_jobs"], 1)

    def test_08_legacy_sync_predict_endpoint_persisted(self):
        with open(TEST_VIDEO_PATH, "rb") as f:
            files = {"video": ("test_video.mp4", f, "video/mp4")}
            r_pred = requests.post(f"{BASE_URL}/predict", files=files, timeout=60)
        
        self.assertEqual(r_pred.status_code, 200)
        pred_data = r_pred.json()
        self.assertIn(pred_data["prediction"], ["Fire", "Accident", "Normal"])
        self.assertGreater(pred_data["confidence"], 0.0)
        self.assertLessEqual(pred_data["confidence"], 1.0)
        self.assertIn("job_id", pred_data)
        
        # Verify that this sync run was recorded in the database
        sync_job_id = pred_data["job_id"]
        r_job = requests.get(f"{BASE_URL}/api/jobs/{sync_job_id}", timeout=5)
        self.assertEqual(r_job.status_code, 200)
        self.assertIn(r_job.json()["prediction"], ["Fire", "Accident", "Normal"])

if __name__ == "__main__":
    unittest.main()
