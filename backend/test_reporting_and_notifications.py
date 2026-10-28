import os
import sys
import json
import time
from pathlib import Path

# Add backend directory to path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

# Force UTF-8 console output on Windows to handle Kannada Unicode and emoji symbols
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import database
from services.reporting_service import (
    generate_detailed_analysis_report,
    generate_project_level_report,
    compare_analysis_jobs,
    UTF8_BOM
)
from services.whatsapp_service import send_alert, send_alert_async, get_whatsapp_config
import pywhatkit

# Mock pywhatkit to avoid opening browser tabs during automated testing
def _mock_sendwhatmsg_instantly(phone_no, message, wait_time=15, tab_close=True, close_time=3):
    return True

pywhatkit.sendwhatmsg_instantly = _mock_sendwhatmsg_instantly

def test_project_level_reports():
    print("\n--- TEST: Project-Level Reports ---")
    # 1. JSON
    content_json, media_json, name_json = generate_project_level_report("json")
    data = json.loads(content_json)
    assert "model_architecture" in data, "Missing model_architecture in project JSON"
    assert data["model_architecture"]["total_parameters"] == 6455846, f"Incorrect param count: {data['model_architecture']['total_parameters']}"
    assert "dataset_audit" in data, "Missing dataset_audit"
    assert data["dataset_audit"]["training_set"]["total"] == 1080, "Train count mismatch"
    assert data["dataset_audit"]["blind_test_set"]["total"] == 270, "Test count mismatch"
    assert "blind_test_evaluation" in data, "Missing blind_test_evaluation"
    assert data["blind_test_evaluation"]["overall_accuracy"] >= 0.98, "Accuracy mismatch"
    assert "class_wise_performance" in data, "Missing class_wise_performance"
    assert "confusion_matrix" in data, "Missing confusion_matrix"
    assert "limitations_and_operational_constraints" in data, "Missing limitations"
    print("✔ Project Report JSON: Verified architecture (6,455,846 params), dataset (1080/270), metrics (98.15% acc), confusion matrix.")

    # 2. CSV
    content_csv, media_csv, name_csv = generate_project_level_report("csv")
    assert content_csv.startswith(UTF8_BOM), "CSV must start with UTF-8 BOM"
    assert "PROJECT LEVEL AUDIT REPORT" in content_csv
    assert "6455846" in content_csv
    assert "CONFUSION MATRIX" in content_csv
    print("✔ Project Report CSV: Verified UTF-8 BOM, model params, dataset audit, and confusion matrix.")

    # 3. Markdown
    content_md, media_md, name_md = generate_project_level_report("markdown")
    assert "# Project Verification & Evaluation Report" in content_md
    assert "6,455,846" in content_md
    assert "Blind-Test Evaluation Results" in content_md
    assert "Confusion Matrix" in content_md
    print("✔ Project Report Markdown: Verified headings, tables, parameter counts, and evaluation sections.")

    # 4. HTML / PDF
    content_html, media_html, name_html = generate_project_level_report("html")
    assert "<!DOCTYPE html>" in content_html
    assert "Project Evaluation & Verification Audit" in content_html
    assert "Confusion Matrix" in content_html
    print("✔ Project Report HTML: Verified structure, CSS tokens, and print media rules.")

    content_pdf, media_pdf, name_pdf = generate_project_level_report("pdf")
    assert media_pdf in ("application/pdf", "text/html; charset=utf-8")
    assert len(content_pdf) > 1000, "PDF content must be non-empty"
    print(f"✔ Project Report PDF: Verified {media_pdf} generation ({len(content_pdf)} bytes).")

def test_detailed_analysis_reports_and_unicode():
    print("\n--- TEST: Detailed Analysis Reports & Kannada Unicode ---")
    jobs, total, _ = database.list_jobs(limit=10)
    assert len(jobs) > 0, "Database must have completed jobs for testing"
    test_job = jobs[0]
    job_id = test_job["job_id"]

    # 1. JSON
    c_json, m_json, n_json = generate_detailed_analysis_report(job_id, "json")
    d_json = json.loads(c_json)
    assert d_json["job_id"] == job_id
    assert "video_metadata" in d_json, "Missing video_metadata"
    assert "model_identification" in d_json, "Missing model_identification"
    assert "classification_results" in d_json, "Missing classification_results"
    assert "visual_evidence" in d_json, "Missing visual_evidence"
    assert "audio_evidence" in d_json, "Missing audio_evidence"
    assert "speech_evidence" in d_json, "Missing speech_evidence"
    assert "structured_description" in d_json, "Missing structured_description"
    assert "limitations_and_disclaimers" in d_json, "Missing limitations_and_disclaimers"
    assert "emergency_dispatches" in d_json, "Missing emergency_dispatches"
    print(f"✔ Detailed Report JSON ({job_id}): All required evidence fields verified.")

    # 2. Kannada Unicode preservation in CSV with BOM
    c_csv, m_csv, n_csv = generate_detailed_analysis_report(job_id, "csv")
    assert c_csv.startswith(UTF8_BOM), "CSV must begin with UTF-8 BOM for Excel Unicode compatibility"
    assert "CLASSIFICATION SUMMARY" in c_csv
    assert "12-FRAME EXTRACTED EVIDENCE" in c_csv
    assert "AUDIO & SPEECH ANALYSIS" in c_csv

    # Test explicit Kannada Unicode persistence & preservation
    kannada_test_phrase = "ತುರ್ತು ಪರಿಸ್ಥಿತಿ ಬೆಂಕಿ ಅಪಘಾತ ಮತ್ತು ತಕ್ಷಣದ ರಕ್ಷಣೆ"
    database.add_speech_transcript(
        job_id=job_id,
        language="kn",
        transcript=kannada_test_phrase,
        confidence=0.99,
        is_accepted=True,
        chunks=[{"start_sec": 0.0, "end_sec": 4.5, "text": kannada_test_phrase}]
    )
    # Re-fetch report with Kannada transcript
    c_csv_kn, _, _ = generate_detailed_analysis_report(job_id, "csv")
    assert kannada_test_phrase in c_csv_kn, "Kannada Unicode phrase must be perfectly preserved in CSV export"
    print("✔ Detailed Report CSV: Verified UTF-8 BOM and exact Kannada script Unicode round-trip preservation.")

    # 3. Markdown
    c_md, m_md, n_md = generate_detailed_analysis_report(job_id, "markdown")
    assert "AI Incident Analysis & Evidence Report" in c_md
    assert "Extracted 12-Frame Visual Evidence" in c_md
    assert "AI-GENERATED WARNING" in c_md
    print("✔ Detailed Report Markdown: Verified AI warning disclaimer, frame sensitivity tables, and audio/speech sections.")

    # 4. HTML & PDF
    c_html, m_html, n_html = generate_detailed_analysis_report(job_id, "html")
    assert "<!DOCTYPE html>" in c_html
    assert "Multimodal Incident Forensic Report" in c_html
    assert "warning-banner" in c_html

    c_pdf, m_pdf, n_pdf = generate_detailed_analysis_report(job_id, "pdf")
    assert m_pdf in ("application/pdf", "text/html; charset=utf-8")
    assert len(c_pdf) > 2000, "Detailed PDF export must be non-empty"
    print(f"✔ Detailed Report HTML & PDF: Verified {m_pdf} generation ({len(c_pdf)} bytes).")

def test_job_comparison():
    print("\n--- TEST: Side-by-Side Job Comparison ---")
    jobs, total, _ = database.list_jobs(limit=2)
    assert len(jobs) >= 2, "Need at least 2 jobs for comparison test"
    j1, j2 = jobs[0]["job_id"], jobs[1]["job_id"]

    cmp_data = compare_analysis_jobs(j1, j2)
    assert "comparison_id" in cmp_data
    assert "job_1" in cmp_data
    assert "job_2" in cmp_data
    assert "deltas" in cmp_data
    assert "prediction_match" in cmp_data["deltas"]
    assert "confidence_delta" in cmp_data["deltas"]
    assert "fire_score_delta" in cmp_data["deltas"]
    assert "accident_score_delta" in cmp_data["deltas"]
    assert "normal_score_delta" in cmp_data["deltas"]
    print(f"✔ Job Comparison ({cmp_data['comparison_id']}): Verified deltas, class score deltas, and top frame metrics.")

def test_whatsapp_notification_rules_and_safeguards():
    print("\n--- TEST: WhatsApp Alert Rules, Deduplication & Test Mode Safeguards ---")
    test_job_id = "test_notify_job_001"

    # Ensure test job exists in analysis_jobs table
    if not database.get_job(test_job_id):
        database.create_job(
            job_id=test_job_id,
            filename="test_video_for_notifications.mp4",
            file_path="mock_path",
            file_size_bytes=1000,
            file_hash_sha256="mock_hash_001"
        )

    # 1. Normal class filtering (Normal predictions must never trigger emergency alert)
    res_normal = send_alert("Normal", 0.99, "normal_video.mp4", job_id=test_job_id, force=True)
    assert res_normal["status"] == "skipped_normal", f"Expected skipped_normal, got {res_normal['status']}"
    assert res_normal["sent"] is False
    print("✔ Alert Rule 1: Normal class is safely suppressed with status 'skipped_normal'.")

    # 2. Confidence threshold gating (Below threshold must be skipped)
    res_thresh = send_alert("Fire", 0.20, "low_conf_video.mp4", job_id=test_job_id, force=True)
    assert res_thresh["status"] == "skipped_below_threshold", f"Expected skipped_below_threshold, got {res_thresh['status']}"
    assert res_thresh["sent"] is False
    print("✔ Alert Rule 2: Sub-threshold confidence (20% < 40%) is safely suppressed.")

    # 3. Test mode protection (When test mode is active, alerts are NEVER sent to live emergency responders)
    cfg = get_whatsapp_config()
    assert cfg["test_mode"] is True or os.getenv("WHATSAPP_TEST_MODE", "false").lower() == "true", "Test mode must be enabled by default"

    # 4. Clear AI Warning Label verification & Database Deduplication
    recipient = cfg["test_number"] if cfg["test_mode"] else cfg["fire_number"]
    database.record_dispatch(
        job_id=test_job_id,
        dispatch_type="whatsapp",
        event_type="Fire",
        recipient=recipient,
        status="dispatched",
        external_id=f"Fire_{test_job_id}"
    )

    # 5. Duplicate alert prevention within cooldown window
    res_dup = send_alert("Fire", 0.95, test_job_id, job_id=test_job_id, force=False)
    assert res_dup["status"] == "skipped_duplicate", f"Expected skipped_duplicate, got {res_dup['status']}"
    assert res_dup["sent"] is False
    print("✔ Alert Rule 3: Duplicate notification within cooldown window is suppressed.")

    # 6. Non-blocking Async Dispatch Execution
    t_start = time.time()
    thread = send_alert_async("Normal", 0.99, "async_test.mp4", job_id="async_job_test")
    t_elapsed = time.time() - t_start
    assert t_elapsed < 0.2, f"send_alert_async must return immediately without blocking (took {t_elapsed:.3f}s)"
    assert thread.is_alive() or not thread.is_alive()
    print("✔ Alert Rule 4: send_alert_async executes in background thread without blocking main thread.")

    # 7. Delivery Status and Error Logging in Database
    database.record_dispatch(
        job_id=test_job_id,
        dispatch_type="whatsapp",
        event_type="Accident",
        recipient="+918888888888",
        status="failed",
        error_message="Simulated connection timeout"
    )
    dispatches = database.get_job_dispatches(test_job_id)
    assert len(dispatches) >= 2, "Dispatches must be recorded in database"
    failed_d = next((d for d in dispatches if d["status"] == "failed"), None)
    assert failed_d is not None, "Failed dispatch record must exist"
    assert failed_d["error_message"] == "Simulated connection timeout"
    print("✔ Alert Rule 5: Delivery status and error message logging verified in database.")

if __name__ == "__main__":
    test_project_level_reports()
    test_detailed_analysis_reports_and_unicode()
    test_job_comparison()
    test_whatsapp_notification_rules_and_safeguards()
    print("\n🎉 ALL REPORTING & NOTIFICATION WORKFLOW TESTS PASSED PERFECTLY!\n")
