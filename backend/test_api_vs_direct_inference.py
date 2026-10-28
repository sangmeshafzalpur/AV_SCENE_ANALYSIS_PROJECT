"""
API vs Direct Inference Verification Script
===========================================
Compares:
1. Trusted Direct Multimodal Inference (`run_direct_multimodal_inference`)
2. Live Synchronous Prediction API (`POST /predict`)
3. Live Asynchronous Pipeline (`POST /api/jobs/analyze` -> `GET /api/jobs/{id}`)
4. Model Metadata & Readiness Endpoints (`GET /api/model/metadata`, `GET /api/model/readiness`)
"""

import os
import sys
import time
import requests
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from services.model_inference_service import (
    run_direct_multimodal_inference,
    locate_model_file,
    compute_sha256,
    CLASS_INDEX_MAP,
    CLASS_NAMES,
    MODEL_FILENAME
)
from services.video_preprocessor import extract_video_frames_with_metadata
from services.audio_preprocessor import compute_audio_logmel_features

BASE_URL = "http://127.0.0.1:8000"
TEST_VIDEO_PATH = WORKSPACE_DIR / "test_video.mp4"
EXPECTED_SHA256 = "404f374484b2304e44074dd44a81a8130add1b22c22dcd7a7e21ea56f8763fde"

def main():
    print("=" * 70)
    print("RUNNING API VS DIRECT INFERENCE VERIFICATION")
    print("=" * 70)

    assert TEST_VIDEO_PATH.exists(), f"Video file {TEST_VIDEO_PATH} missing."

    # ----------------------------------------------------
    # Step 1: Live Asynchronous Analysis Engine (/api/jobs/analyze)
    # ----------------------------------------------------
    print("\n[1] Executing Live Asynchronous Engine (POST /api/jobs/analyze)...")
    t1 = time.time()
    with open(TEST_VIDEO_PATH, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/jobs/analyze",
            params={"force_reanalysis": "true"},
            files={"video": ("test_video.mp4", f, "video/mp4")}
        )
    assert resp.status_code in [200, 202], f"Async analyze failed ({resp.status_code}): {resp.text}"
    job_info = resp.json()
    job_id = job_info["job_id"]
    print(f"  -> Submitted job ID: {job_id}")

    # Poll until completed
    max_wait = 180
    job_data = None
    start_poll = time.time()
    while time.time() - start_poll < max_wait:
        poll_resp = requests.get(f"{BASE_URL}/api/jobs/{job_id}")
        assert poll_resp.status_code == 200, f"Poll job failed: {poll_resp.text}"
        job_data = poll_resp.json()
        status = job_data.get("status")
        if status == "completed":
            print(f"  -> Job completed in {round(time.time() - start_poll, 2)}s.")
            break
        elif status == "failed":
            raise RuntimeError(f"Job failed with error: {job_data.get('error_message')}")
        time.sleep(1.0)
    
    assert job_data.get("status") == "completed", "Job did not complete in time."
    extracted_transcript = job_data.get("transcript")
    print(f"  -> Async Prediction: {job_data['prediction']}")
    print(f"  -> Async Confidence: {job_data['confidence']:.6f}")
    print(f"  -> Async Scores: {job_data['probabilities']}")
    print(f"  -> Async Transcript: '{extracted_transcript}'")
    print(f"  -> Async Model Checksum: {job_data.get('model_checksum')}")

    # ----------------------------------------------------
    # Step 2: Live Synchronous API Endpoint (/predict)
    # ----------------------------------------------------
    print("\n[2] Executing Live Synchronous API Endpoint (POST /predict)...")
    t2 = time.time()
    with open(TEST_VIDEO_PATH, "rb") as f:
        resp = requests.post(f"{BASE_URL}/predict", files={"video": ("test_video.mp4", f, "video/mp4")})
    sync_api_duration = round(time.time() - t2, 3)
    assert resp.status_code == 200, f"Sync predict failed ({resp.status_code}): {resp.text}"
    sync_data = resp.json()
    print(f"  -> API Prediction: {sync_data['prediction']}")
    print(f"  -> API Confidence: {sync_data['confidence']:.6f}")
    print(f"  -> API Scores: {sync_data['probabilities']}")
    print(f"  -> Model SHA256: {sync_data.get('model_identification', {}).get('sha256')}")
    print(f"  -> Score Semantics: {sync_data.get('score_semantics')} (is_calibrated={sync_data.get('is_calibrated')})")
    print(f"  -> API Roundtrip Duration: {sync_api_duration}s")

    # ----------------------------------------------------
    # Step 3: Trusted Direct Inference Path (with identical extracted inputs)
    # ----------------------------------------------------
    print("\n[3] Executing Trusted Direct Inference Path (Identical Inputs)...")
    t0 = time.time()
    video_res = extract_video_frames_with_metadata(TEST_VIDEO_PATH)
    frames = video_res["frames"]
    audio_res = compute_audio_logmel_features(TEST_VIDEO_PATH)
    mel = audio_res["mel_tensor"]
    audio_available = audio_res["audio_available"]

    direct_res = run_direct_multimodal_inference(
        video_frames=frames,
        audio_spectrogram=mel,
        audio_available=audio_available,
        transcript=extracted_transcript
    )
    direct_duration = round(time.time() - t0, 3)
    print(f"  -> Direct Prediction: {direct_res['predicted_class']}")
    print(f"  -> Direct Confidence: {direct_res['confidence']:.6f}")
    print(f"  -> Direct Scores: {direct_res['class_scores']}")
    print(f"  -> Model SHA256: {direct_res['model_identification']['sha256']}")
    print(f"  -> Score Semantics: {direct_res['score_semantics']} (is_calibrated={direct_res['is_calibrated']})")
    print(f"  -> Direct Duration: {direct_duration}s")

    # Also test Direct Inference with missing transcript to verify fallback policy
    print("\n[3b] Executing Direct Inference with Missing Transcript Fallback...")
    direct_notext = run_direct_multimodal_inference(
        video_frames=frames,
        audio_spectrogram=mel,
        audio_available=audio_available,
        transcript=None
    )
    print(f"  -> Direct (No Text) Prediction: {direct_notext['predicted_class']} (Confidence: {direct_notext['confidence']:.6f})")
    print(f"  -> Direct (No Text) Warnings: {direct_notext['warnings']}")
    assert direct_notext["modality_availability"]["speech"] is False
    assert len(direct_notext["warnings"]) > 0

    # ----------------------------------------------------
    # Step 4: Consistency Assertions
    # ----------------------------------------------------
    print("\n[4] Validating Cross-Path Consistency Assertions...")
    # Check predictions match
    assert direct_res["predicted_class"] == sync_data["prediction"], (
        f"Direct ({direct_res['predicted_class']}) != Sync API ({sync_data['prediction']})"
    )
    assert direct_res["predicted_class"] == job_data["prediction"], (
        f"Direct ({direct_res['predicted_class']}) != Async Job ({job_data['prediction']})"
    )
    print("  [PASS] Predicted class matches across Direct, Sync API, and Async Job: " + direct_res["predicted_class"])

    # Check scores consistency
    for cls in ["Accident", "Fire", "Normal"]:
        d_score = direct_res["class_scores"][cls]
        s_score = sync_data["probabilities"][cls]
        j_score = job_data["probabilities"][cls]
        assert abs(d_score - s_score) < 1e-4, f"Direct score ({d_score}) != Sync score ({s_score}) for {cls}"
        assert abs(d_score - j_score) < 1e-4, f"Direct score ({d_score}) != Async score ({j_score}) for {cls}"
    print("  [PASS] Numerical class-wise scores identical across all three execution paths.")

    # Check model checksum
    assert direct_res["model_identification"]["sha256"] == EXPECTED_SHA256
    assert sync_data["model_identification"]["sha256"] == EXPECTED_SHA256
    assert job_data["model_checksum"] == EXPECTED_SHA256
    print(f"  [PASS] Model SHA256 checksum verified across all paths: {EXPECTED_SHA256}")

    # Check calibration semantics
    assert direct_res["is_calibrated"] is False
    assert sync_data["is_calibrated"] is False
    print("  [PASS] Calibration transparency verified: scores declared uncalibrated softmax logits.")

    # ----------------------------------------------------
    # Step 5: Metadata & Readiness API Checks
    # ----------------------------------------------------
    print("\n[5] Checking Model Metadata & Readiness Endpoints...")
    meta_resp = requests.get(f"{BASE_URL}/api/model/metadata")
    assert meta_resp.status_code == 200, f"Metadata endpoint failed: {meta_resp.text}"
    meta = meta_resp.json()
    assert meta["model_identification"]["active_model_filename"] == MODEL_FILENAME
    assert meta["model_identification"]["sha256_checksum"] == EXPECTED_SHA256
    assert meta["verified_facts"]["parameters"]["total_parameters"] == 6455846
    assert meta["verified_facts"]["class_mapping"]["0"] == "Accident"
    assert meta["verified_facts"]["class_mapping"]["1"] == "Fire"
    assert meta["verified_facts"]["class_mapping"]["2"] == "Normal"
    print("  [PASS] GET /api/model/metadata returns verified facts and architecture layers.")

    ready_resp = requests.get(f"{BASE_URL}/api/model/readiness")
    assert ready_resp.status_code == 200, f"Readiness endpoint failed: {ready_resp.text}"
    readiness = ready_resp.json()
    assert readiness["ready"] is True
    assert readiness["components"]["model_file"]["status"] == "ready"
    assert readiness["components"]["multimodal_keras_model"]["status"] == "loaded"
    print("  [PASS] GET /api/model/readiness reports status=ready and all models loaded.")

    print("\n" + "=" * 70)
    print("ALL API VS DIRECT INFERENCE VALIDATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
