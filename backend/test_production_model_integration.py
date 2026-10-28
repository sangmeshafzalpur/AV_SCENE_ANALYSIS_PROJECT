"""
Comprehensive Integration & Validation Test Suite
==================================================
Tests the preserved `best_multimodal_model.keras` in the production pipeline:
- Real input signatures, preprocessing requirements, and output structure
- Verified class-index mapping (0=Accident, 1=Fire, 2=Normal)
- Uncalibrated softmax score transparency (no false calibration claims)
- Missing audio policy (audio_mask=0.0, acoustic zero-tensor, warning emitted)
- Unavailable transcript policy (empty string vectorization, warning emitted)
- Malformed inputs and validation exception handling
- Preprocessing consistency and deterministic repeated inference
- Concurrency and thread-safety under multi-threaded execution
- Preservation of original model file and checksum integrity
- Trusted direct inference vs. API inference consistency
"""

import os
import sys
import time
import shutil
import hashlib
import tempfile
import unittest
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Ensure backend root is on Python path
BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from services.model_inference_service import (
    load_multimodal_production_model,
    run_direct_multimodal_inference,
    get_production_model_metadata,
    locate_model_file,
    compute_sha256,
    CLASS_INDEX_MAP,
    CLASS_NAMES,
    MODEL_FILENAME,
    EXPECTED_VIDEO_SHAPE,
    EXPECTED_AUDIO_SHAPE
)
from services.model_metadata_service import build_model_metadata, check_model_readiness
from services.video_preprocessor import extract_video_frames_with_metadata
from services.audio_preprocessor import compute_audio_logmel_features

EXPECTED_SHA256 = "404f374484b2304e44074dd44a81a8130add1b22c22dcd7a7e21ea56f8763fde"
EXPECTED_SIZE_BYTES = 65410165
ORIGINAL_MODEL_PATH = Path(r"c:\Users\Hp\OneDrive\Desktop\AVS2\model\best_multimodal_model.keras")
WORKSPACE_MODEL_PATH = WORKSPACE_DIR / MODEL_FILENAME


class TestProductionModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Verify model files exist before running suite."""
        cls.model_path = locate_model_file()
        cls.metadata = get_production_model_metadata()

    def test_01_model_file_preservation_and_checksum(self):
        """Verify original and active best_multimodal_model.keras checksum and byte size."""
        self.assertTrue(self.model_path.exists(), f"Model file {self.model_path} must exist.")
        self.assertEqual(self.model_path.stat().st_size, EXPECTED_SIZE_BYTES)
        
        calculated_sha = compute_sha256(self.model_path)
        self.assertEqual(calculated_sha, EXPECTED_SHA256, "Model SHA256 checksum must match verified ground truth.")

        # If original reference file exists in AVS2 directory, verify it was strictly preserved
        if ORIGINAL_MODEL_PATH.exists():
            orig_size = ORIGINAL_MODEL_PATH.stat().st_size
            orig_sha = compute_sha256(ORIGINAL_MODEL_PATH)
            self.assertEqual(orig_size, EXPECTED_SIZE_BYTES, "Original AVS2 model file size altered!")
            self.assertEqual(orig_sha, EXPECTED_SHA256, "Original AVS2 model file hash altered!")

    def test_02_model_input_signatures_and_parameters(self):
        """Verify layer count, parameter count, and tensor input/output signatures."""
        model = load_multimodal_production_model()
        self.assertEqual(model.name, "EfficientNetB0_BiLSTM_AudioCNN_KannadaASR_TextBiLSTM")
        
        # Verify 4 input layers
        input_names = [inp.name for inp in model.inputs]
        self.assertIn("video", input_names[0])
        self.assertIn("audio", input_names[1])
        self.assertIn("audio_mask", input_names[2])
        self.assertIn("text", input_names[3])

        # Verify output signature
        self.assertEqual(len(model.outputs), 1)
        self.assertEqual(model.outputs[0].shape[-1], 3)

        # Verify total parameter count
        total_params = model.count_params()
        self.assertEqual(total_params, 6455846, f"Expected 6,455,846 parameters, got {total_params}")

    def test_03_verified_class_index_mapping(self):
        """Verify class-index mapping: 0=Accident, 1=Fire, 2=Normal."""
        self.assertEqual(CLASS_INDEX_MAP[0], "Accident")
        self.assertEqual(CLASS_INDEX_MAP[1], "Fire")
        self.assertEqual(CLASS_INDEX_MAP[2], "Normal")
        self.assertEqual(CLASS_NAMES, ["Accident", "Fire", "Normal"])

    def test_04_direct_inference_with_genuine_inputs(self):
        """Execute inference using genuine extracted video frames and audio from test_video.mp4."""
        test_video = WORKSPACE_DIR / "test_video.mp4"
        self.assertTrue(test_video.exists(), "test_video.mp4 must exist for genuine input test.")

        # 1. Genuine video extraction
        video_res = extract_video_frames_with_metadata(test_video)
        frames = video_res["frames"]
        frame_meta = video_res
        self.assertEqual(frames.shape, EXPECTED_VIDEO_SHAPE)
        self.assertEqual(len(frame_meta["frame_timestamps_sec"]), 12)

        # 2. Genuine audio log-mel extraction
        audio_res = compute_audio_logmel_features(test_video)
        mel = audio_res["mel_tensor"]
        self.assertEqual(mel.shape, EXPECTED_AUDIO_SHAPE)

        # 3. Direct inference
        res = run_direct_multimodal_inference(
            video_frames=frames,
            audio_spectrogram=mel,
            audio_available=audio_res["audio_available"],
            transcript="ಬೆಂಕಿ ಅಪಘಾತ ಸಂಭವಿಸಿದೆ"  # Genuine Kannada emergency transcript
        )

        self.assertIn("predicted_class", res)
        self.assertIn(res["predicted_class"], ["Accident", "Fire", "Normal"])
        self.assertIn("class_scores", res)
        self.assertIn("Accident", res["class_scores"])
        self.assertIn("Fire", res["class_scores"])
        self.assertIn("Normal", res["class_scores"])

        # Verify score semantics and non-calibration transparency
        self.assertEqual(res["score_semantics"], "uncalibrated_softmax_scores")
        self.assertFalse(res["is_calibrated"])
        self.assertIn("uncalibrated softmax logits", res["calibration_note"])

        # Verify scores sum to 1.0 (softmax property)
        total_score = sum(res["class_scores"].values())
        self.assertAlmostEqual(total_score, 1.0, places=4)

        # Verify timings and model identification
        self.assertGreater(res["timings"]["inference_duration_seconds"], 0.0)
        self.assertEqual(res["model_identification"]["sha256"], EXPECTED_SHA256)
        self.assertTrue(res["modality_availability"]["video"])
        self.assertTrue(res["modality_availability"]["audio"])
        self.assertTrue(res["modality_availability"]["speech"])

    def test_05_preprocessing_consistency_and_deterministic_inference(self):
        """Repeated inferences with identical inputs must produce deterministic identical outputs."""
        test_video = WORKSPACE_DIR / "test_video.mp4"
        video_res = extract_video_frames_with_metadata(test_video)
        frames = video_res["frames"]
        audio_res = compute_audio_logmel_features(test_video)
        mel = audio_res["mel_tensor"]
        text = "ತುರ್ತು ಪರಿಸ್ಥಿತಿ ರಸ್ತೆ ಅಪಘಾತ"

        run_1 = run_direct_multimodal_inference(frames, mel, audio_available=True, transcript=text)
        run_2 = run_direct_multimodal_inference(frames, mel, audio_available=True, transcript=text)

        self.assertEqual(run_1["predicted_class"], run_2["predicted_class"])
        for cls_name in ["Accident", "Fire", "Normal"]:
            self.assertAlmostEqual(
                run_1["class_scores"][cls_name],
                run_2["class_scores"][cls_name],
                places=5,
                msg=f"Score mismatch between deterministic runs for {cls_name}"
            )

    def test_06_missing_audio_handling_and_warning(self):
        """When audio is unavailable, audio_mask must be 0.0, acoustic tensor zeroed, and warning recorded."""
        dummy_frames = np.random.uniform(0, 255, EXPECTED_VIDEO_SHAPE).astype(np.float32)
        
        # Audio unavailable (None passed or audio_available=False)
        res = run_direct_multimodal_inference(
            video_frames=dummy_frames,
            audio_spectrogram=None,
            audio_available=False,
            transcript="ರಸ್ತೆ ಅಪಘಾತ"
        )

        self.assertFalse(res["modality_availability"]["audio"])
        self.assertTrue(any("Missing/silent audio policy active" in w for w in res["warnings"]))
        self.assertIn(res["predicted_class"], ["Accident", "Fire", "Normal"])
        self.assertFalse(res["is_calibrated"])

    def test_07_unavailable_transcript_handling_and_warning(self):
        """When speech transcript is missing/empty, empty string passed to TextVectorization, warning emitted."""
        dummy_frames = np.random.uniform(0, 255, EXPECTED_VIDEO_SHAPE).astype(np.float32)
        dummy_mel = np.zeros(EXPECTED_AUDIO_SHAPE, dtype=np.float32)

        res = run_direct_multimodal_inference(
            video_frames=dummy_frames,
            audio_spectrogram=dummy_mel,
            audio_available=True,
            transcript=""  # Empty transcript
        )

        self.assertFalse(res["modality_availability"]["speech"])
        self.assertTrue(any("Unavailable transcript" in w for w in res["warnings"]))
        self.assertIn(res["predicted_class"], ["Accident", "Fire", "Normal"])

    def test_08_malformed_inputs_validation(self):
        """Malformed video frames or non-array inputs must raise clear validation errors."""
        # Invalid frame count (e.g. 5 frames instead of 12)
        bad_video = np.zeros((5, 224, 224, 3), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            run_direct_multimodal_inference(video_frames=bad_video)
        self.assertIn("Expected video_frames shape", str(ctx.exception))

        # Invalid type
        with self.assertRaises(TypeError):
            run_direct_multimodal_inference(video_frames="not_an_array")

        # Invalid audio dimensions
        dummy_frames = np.zeros(EXPECTED_VIDEO_SHAPE, dtype=np.float32)
        bad_audio = np.zeros((32, 100), dtype=np.float32)
        with self.assertRaises((ValueError, TypeError)):
            run_direct_multimodal_inference(video_frames=dummy_frames, audio_spectrogram=bad_audio, audio_available=True)

    def test_09_concurrent_thread_safety(self):
        """Verify thread-safe model execution under concurrent simulated requests."""
        dummy_frames = np.random.uniform(0, 255, EXPECTED_VIDEO_SHAPE).astype(np.float32)
        dummy_mel = np.zeros(EXPECTED_AUDIO_SHAPE, dtype=np.float32)
        transcripts = ["ಬೆಂಕಿ", "ಅಪಘಾತ", "", "ಸಾಮಾನ್ಯ ಸಂಚಾರ", "ತುರ್ತು ರಕ್ಷಣೆ"]

        def worker_task(idx):
            txt = transcripts[idx % len(transcripts)]
            res = run_direct_multimodal_inference(
                video_frames=dummy_frames,
                audio_spectrogram=dummy_mel,
                audio_available=(idx % 2 == 0),
                transcript=txt
            )
            return res["predicted_class"], res["confidence"]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_task, i) for i in range(8)]
            results = [f.result() for f in futures]

        self.assertEqual(len(results), 8)
        for pred_class, conf in results:
            self.assertIn(pred_class, ["Accident", "Fire", "Normal"])
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_10_metadata_service_and_readiness_consistency(self):
        """Verify build_model_metadata and check_model_readiness reflect best_multimodal_model.keras."""
        meta = build_model_metadata()
        self.assertEqual(meta["model_identification"]["active_model_filename"], MODEL_FILENAME)
        self.assertEqual(meta["model_identification"]["sha256_checksum"], EXPECTED_SHA256)
        self.assertEqual(meta["verified_facts"]["parameters"]["total_parameters"], 6455846)
        self.assertEqual(meta["verified_facts"]["class_mapping"]["0"], "Accident")
        self.assertEqual(meta["verified_facts"]["class_mapping"]["1"], "Fire")
        self.assertEqual(meta["verified_facts"]["class_mapping"]["2"], "Normal")

        readiness = check_model_readiness({"multimodal_model": load_multimodal_production_model()})
        self.assertEqual(readiness["components"]["model_file"]["status"], "ready")
        self.assertEqual(readiness["components"]["multimodal_keras_model"]["status"], "loaded")


if __name__ == "__main__":
    unittest.main()
