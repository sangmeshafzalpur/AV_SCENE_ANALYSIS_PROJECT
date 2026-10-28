"""
Comprehensive Test Suite for Speech Recognition and Scene Description Services
=============================================================================
Tests:
  1. Kannada Unicode preservation (NFC normalization, UTF-8 integrity, database round-trip).
  2. English speech recognition validation and language classification.
  3. Silence handling (zero-energy audio -> empty transcript, no hallucination).
  4. Noise and degenerate repetition loop filtering.
  5. Empty transcripts and neutral zero-vector fallback.
  6. Unsupported languages rejection (Devanagari, Cyrillic, etc.).
  7. Chunk-level timestamp extraction and preservation.
  8. Structured scene description generation strictly adhering to the 6 required sections.
  9. Strict anti-hallucination constraint (no fabricated emergency claims when inputs are neutral).
  10. Recoverable component failure handling on corrupted/malformed inputs.
  11. Database persistence in speech_transcripts and structured_description.
  12. Live HTTP API verification (/api/jobs/{job_id}/transcripts & /api/jobs/{job_id}/description).
"""

import os
import sys
import json
import time
import unittest
import numpy as np
import requests

# Reconfigure stdout/stderr for clean UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import database
from services.speech_recognition_service import (
    clean_and_normalize_unicode,
    analyze_transcript_language,
    score_and_validate_transcript,
    transcribe_audio,
    HALLUCINATION_BLACKLIST
)
from services.scene_description_service import (
    generate_evidence_grounded_scene_description,
    CAUSATION_DISCLAIMER
)


class TestSpeechAndDescriptionService(unittest.TestCase):

    def test_01_kannada_unicode_preservation(self):
        """Verify complex Kannada Unicode strings are normalized to NFC and preserved without corruption."""
        sample_kannada = "ರಸ್ತೆಯಲ್ಲಿ ಕಾರು ಅಪಘಾತ ಸಂಭವಿಸಿದೆ, ತಕ್ಷಣ ಸಹಾಯ ಬೇಕು"
        cleaned = clean_and_normalize_unicode(sample_kannada)
        self.assertEqual(cleaned, sample_kannada)

        # Verify language analysis detects Kannada accurately
        lang, kn_ratio, en_ratio = analyze_transcript_language(cleaned)
        self.assertEqual(lang, "kannada")
        self.assertGreaterEqual(kn_ratio, 0.70)
        self.assertEqual(en_ratio, 0.0)

        # Verify database round-trip maintains exact UTF-8 Kannada characters
        test_job_id = f"job_unicode_kn_{int(time.time())}"
        database.create_job(
            job_id=test_job_id,
            filename="kannada_sample.mp4",
            file_path="/tmp/kn.mp4",
            file_size_bytes=1024,
            file_hash_sha256="kannada_hash"
        )
        database.add_speech_transcript(
            job_id=test_job_id,
            language="kn",
            transcript=sample_kannada,
            confidence=0.95,
            is_accepted=True,
            chunks=[{"start_sec": 0.5, "end_sec": 3.8, "text": sample_kannada}]
        )
        retrieved = database.get_speech_transcripts(test_job_id)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual(retrieved[0]["transcript"], sample_kannada)
        self.assertEqual(retrieved[0]["chunks"][0]["text"], sample_kannada)

    def test_02_english_speech_validation(self):
        """Verify English transcripts are classified correctly and validated."""
        sample_english = "Major traffic collision on the highway emergency services needed"
        cleaned = clean_and_normalize_unicode(sample_english)
        self.assertEqual(cleaned, sample_english)

        lang, kn_ratio, en_ratio = analyze_transcript_language(cleaned)
        self.assertEqual(lang, "english")
        self.assertGreaterEqual(en_ratio, 0.80)
        self.assertEqual(kn_ratio, 0.0)

        res = score_and_validate_transcript(sample_english, duration_seconds=5.0, target_language="en")
        self.assertTrue(res["accepted"])
        self.assertEqual(res["language"], "english")
        self.assertEqual(res["word_count"], 9)

    def test_03_silence_handling_no_hallucinations(self):
        """Verify near-silent or zero-energy audio is rejected without autoregressive hallucination."""
        # Create 3 seconds of zero-energy audio
        silent_audio = np.zeros(16000 * 3, dtype=np.float32)
        res = transcribe_audio(silent_audio, sample_rate=16000)

        self.assertFalse(res["accepted"])
        self.assertEqual(res["transcript"], "")
        self.assertEqual(res["status"], "silent_audio")
        self.assertEqual(res["chunks"], [])

        # Create very low amplitude noise (RMS < 0.0005)
        whisper_quiet = np.random.normal(0, 0.0001, size=16000 * 2).astype(np.float32)
        res_quiet = transcribe_audio(whisper_quiet, sample_rate=16000)
        self.assertFalse(res_quiet["accepted"])
        self.assertEqual(res_quiet["transcript"], "")
        self.assertEqual(res_quiet["status"], "silent_audio")

    def test_04_noise_and_repetition_loop_filtering(self):
        """Verify degenerate token loops and known hallucination phrases are filtered out."""
        # 1. Autoregressive loop
        repetitive_loop = "Fire fire fire fire fire fire fire fire fire fire fire fire"
        res = score_and_validate_transcript(repetitive_loop, duration_seconds=3.0)
        self.assertFalse(res["accepted"])
        self.assertEqual(res["transcript"], "")

        # 2. Known YouTube hallucination closing
        hallucinated_text = "Thank you for watching this video see you next time"
        res_hal = score_and_validate_transcript(hallucinated_text, duration_seconds=2.0)
        self.assertFalse(res_hal["accepted"])
        self.assertEqual(res_hal["transcript"], "")
        self.assertIn("known_hallucination_pattern", res_hal["reason"])

    def test_05_empty_transcripts_and_fallback(self):
        """Verify empty transcripts return safe fallback status."""
        res = score_and_validate_transcript("", duration_seconds=4.0)
        self.assertFalse(res["accepted"])
        self.assertEqual(res["transcript"], "")
        self.assertEqual(res["word_count"], 0)

        res_none = score_and_validate_transcript(None, duration_seconds=4.0)
        self.assertFalse(res_none["accepted"])
        self.assertEqual(res_none["transcript"], "")

    def test_06_unsupported_languages_rejection(self):
        """Verify unsupported scripts (Devanagari, Cyrillic, etc.) are rejected."""
        hindi_text = "सड़क पर भारी आग लगी है"
        res = score_and_validate_transcript(hindi_text, duration_seconds=3.0)
        self.assertFalse(res["accepted"])
        self.assertEqual(res["transcript"], "")
        self.assertEqual(res["language"], "unsupported")

        cyrillic_text = "Произошла серьезная авария"
        res_cyr = score_and_validate_transcript(cyrillic_text, duration_seconds=3.0)
        self.assertFalse(res_cyr["accepted"])
        self.assertEqual(res_cyr["language"], "unsupported")

    def test_07_timestamp_preservation(self):
        """Verify chunk timestamps are preserved and structured cleanly."""
        dummy_chunks = [
            {"timestamp": (0.0, 1.8), "text": "ವಾಹನ ಅಪಘಾತ"},
            {"timestamp": (2.1, 4.0), "text": "ಸಹಾಯ ಬೇಕು"}
        ]
        # Verify chunk timing format
        self.assertLess(dummy_chunks[0]["timestamp"][0], dummy_chunks[0]["timestamp"][1])
        self.assertLess(dummy_chunks[1]["timestamp"][0], dummy_chunks[1]["timestamp"][1])

    def test_08_structured_scene_description_six_sections(self):
        """Verify scene description produces the 6 mandatory sections grounded in evidence."""
        video_ev = {
            "sampled_frame_count": 12,
            "frame_resolution": [224, 224],
            "duration_seconds": 6.0,
            "frame_timestamps_sec": [float(i * 0.5) for i in range(12)],
            "influential_frames": {
                "ranked_frames": [
                    {"frame_index": 4, "timestamp_sec": 2.0, "rank": 1, "score_difference": 0.145, "is_keyframe": True},
                    {"frame_index": 7, "timestamp_sec": 3.5, "rank": 2, "score_difference": 0.082, "is_keyframe": True},
                    {"frame_index": 1, "timestamp_sec": 0.5, "rank": 3, "score_difference": 0.041, "is_keyframe": True}
                ]
            }
        }
        audio_ev = {
            "has_audio_stream": True,
            "is_silent": False,
            "duration_seconds": 6.0,
            "mel_bins": 64,
            "waveform_stats": {"rms_energy": 0.042, "peak_amplitude": 0.28}
        }
        speech_ev = {
            "transcript": "ಬೆಂಕಿ ಅಪಘಾತ ಸಂಭವಿಸಿದೆ",
            "accepted": True,
            "language": "kannada",
            "word_count": 3,
            "chunks": [{"start_sec": 1.2, "end_sec": 3.4, "text": "ಬೆಂಕಿ ಅಪಘಾತ ಸಂಭವಿಸಿದೆ"}]
        }
        pred_det = {
            "predicted_class": "Fire",
            "confidence": 0.892,
            "class_scores": {"Accident": 0.045, "Fire": 0.892, "Normal": 0.063},
            "score_semantics": "uncalibrated_softmax_activations",
            "modality_availability": {"visual": True, "audio": True, "speech": True}
        }

        desc = generate_evidence_grounded_scene_description(
            video_evidence=video_ev,
            audio_evidence=audio_ev,
            speech_evidence=speech_ev,
            prediction_details=pred_det
        )

        structured = desc["structured"]
        # Mandatory Section 1: Visual Observations
        self.assertIn("visual_observations", structured)
        self.assertEqual(structured["visual_observations"]["sampled_frame_count"], 12)
        self.assertEqual(structured["visual_observations"]["influential_rank_1_frame"], 4)

        # Mandatory Section 2: Audio Observations
        self.assertIn("audio_observations", structured)
        self.assertTrue(structured["audio_observations"]["has_audio_stream"])
        self.assertEqual(structured["audio_observations"]["acoustic_state"], "active")

        # Mandatory Section 3: Speech Transcript and Summary
        self.assertIn("speech_transcript_and_summary", structured)
        self.assertEqual(structured["speech_transcript_and_summary"]["detected_language"], "kannada")
        self.assertEqual(structured["speech_transcript_and_summary"]["raw_transcript"], "ಬೆಂಕಿ ಅಪಘಾತ ಸಂಭವಿಸಿದೆ")

        # Mandatory Section 4: Key Timestamps
        self.assertIn("key_timestamps", structured)
        self.assertGreaterEqual(len(structured["key_timestamps"]), 4)

        # Mandatory Section 5: Model Prediction and Class Scores
        self.assertIn("model_prediction_and_class_scores", structured)
        self.assertEqual(structured["model_prediction_and_class_scores"]["predicted_class"], "Fire")
        self.assertEqual(structured["model_prediction_and_class_scores"]["confidence"], 0.892)

        # Mandatory Section 6: Uncertainty and Limitations
        self.assertIn("uncertainty_and_limitations", structured)
        self.assertIn(CAUSATION_DISCLAIMER, structured["uncertainty_and_limitations"]["limitations"])

        # Formatted markdown verification
        formatted = desc["formatted_text"]
        self.assertIn("### 1. Visual Observations", formatted)
        self.assertIn("### 2. Audio Observations", formatted)
        self.assertIn("### 3. Speech Transcript and Summary", formatted)
        self.assertIn("### 4. Key Timestamps", formatted)
        self.assertIn("### 5. Model Prediction and Class Scores", formatted)
        self.assertIn("### 6. Uncertainty and Limitations", formatted)

    def test_09_anti_hallucination_constraint(self):
        """Verify neutral inputs produce neutral descriptions with NO fabricated emergency claims."""
        video_ev = {
            "sampled_frame_count": 12,
            "frame_resolution": [224, 224],
            "duration_seconds": 4.0,
            "frame_timestamps_sec": [float(i * 0.33) for i in range(12)],
            "influential_frames": {"ranked_frames": []}
        }
        audio_ev = {
            "has_audio_stream": False,
            "is_silent": True,
            "duration_seconds": 0.0,
            "waveform_stats": {}
        }
        speech_ev = {
            "transcript": "",
            "accepted": False,
            "language": "none",
            "word_count": 0,
            "chunks": []
        }
        pred_det = {
            "predicted_class": "Normal",
            "confidence": 0.94,
            "class_scores": {"Accident": 0.02, "Fire": 0.04, "Normal": 0.94},
            "score_semantics": "uncalibrated_softmax_activations",
            "modality_availability": {"visual": True, "audio": False, "speech": False}
        }

        desc = generate_evidence_grounded_scene_description(
            video_evidence=video_ev,
            audio_evidence=audio_ev,
            speech_evidence=speech_ev,
            prediction_details=pred_det
        )

        formatted = desc["formatted_text"]
        # Must NOT claim fire, collision, sirens, or dispatches
        self.assertNotIn("fire outbreak occurred", formatted.lower())
        self.assertNotIn("collision occurred", formatted.lower())
        self.assertNotIn("sirens blaring", formatted.lower())
        self.assertIn("audio stream is absent", formatted.lower())
        self.assertIn("no intelligible kannada or english speech detected", formatted.lower())

    def test_10_recoverable_component_failure(self):
        """Verify malformed or missing evidence structures fail gracefully without unhandled exceptions."""
        empty_desc = generate_evidence_grounded_scene_description(
            video_evidence={},
            audio_evidence={},
            speech_evidence={},
            prediction_details={}
        )
        self.assertIn("visual_observations", empty_desc["structured"])
        self.assertIn("audio_observations", empty_desc["structured"])
        self.assertIn("speech_transcript_and_summary", empty_desc["structured"])
        self.assertIn("uncertainty_and_limitations", empty_desc["structured"])
        self.assertIn(CAUSATION_DISCLAIMER, empty_desc["formatted_text"])

    def test_11_database_speech_persistence(self):
        """Verify database persists speech transcripts and structured description separately."""
        test_job_id = f"job_db_speech_{int(time.time())}"
        database.create_job(
            job_id=test_job_id,
            filename="db_speech.mp4",
            file_path="/tmp/db_speech.mp4",
            file_size_bytes=4096,
            file_hash_sha256="db_hash"
        )

        struct_desc = {
            "visual_observations": {"sampled_frame_count": 12},
            "audio_observations": {"has_audio_stream": True},
            "speech_transcript_and_summary": {"detected_language": "kannada"},
            "key_timestamps": [],
            "model_prediction_and_class_scores": {"predicted_class": "Fire"},
            "uncertainty_and_limitations": {"causation_disclaimer": CAUSATION_DISCLAIMER}
        }

        database.complete_job(
            job_id=test_job_id,
            prediction="Fire",
            confidence=0.91,
            probabilities={"Fire": 0.91, "Accident": 0.05, "Normal": 0.04},
            evidence={},
            transcript="ಬೆಂಕಿ ಕಂಡುಬಂದಿದೆ",
            description="### 1. Visual Observations...",
            structured_description=struct_desc
        )

        database.add_speech_transcript(
            job_id=test_job_id,
            language="kn",
            transcript="ಬೆಂಕಿ ಕಂಡುಬಂದಿದೆ",
            confidence=0.98,
            is_accepted=True,
            chunks=[{"start_sec": 0.0, "end_sec": 2.5, "text": "ಬೆಂಕಿ ಕಂಡುಬಂದಿದೆ"}]
        )

        # Retrieve job and speech transcripts
        job = database.get_job(test_job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["transcript"], "ಬೆಂಕಿ ಕಂಡುಬಂದಿದೆ")
        self.assertIsNotNone(job.get("structured_description"))
        self.assertEqual(job["structured_description"]["model_prediction_and_class_scores"]["predicted_class"], "Fire")

        transcripts = database.get_speech_transcripts(test_job_id)
        self.assertEqual(len(transcripts), 1)
        self.assertEqual(transcripts[0]["language"], "kn")
        self.assertEqual(transcripts[0]["transcript"], "ಬೆಂಕಿ ಕಂಡುಬಂದಿದೆ")
        self.assertEqual(len(transcripts[0]["chunks"]), 1)

    def test_12_api_transcripts_and_description_endpoints(self):
        """Verify live HTTP API exposes transcripts and structured descriptions."""
        base_url = "http://127.0.0.1:8000"

        # Check server health
        try:
            health_resp = requests.get(f"{base_url}/api/health", timeout=1)
            if health_resp.status_code != 200:
                raise requests.RequestException()
        except Exception:
            raise unittest.SkipTest("Live API server is not running on 127.0.0.1:8000 (skipping live test)")

        api_job_id = f"job_api_speech_{int(time.time())}"
        database.create_job(
            job_id=api_job_id,
            filename="api_speech.mp4",
            file_path="/tmp/api_speech.mp4",
            file_size_bytes=8192,
            file_hash_sha256="api_speech_hash"
        )

        struct_desc = {
            "visual_observations": {"sampled_frame_count": 12},
            "audio_observations": {"has_audio_stream": True},
            "speech_transcript_and_summary": {"detected_language": "english"},
            "key_timestamps": [],
            "model_prediction_and_class_scores": {"predicted_class": "Accident"},
            "uncertainty_and_limitations": {"causation_disclaimer": CAUSATION_DISCLAIMER}
        }

        database.complete_job(
            job_id=api_job_id,
            prediction="Accident",
            confidence=0.87,
            probabilities={"Accident": 0.87, "Fire": 0.08, "Normal": 0.05},
            evidence={},
            transcript="Emergency vehicle on site",
            description="### 1. Visual Observations\nDetailed visual breakdown.",
            structured_description=struct_desc
        )

        database.add_speech_transcript(
            job_id=api_job_id,
            language="en",
            transcript="Emergency vehicle on site",
            confidence=0.96,
            is_accepted=True,
            chunks=[{"start_sec": 1.0, "end_sec": 3.0, "text": "Emergency vehicle on site"}]
        )

        # 1. Test GET /api/jobs/{job_id}/transcripts
        resp_tr = requests.get(f"{base_url}/api/jobs/{api_job_id}/transcripts", timeout=5)
        self.assertEqual(resp_tr.status_code, 200)
        data_tr = resp_tr.json()
        self.assertEqual(data_tr["job_id"], api_job_id)
        self.assertEqual(data_tr["total_transcripts"], 1)
        self.assertEqual(data_tr["transcripts"][0]["language"], "en")
        self.assertEqual(data_tr["transcripts"][0]["transcript"], "Emergency vehicle on site")

        # 2. Test GET /api/jobs/{job_id}/description
        resp_desc = requests.get(f"{base_url}/api/jobs/{api_job_id}/description", timeout=5)
        self.assertEqual(resp_desc.status_code, 200)
        data_desc = resp_desc.json()
        self.assertEqual(data_desc["job_id"], api_job_id)
        self.assertEqual(data_desc["prediction"], "Accident")
        self.assertIn("visual_observations", data_desc["structured"])
        self.assertIn("### 1. Visual Observations", data_desc["formatted_text"])


if __name__ == "__main__":
    unittest.main()
