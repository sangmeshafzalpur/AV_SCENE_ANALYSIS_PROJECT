"""
Comprehensive Test Suite for Evidence-Based Influential Frame Attribution
========================================================================
Tests:
  1. Frame perturbation strategies (neutral_gray, temporal_mean, adjacent_interpolation, zero_mask).
  2. Baseline score, perturbed score, and non-arbitrary score difference calculation.
  3. Ranking logic, positive-drop normalization, and keyframe designation.
  4. Deterministic reproducibility across repeated evaluations.
  5. Exact 12-frame-to-timestamp mapping integrity.
  6. Causation disclaimer and diffuse/unreliable sensitivity detection.
  7. Valid gradient-based temporal saliency method execution.
  8. Failure and malformed input handling.
  9. Database persistence, migration schema, and rank/time query sorting.
  10. End-to-end API integration (/predict and GET /api/jobs/{job_id}/attributions).
"""

import os
import sys
import time
import unittest
import numpy as np
import requests

# Reconfigure stdout/stderr for clean utf-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import database
from services.frame_attribution_service import (
    perturb_frame,
    compute_evidence_based_frame_attributions,
    compute_gradient_frame_attributions,
    CAUSATION_DISCLAIMER,
    SUPPORTED_STRATEGIES
)
from services.model_inference_service import (
    CLASS_NAMES,
    EXPECTED_VIDEO_SHAPE,
    EXPECTED_AUDIO_SHAPE,
    run_direct_multimodal_inference
)


class TestInfluentialFrameAttribution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create deterministic synthetic video frames (12, 224, 224, 3)
        np.random.seed(42)
        cls.dummy_video = np.random.uniform(20.0, 230.0, size=(12, 224, 224, 3)).astype(np.float32)
        # Create distinct features in frame 4 (e.g. bright red flame-like patch)
        cls.dummy_video[4, 80:150, 80:150, 0] = 255.0  # High red channel
        cls.dummy_video[4, 80:150, 80:150, 1:] = 10.0   # Low green/blue

        cls.dummy_timestamps = [float(i * 0.75) for i in range(12)]
        cls.dummy_audio = np.random.uniform(-40.0, 0.0, size=EXPECTED_AUDIO_SHAPE).astype(np.float32)
        cls.dummy_transcript = "ಬೆಂಕಿ ಅಪಘಾತ ಸಂಭವಿಸಿದೆ"  # Fire accident occurred

    def test_01_perturbation_strategies(self):
        """Verify all perturbation strategies modify ONLY the target frame correctly."""
        for strategy in SUPPORTED_STRATEGIES:
            target_idx = 4
            perturbed = perturb_frame(self.dummy_video, frame_index=target_idx, strategy=strategy)
            
            # Check shape unchanged
            self.assertEqual(perturbed.shape, EXPECTED_VIDEO_SHAPE, f"Shape mismatch for {strategy}")

            # Verify all other 11 frames are strictly identical
            for k in range(12):
                if k != target_idx:
                    np.testing.assert_array_equal(
                        perturbed[k],
                        self.dummy_video[k],
                        err_msg=f"Unperturbed frame {k} was altered during strategy {strategy}"
                    )

            # Verify target frame modification
            if strategy == "neutral_gray":
                self.assertTrue(np.all(perturbed[target_idx] == 128.0))
            elif strategy == "zero_mask":
                self.assertTrue(np.all(perturbed[target_idx] == 0.0))
            elif strategy == "temporal_mean":
                expected_mean = np.mean(self.dummy_video, axis=0)
                np.testing.assert_allclose(perturbed[target_idx], expected_mean, rtol=1e-5)
            elif strategy == "adjacent_interpolation":
                expected_interp = (self.dummy_video[3] + self.dummy_video[5]) / 2.0
                np.testing.assert_allclose(perturbed[target_idx], expected_interp, rtol=1e-5)

        # Test boundary cases for adjacent_interpolation (index 0 and 11)
        p0 = perturb_frame(self.dummy_video, 0, strategy="adjacent_interpolation")
        np.testing.assert_array_equal(p0[0], self.dummy_video[1])
        p11 = perturb_frame(self.dummy_video, 11, strategy="adjacent_interpolation")
        np.testing.assert_array_equal(p11[11], self.dummy_video[10])

    def test_02_score_calculation_and_deltas(self):
        """Verify target class baseline score, perturbed scores, and deltas match model outputs."""
        res = compute_evidence_based_frame_attributions(
            video_frames=self.dummy_video,
            frame_timestamps_sec=self.dummy_timestamps,
            audio_spectrogram=self.dummy_audio,
            audio_available=True,
            transcript=self.dummy_transcript,
            perturbation_strategy="neutral_gray"
        )

        self.assertIn("target_class", res)
        self.assertIn("baseline_score", res)
        self.assertEqual(res["total_frames_evaluated"], 12)
        self.assertEqual(len(res["frames"]), 12)

        base_score = res["baseline_score"]
        self.assertGreater(base_score, 0.0)
        self.assertLessEqual(base_score, 1.0)

        # Check each frame delta calculation
        for f in res["frames"]:
            expected_diff = round(f["baseline_score"] - f["perturbed_score"], 6)
            self.assertAlmostEqual(f["score_difference"], expected_diff, places=5)
            self.assertEqual(f["baseline_score"], base_score)

    def test_03_ranking_and_normalization_logic(self):
        """Verify ranking is strictly sorted by descending score difference and normalized properly."""
        res = compute_evidence_based_frame_attributions(
            video_frames=self.dummy_video,
            frame_timestamps_sec=self.dummy_timestamps,
            audio_spectrogram=self.dummy_audio,
            audio_available=True,
            transcript=self.dummy_transcript,
            top_k_keyframes=3
        )

        frames = res["frames"]
        ranks = [f["rank"] for f in frames]
        # Ranks must be a permutation of 1..12
        self.assertEqual(sorted(ranks), list(range(1, 13)))

        # Verify rank 1 has max score_difference
        rank_1_frame = next(f for f in frames if f["rank"] == 1)
        max_diff = max(f["score_difference"] for f in frames)
        self.assertEqual(rank_1_frame["score_difference"], max_diff)

        # Verify keyframes
        keyframes = [f for f in frames if f["is_keyframe"]]
        self.assertEqual(len(keyframes), 3)
        self.assertTrue(all(f["rank"] <= 3 for f in keyframes))

        # Check relative importance normalization
        total_importance = sum(f["attribution_score"] for f in frames)
        self.assertAlmostEqual(total_importance, 1.0, places=3)

    def test_04_reproducibility(self):
        """Verify identical inputs yield strictly identical scores, rankings, and attributions."""
        run1 = compute_evidence_based_frame_attributions(
            video_frames=self.dummy_video,
            frame_timestamps_sec=self.dummy_timestamps,
            audio_spectrogram=self.dummy_audio,
            audio_available=True,
            transcript=self.dummy_transcript
        )
        run2 = compute_evidence_based_frame_attributions(
            video_frames=self.dummy_video,
            frame_timestamps_sec=self.dummy_timestamps,
            audio_spectrogram=self.dummy_audio,
            audio_available=True,
            transcript=self.dummy_transcript
        )

        self.assertEqual(run1["target_class"], run2["target_class"])
        self.assertEqual(run1["baseline_score"], run2["baseline_score"])
        for f1, f2 in zip(run1["frames"], run2["frames"]):
            self.assertEqual(f1["frame_index"], f2["frame_index"])
            self.assertEqual(f1["score_difference"], f2["score_difference"])
            self.assertEqual(f1["attribution_score"], f2["attribution_score"])
            self.assertEqual(f1["rank"], f2["rank"])
            self.assertEqual(f1["is_keyframe"], f2["is_keyframe"])

    def test_05_frame_to_timestamp_mapping(self):
        """Verify all 12 frames maintain exact 1-to-1 mapping with video timestamps."""
        custom_timestamps = [0.0, 1.25, 2.50, 3.75, 5.0, 6.25, 7.5, 8.75, 10.0, 11.25, 12.5, 13.75]
        res = compute_evidence_based_frame_attributions(
            video_frames=self.dummy_video,
            frame_timestamps_sec=custom_timestamps,
            audio_spectrogram=self.dummy_audio,
            audio_available=True,
            transcript=self.dummy_transcript
        )

        for i, f in enumerate(res["frames"]):
            self.assertEqual(f["frame_index"], i)
            self.assertEqual(f["timestamp_sec"], custom_timestamps[i])

    def test_06_causation_disclaimer_and_reliability(self):
        """Verify causation disclaimer is explicitly present and sensitivity reliability is assessed."""
        res = compute_evidence_based_frame_attributions(
            video_frames=self.dummy_video,
            frame_timestamps_sec=self.dummy_timestamps,
            audio_spectrogram=self.dummy_audio,
            audio_available=True,
            transcript=self.dummy_transcript
        )

        self.assertIn("causation_disclaimer", res)
        self.assertEqual(res["causation_disclaimer"], CAUSATION_DISCLAIMER)
        self.assertIn("reliability", res)
        self.assertIn(res["reliability"], ["reliable_sensitivity", "diffuse_or_modality_dominated"])

        # Test diffuse/flat sensitivity detection when score changes are zero
        flat_video = np.full((12, 224, 224, 3), 128.0, dtype=np.float32)
        flat_res = compute_evidence_based_frame_attributions(
            video_frames=flat_video,
            frame_timestamps_sec=self.dummy_timestamps,
            audio_spectrogram=None,
            audio_available=False,
            transcript=None,
            perturbation_strategy="neutral_gray"  # Replacing 128 with 128 causes exactly zero change!
        )
        self.assertEqual(flat_res["reliability"], "diffuse_or_modality_dominated")
        self.assertTrue(any("Low visual attribution sensitivity" in w or "score drops were <= 0" in w for w in flat_res["warnings"]))

    def test_07_gradient_attribution_method(self):
        """Verify gradient-based temporal saliency method computes valid Frobenius norm gradients."""
        grad_res = compute_gradient_frame_attributions(
            video_frames=self.dummy_video,
            frame_timestamps_sec=self.dummy_timestamps,
            audio_spectrogram=self.dummy_audio,
            audio_available=True,
            transcript=self.dummy_transcript,
            top_k_keyframes=3
        )

        self.assertEqual(grad_res["attribution_method"], "gradient_temporal_saliency (Frobenius norm)")
        self.assertEqual(len(grad_res["frames"]), 12)
        self.assertIn("causation_disclaimer", grad_res)

        # Verify gradient norms are non-negative and non-trivial
        for f in grad_res["frames"]:
            self.assertGreaterEqual(f["gradient_frobenius_norm"], 0.0)
            self.assertGreaterEqual(f["attribution_score"], 0.0)
            self.assertIn("rank", f)

        # Check ranks
        ranks = [f["rank"] for f in grad_res["frames"]]
        self.assertEqual(sorted(ranks), list(range(1, 13)))

    def test_08_failure_handling(self):
        """Verify robust error handling for invalid or malformed inputs."""
        # 1. Invalid video shape (wrong number of frames)
        bad_video = np.zeros((10, 224, 224, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            compute_evidence_based_frame_attributions(
                video_frames=bad_video,
                frame_timestamps_sec=self.dummy_timestamps
            )

        # 2. Invalid timestamps length
        with self.assertRaises(ValueError):
            compute_evidence_based_frame_attributions(
                video_frames=self.dummy_video,
                frame_timestamps_sec=[1.0, 2.0]  # Only 2 timestamps
            )

        # 3. Invalid strategy
        with self.assertRaises(ValueError):
            perturb_frame(self.dummy_video, 0, strategy="magic_wand")

        # 4. Invalid target class
        with self.assertRaises(ValueError):
            compute_evidence_based_frame_attributions(
                video_frames=self.dummy_video,
                frame_timestamps_sec=self.dummy_timestamps,
                target_class="Tornado"
            )

    def test_09_database_persistence_and_queries(self):
        """Verify enhanced frame attribution database schema, insertion, and rank/time queries."""
        test_job_id = f"job_test_attr_{int(time.time())}"
        database.create_job(
            job_id=test_job_id,
            filename="attr_test.mp4",
            file_path="/tmp/attr_test.mp4",
            file_size_bytes=1024,
            file_hash_sha256="test_hash"
        )

        # Insert 12 frame attributions with ranks and score deltas
        for i in range(12):
            score_diff = round((11 - i) * 0.05, 4)  # Frame 0 has highest diff, Frame 11 lowest
            database.add_frame_attribution(
                job_id=test_job_id,
                frame_index=i,
                timestamp_sec=float(i * 0.8),
                attribution_score=round(1.0 / 12.0, 4),
                baseline_score=0.92,
                perturbed_score=round(0.92 - score_diff, 4),
                score_difference=score_diff,
                attribution_method="temporal_frame_ablation (neutral_gray)",
                rank=i + 1,
                is_keyframe=(i < 3),
                label=f"Rank {i + 1} (Δ {score_diff:+.4f})"
            )

        # Query chronological order (order_by_rank=False)
        chrono_attrs = database.get_frame_attributions(test_job_id, order_by_rank=False)
        self.assertEqual(len(chrono_attrs), 12)
        for i in range(12):
            self.assertEqual(chrono_attrs[i]["frame_index"], i)
            self.assertEqual(chrono_attrs[i]["baseline_score"], 0.92)
            self.assertEqual(chrono_attrs[i]["attribution_method"], "temporal_frame_ablation (neutral_gray)")

        # Query rank order (order_by_rank=True)
        rank_attrs = database.get_frame_attributions(test_job_id, order_by_rank=True)
        self.assertEqual(len(rank_attrs), 12)
        for idx, attr in enumerate(rank_attrs):
            self.assertEqual(attr["rank"], idx + 1)

    def test_10_api_integration(self):
        """Verify backend API exposes frame attribution results with disclaimer, ranking, and sorting."""
        base_url = "http://127.0.0.1:8000"

        # Check server health
        try:
            health_resp = requests.get(f"{base_url}/api/health", timeout=1)
            if health_resp.status_code != 200:
                raise requests.RequestException()
        except Exception:
            raise unittest.SkipTest("Live API server is not running on 127.0.0.1:8000 (skipping live test)")

        # 1. Create a job with attributions in DB
        api_job_id = f"job_api_attr_{int(time.time())}"
        database.create_job(
            job_id=api_job_id,
            filename="api_test.mp4",
            file_path="/tmp/api_test.mp4",
            file_size_bytes=2048,
            file_hash_sha256="api_test_hash"
        )
        database.complete_job(
            job_id=api_job_id,
            prediction="Fire",
            confidence=0.88,
            probabilities={"Accident": 0.05, "Fire": 0.88, "Normal": 0.07},
            evidence={
                "visual_evidence": {
                    "influential_frames": {
                        "attribution_method": "temporal_frame_ablation (neutral_gray)",
                        "target_class": "Fire",
                        "baseline_score": 0.88,
                        "causation_disclaimer": CAUSATION_DISCLAIMER,
                        "reliability": "reliable_sensitivity",
                        "score_range": 0.25
                    }
                }
            },
            model_identifier="best_multimodal_model.keras",
            model_checksum="404f374484b2304e44074dd44a81a8130add1b22c22dcd7a7e21ea56f8763fde"
        )

        for i in range(12):
            diff = round(0.20 - (i * 0.015), 4)
            database.add_frame_attribution(
                job_id=api_job_id,
                frame_index=i,
                timestamp_sec=float(i * 1.0),
                attribution_score=0.08,
                baseline_score=0.88,
                perturbed_score=round(0.88 - diff, 4),
                score_difference=diff,
                attribution_method="temporal_frame_ablation (neutral_gray)",
                rank=i + 1,
                is_keyframe=(i < 3)
            )

        # 2. Test GET /api/jobs/{job_id}/attributions?sort_by=rank
        resp_rank = requests.get(f"{base_url}/api/jobs/{api_job_id}/attributions?sort_by=rank", timeout=5)
        self.assertEqual(resp_rank.status_code, 200)
        data_rank = resp_rank.json()

        self.assertEqual(data_rank["job_id"], api_job_id)
        self.assertEqual(data_rank["target_class"], "Fire")
        self.assertEqual(data_rank["baseline_score"], 0.88)
        self.assertEqual(data_rank["causation_disclaimer"], CAUSATION_DISCLAIMER)
        self.assertEqual(data_rank["total_frames_attributed"], 12)
        self.assertEqual(data_rank["sort_by"], "rank")
        self.assertEqual([a["rank"] for a in data_rank["attributions"]], list(range(1, 13)))

        # 3. Test GET /api/jobs/{job_id}/attributions?sort_by=time
        resp_time = requests.get(f"{base_url}/api/jobs/{api_job_id}/attributions?sort_by=time", timeout=5)
        self.assertEqual(resp_time.status_code, 200)
        data_time = resp_time.json()
        self.assertEqual(data_time["sort_by"], "time")
        self.assertEqual([a["frame_index"] for a in data_time["attributions"]], list(range(12)))


if __name__ == "__main__":
    unittest.main()
