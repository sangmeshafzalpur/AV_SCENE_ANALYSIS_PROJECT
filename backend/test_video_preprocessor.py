import os
import unittest
import tempfile
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import tensorflow as tf
from transformers import AutoImageProcessor, AutoModel

from services.video_preprocessor import (
    NUM_FRAMES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    VISUAL_FEATURE_DIM,
    validate_video_file_detailed,
    get_uniform_frame_indices,
    extract_video_frames_with_metadata,
    compute_visual_embeddings,
    generate_video_thumbnails_and_contact_sheet
)
import database

def create_synthetic_test_video(
    filepath: Path,
    num_frames: int = 60,
    fps: float = 30.0,
    width: int = 320,
    height: int = 240
) -> Path:
    """Helper to synthesize genuine test MP4 videos with distinct visual patterns."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
    
    for i in range(num_frames):
        # Create distinct recognizable colored frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Gradient background
        color_val = int((i / max(1, num_frames - 1)) * 200) + 20
        frame[:, :] = (color_val, 100, 255 - color_val)
        
        # Burn in frame number text
        text = f"Frame {i:03d} - {round(i / fps, 2)}s"
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        
    writer.release()
    return filepath

class TestVideoPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        
        # Load visual model for tensor contract verification
        cls.visual_processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")
        cls.visual_model = AutoModel.from_pretrained(
            "facebook/convnextv2-tiny-1k-224",
            torch_dtype=torch.float32
        ).to("cpu").eval()
        
        # Load Keras Multimodal Model
        backend_dir = Path(__file__).resolve().parent
        model_path = backend_dir.parent / "kannada_av_scene_model.keras"
        if not model_path.exists():
            model_path = Path(r"c:\Users\Hp\OneDrive\Desktop\hphphphphhphph\kannada_av_scene_model.keras")
        cls.keras_model = tf.keras.models.load_model(str(model_path))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_invalid_and_malformed_video_rejection(self):
        """Verify strict pre-flight validation on missing, invalid, and corrupt files."""
        # 1. Non-existent file
        val_missing = validate_video_file_detailed(self.temp_dir / "non_existent.mp4")
        self.assertFalse(val_missing["is_valid"])
        self.assertIn("not found", val_missing["error"])

        # 2. Unsupported extension (.txt)
        txt_path = self.temp_dir / "test.txt"
        with open(txt_path, "w") as f:
            f.write("Not a video")
        val_ext = validate_video_file_detailed(txt_path)
        self.assertFalse(val_ext["is_valid"])
        self.assertIn("Unsupported video format", val_ext["error"])

        # 3. Empty / truncated file
        empty_mp4 = self.temp_dir / "empty.mp4"
        with open(empty_mp4, "wb") as f:
            f.write(b"")
        val_empty = validate_video_file_detailed(empty_mp4)
        self.assertFalse(val_empty["is_valid"])
        self.assertIn("too small or empty", val_empty["error"])

        # 4. Corrupt header file
        corrupt_mp4 = self.temp_dir / "corrupt.mp4"
        with open(corrupt_mp4, "wb") as f:
            f.write(b"\x00" * 4096)
        val_corrupt = validate_video_file_detailed(corrupt_mp4)
        self.assertFalse(val_corrupt["is_valid"])
        self.assertIn("Corrupt or unreadable", val_corrupt["error"])

    def test_02_standard_video_uniform_sampling_and_timestamps(self):
        """Verify uniform 12-frame sampling, timestamp preservation, and resizing."""
        vid_path = self.temp_dir / "standard_30fps.mp4"
        create_synthetic_test_video(vid_path, num_frames=60, fps=30.0)

        result = extract_video_frames_with_metadata(vid_path, num_frames=12)
        frames = result["frames"]
        timestamps = result["frame_timestamps_sec"]
        indices = result["sampled_indices"]

        # Dimensions & type checks
        self.assertEqual(frames.shape, (12, 224, 224, 3))
        self.assertEqual(frames.dtype, np.uint8)
        self.assertEqual(result["color_format"], "RGB")

        # Temporal sampling checks
        expected_indices = get_uniform_frame_indices(60, 12).tolist()
        self.assertEqual(indices, expected_indices)
        self.assertEqual(len(timestamps), 12)

        # Monotonicity of timestamps
        for i in range(len(timestamps) - 1):
            self.assertLessEqual(timestamps[i], timestamps[i + 1])
        self.assertAlmostEqual(timestamps[-1], 59 / 30.0, places=1)

    def test_03_various_frame_rates_handling(self):
        """Verify extraction across different FPS rates (15, 24, 60 FPS)."""
        fps_list = [15.0, 24.0, 60.0]
        for fps in fps_list:
            vpath = self.temp_dir / f"test_{int(fps)}fps.mp4"
            create_synthetic_test_video(vpath, num_frames=int(fps * 2), fps=fps)

            res = extract_video_frames_with_metadata(vpath, num_frames=12)
            self.assertEqual(res["frames"].shape, (12, 224, 224, 3))
            self.assertEqual(len(res["frame_timestamps_sec"]), 12)
            self.assertAlmostEqual(res["fps"], fps, delta=1.5)

    def test_04_short_video_without_fabrication(self):
        """Verify verified sampling policy on short videos (< 12 frames) without synthetic frames."""
        vpath = self.temp_dir / "short_5frames.mp4"
        create_synthetic_test_video(vpath, num_frames=5, fps=10.0)

        res = extract_video_frames_with_metadata(vpath, num_frames=12)
        frames = res["frames"]
        indices = res["sampled_indices"]

        # Exact 12 frames output
        self.assertEqual(frames.shape, (12, 224, 224, 3))
        self.assertEqual(len(indices), 12)

        # Verified mathematical rounding policy check
        expected_indices = np.round(np.linspace(0, 4, 12)).astype(int).tolist()
        self.assertEqual(indices, expected_indices)

        # All indices must be within genuine range [0, 4]
        for idx in indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 5)

        # Warning should be recorded
        self.assertTrue(any("short" in w.lower() for w in res["warnings"]))

    def test_05_single_frame_edge_case(self):
        """Verify edge case of a 1-frame video."""
        vpath = self.temp_dir / "single_frame.mp4"
        create_synthetic_test_video(vpath, num_frames=1, fps=1.0)

        res = extract_video_frames_with_metadata(vpath, num_frames=12)
        self.assertEqual(res["frames"].shape, (12, 224, 224, 3))
        self.assertEqual(res["sampled_indices"], [0] * 12)

    def test_06_thumbnails_and_contact_sheet_artifacts(self):
        """Verify generation of individual frame JPEGs, primary thumbnail, and contact sheet filmstrip."""
        vpath = self.temp_dir / "thumb_test.mp4"
        create_synthetic_test_video(vpath, num_frames=36, fps=18.0)

        res = extract_video_frames_with_metadata(vpath, num_frames=12)
        job_id = f"job_test_thumb_{uuid.uuid4().hex[:8]}"
        
        # Ensure job exists in DB for foreign key
        database.create_job(job_id=job_id, filename="thumb_test.mp4", file_path="uploads/test.mp4", file_size_bytes=1000)

        thumb_info = generate_video_thumbnails_and_contact_sheet(
            job_id=job_id,
            frames_rgb=res["frames"],
            timestamps_sec=res["frame_timestamps_sec"]
        )

        self.assertIn("thumbnail_url", thumb_info)
        self.assertIn("filmstrip_url", thumb_info)
        self.assertEqual(len(thumb_info["frames"]), 12)

        # Verify artifacts exist on disk
        thumb_art = database.get_artifact(thumb_info["thumbnail_artifact_id"])
        self.assertIsNotNone(thumb_art)

        filmstrip_art = database.get_artifact(thumb_info["filmstrip_artifact_id"])
        self.assertIsNotNone(filmstrip_art)

    def test_07_convnext_embedding_tensor_contract_with_keras_model(self):
        """Verify that preprocessor produces (1, 12, 768) float32 tensor matching Keras model input."""
        vpath = self.temp_dir / "tensor_contract.mp4"
        create_synthetic_test_video(vpath, num_frames=30, fps=15.0)

        res = extract_video_frames_with_metadata(vpath, num_frames=12)
        frames = res["frames"]

        visual_tensor, proc_time = compute_visual_embeddings(
            frames_rgb=frames,
            visual_processor=self.visual_processor,
            visual_model=self.visual_model,
            device=torch.device("cpu")
        )

        # 1. Exact shape & dtype contract
        self.assertEqual(visual_tensor.shape, (1, 12, 768))
        self.assertEqual(visual_tensor.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(visual_tensor)), "Visual embedding tensor must contain finite numbers")
        self.assertGreater(proc_time, 0.0)

        # 2. Feed directly into Keras Multimodal Model along with dummy audio and speech
        dummy_audio = np.zeros((1, 64, 313, 1), dtype=np.float32)
        dummy_speech = np.zeros((1, 384), dtype=np.float32)

        predictions = self.keras_model.predict([visual_tensor, dummy_audio, dummy_speech], verbose=0)
        
        # 3. Verify prediction contract
        self.assertEqual(predictions.shape, (1, 3))
        self.assertAlmostEqual(float(np.sum(predictions[0])), 1.0, places=4)
        for prob in predictions[0]:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

if __name__ == "__main__":
    unittest.main()
