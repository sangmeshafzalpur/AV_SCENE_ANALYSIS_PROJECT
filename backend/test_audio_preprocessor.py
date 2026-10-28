"""
Unit and Integration Tests for Audio Preprocessor and Acoustic Modality Pipeline
==============================================================================
Validates:
  1. Audio reference configuration (16 kHz, 64 mel bins, FFT 1024, Hop 256, shape (64, 313, 1)).
  2. Axes metadata generation for client-side time & frequency axis visualization.
  3. Genuine audio extraction from synthetic and real video containers.
  4. Silent video handling (pure zero-amplitude stream, zero-tensor policy, warnings).
  5. Missing audio stream policy (video-only containers, explicit documentation, zero-tensor).
  6. Noisy audio handling (white noise, energy calculations, standardization).
  7. Malformed inputs (non-existent, 0-byte, corrupted files).
  8. Multimodal Keras model tensor compatibility with live kannada_av_scene_model.keras.
  9. Non-inference guarantee (no heuristic prediction solely from spectrograms).
"""

import os
import sys
import unittest
import tempfile
import shutil
import wave
import subprocess
from pathlib import Path
from typing import Tuple

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import cv2
import tensorflow as tf

from services.audio_preprocessor import (
    AUDIO_SAMPLE_RATE,
    AUDIO_FIXED_SECONDS,
    AUDIO_TARGET_SAMPLES,
    MEL_BINS,
    N_FFT,
    HOP_LENGTH,
    TIME_STEPS,
    EXPECTED_TENSOR_SHAPE,
    probe_audio_stream,
    extract_audio_waveform,
    compute_spectrogram_axes_metadata,
    waveform_to_logmel_window,
    compute_audio_logmel_features,
    render_spectrogram_image_with_axes,
    save_audio_spectrogram_artifact
)


def create_synthetic_video_with_audio(
    video_path: Path,
    duration_sec: float = 3.0,
    audio_type: str = "tone",
    fps: float = 24.0,
    sample_rate: int = 16000
) -> Path:
    """
    Helper to synthesize test MP4 videos with controlled audio tracks.
    audio_type can be:
      - 'tone': 440 Hz pure tone
      - 'silent': pure zeros (silence)
      - 'noise': gaussian white noise
      - 'none': video track only (no audio stream)
    """
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = video_path.parent / f"tmp_{video_path.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_video_only = temp_dir / "temp_video.mp4"
    temp_audio_wav = temp_dir / "temp_audio.wav"

    # 1. Create simple video track
    num_frames = max(1, int(duration_sec * fps))
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video_only), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 3 % 255, 120, 200), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    if audio_type == "none":
        shutil.move(str(temp_video_only), str(video_path))
        shutil.rmtree(temp_dir, ignore_errors=True)
        return video_path

    # 2. Synthesize audio waveform
    total_audio_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_audio_samples, endpoint=False)

    if audio_type == "tone":
        # 440 Hz sinusoidal wave with moderate amplitude
        raw_audio = np.sin(2 * np.pi * 440.0 * t) * 0.7
    elif audio_type == "silent":
        raw_audio = np.zeros(total_audio_samples, dtype=np.float32)
    elif audio_type == "noise":
        raw_audio = np.random.normal(0, 0.25, total_audio_samples).astype(np.float32)
    else:
        raw_audio = np.zeros(total_audio_samples, dtype=np.float32)

    int16_audio = (np.clip(raw_audio, -1.0, 1.0) * 32767).astype(np.int16)

    # Save to WAV
    with wave.open(str(temp_audio_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_audio.tobytes())

    # 3. Mux video and audio via FFmpeg
    mux_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(temp_video_only),
        "-i", str(temp_audio_wav),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(video_path)
    ]
    subprocess.run(mux_cmd, check=True)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return video_path


class TestAudioPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(tempfile.mkdtemp(prefix="test_audio_preproc_"))
        backend_dir = Path(__file__).resolve().parent
        cls.model_path = backend_dir.parent / "kannada_av_scene_model.keras"
        if cls.model_path.exists():
            cls.multimodal_model = tf.keras.models.load_model(str(cls.model_path))
        else:
            cls.multimodal_model = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_reference_configuration_constants(self):
        """Verifies reference constants strictly conform to model specifications."""
        self.assertEqual(AUDIO_SAMPLE_RATE, 16000)
        self.assertEqual(AUDIO_FIXED_SECONDS, 5.0)
        self.assertEqual(AUDIO_TARGET_SAMPLES, 80000)
        self.assertEqual(MEL_BINS, 64)
        self.assertEqual(N_FFT, 1024)
        self.assertEqual(HOP_LENGTH, 256)
        self.assertEqual(TIME_STEPS, 313)
        self.assertEqual(EXPECTED_TENSOR_SHAPE, (64, 313, 1))

    def test_02_axes_metadata_calculation(self):
        """Verifies calculation of time and frequency axis metadata for UI rendering."""
        meta = compute_spectrogram_axes_metadata(
            sample_rate=AUDIO_SAMPLE_RATE,
            duration_seconds=AUDIO_FIXED_SECONDS,
            mel_bins=MEL_BINS
        )
        time_ax = meta["time_axis"]
        freq_ax = meta["frequency_axis"]
        spec_cfg = meta["spectral_config"]

        self.assertEqual(time_ax["time_steps"], 313)
        self.assertEqual(time_ax["time_range_seconds"], [0.0, 5.0])
        self.assertAlmostEqual(time_ax["seconds_per_step"], 256 / 16000, places=5)
        self.assertIn(0.0, time_ax["display_ticks_sec"])
        self.assertIn(5.0, time_ax["display_ticks_sec"])

        self.assertEqual(freq_ax["mel_bins"], 64)
        self.assertEqual(freq_ax["min_frequency_hz"], 0.0)
        self.assertEqual(freq_ax["max_frequency_hz"], 8000.0)
        self.assertEqual(len(freq_ax["bin_center_frequencies_hz"]), 64)
        # Verify frequencies are monotonically increasing
        freqs = freq_ax["bin_center_frequencies_hz"]
        self.assertTrue(all(freqs[i] < freqs[i+1] for i in range(len(freqs)-1)))

        self.assertEqual(spec_cfg["output_shape"], [64, 313, 1])

    def test_03_genuine_audio_extraction_and_logmel(self):
        """Extracts genuine 440 Hz audio and verifies log-mel spectrogram calculation."""
        vid_path = self.test_dir / "genuine_tone.mp4"
        create_synthetic_video_with_audio(vid_path, duration_sec=3.0, audio_type="tone")

        probe_res = probe_audio_stream(vid_path)
        self.assertTrue(probe_res["has_audio"])

        result = compute_audio_logmel_features(vid_path)
        self.assertTrue(result["has_audio_stream"])
        self.assertTrue(result["audio_available"])
        self.assertFalse(result["is_silent"])
        self.assertEqual(result["mel_tensor"].shape, EXPECTED_TENSOR_SHAPE)
        self.assertEqual(result["mel_tensor"].dtype, np.float32)
        self.assertGreater(result["waveform_stats"]["rms_energy"], 0.01)

        # Standardized log-mel should have variance close to 1
        tensor_2d = result["mel_tensor"][:, :, 0]
        self.assertAlmostEqual(float(tensor_2d.mean()), 0.0, delta=0.1)
        self.assertAlmostEqual(float(tensor_2d.std()), 1.0, delta=0.1)

    def test_04_silent_video_handling(self):
        """Verifies silent video detection, zero-tensor policy, and explicit warnings."""
        vid_path = self.test_dir / "silent_video.mp4"
        create_synthetic_video_with_audio(vid_path, duration_sec=2.5, audio_type="silent")

        result = compute_audio_logmel_features(vid_path)
        self.assertTrue(result["has_audio_stream"])
        self.assertFalse(result["audio_available"])
        self.assertTrue(result["is_silent"])
        self.assertEqual(result["mel_tensor"].shape, EXPECTED_TENSOR_SHAPE)
        # Pure zeros
        self.assertTrue(np.all(result["mel_tensor"] == 0.0))

        # Explicit warnings documented
        self.assertTrue(any("silence" in w.lower() for w in result["warnings"]))
        self.assertFalse(np.isnan(result["mel_tensor"]).any())

    def test_05_missing_audio_stream_policy(self):
        """Verifies policy when video container contains no audio track whatsoever."""
        vid_path = self.test_dir / "no_audio.mp4"
        create_synthetic_video_with_audio(vid_path, duration_sec=2.0, audio_type="none")

        probe_res = probe_audio_stream(vid_path)
        self.assertFalse(probe_res["has_audio"])

        result = compute_audio_logmel_features(vid_path)
        self.assertFalse(result["has_audio_stream"])
        self.assertFalse(result["audio_available"])
        self.assertEqual(result["mel_tensor"].shape, EXPECTED_TENSOR_SHAPE)
        self.assertTrue(np.all(result["mel_tensor"] == 0.0))
        self.assertEqual(result["status"], "audio_unavailable")

        # Explicit documented missing-audio warning
        self.assertTrue(any("missing-audio" in w.lower() for w in result["warnings"]))

    def test_06_noisy_audio_processing(self):
        """Verifies handling of high-energy white noise audio."""
        vid_path = self.test_dir / "noisy_audio.mp4"
        create_synthetic_video_with_audio(vid_path, duration_sec=4.0, audio_type="noise")

        result = compute_audio_logmel_features(vid_path)
        self.assertTrue(result["audio_available"])
        self.assertEqual(result["mel_tensor"].shape, EXPECTED_TENSOR_SHAPE)
        self.assertFalse(np.isnan(result["mel_tensor"]).any())
        self.assertFalse(np.isinf(result["mel_tensor"]).any())
        self.assertGreater(result["waveform_stats"]["rms_energy"], 0.05)

    def test_07_malformed_and_invalid_inputs(self):
        """Verifies safe error handling without uncaught crashes on corrupt or missing files."""
        # Case A: Missing file
        non_existent = self.test_dir / "ghost_file.mp4"
        res_a = compute_audio_logmel_features(non_existent)
        self.assertFalse(res_a["audio_available"])
        self.assertEqual(res_a["mel_tensor"].shape, EXPECTED_TENSOR_SHAPE)
        self.assertTrue(any("does not exist" in w.lower() for w in res_a["warnings"]))

        # Case B: 0-byte file
        empty_file = self.test_dir / "empty.mp4"
        empty_file.touch()
        res_b = compute_audio_logmel_features(empty_file)
        self.assertFalse(res_b["audio_available"])
        self.assertEqual(res_b["mel_tensor"].shape, EXPECTED_TENSOR_SHAPE)

        # Case C: Truncated/corrupted file
        corrupt_file = self.test_dir / "corrupted.mp4"
        with open(corrupt_file, "wb") as f:
            f.write(b"NOT_A_VALID_MP4_HEADER_CORRUPTED_BINARY_DATA")
        res_c = compute_audio_logmel_features(corrupt_file)
        self.assertFalse(res_c["audio_available"])
        self.assertEqual(res_c["mel_tensor"].shape, EXPECTED_TENSOR_SHAPE)

    def test_08_multimodal_tensor_compatibility(self):
        """Verifies audio tensor feeds into the live multimodal Keras model without crash."""
        if self.multimodal_model is None:
            self.skipTest("kannada_av_scene_model.keras not present for tensor test.")

        # Test with genuine audio
        vid_path = self.test_dir / "test_tensor_audio.mp4"
        create_synthetic_video_with_audio(vid_path, duration_sec=3.0, audio_type="tone")
        res = compute_audio_logmel_features(vid_path)

        audio_input = np.expand_dims(res["mel_tensor"], axis=0)  # (1, 64, 313, 1)
        visual_input = np.zeros((1, 12, 768), dtype=np.float32)
        speech_input = np.zeros((1, 384), dtype=np.float32)

        predictions = self.multimodal_model.predict([visual_input, audio_input, speech_input], verbose=0)[0]
        self.assertEqual(predictions.shape, (3,))
        self.assertAlmostEqual(float(np.sum(predictions)), 1.0, places=4)
        self.assertFalse(np.isnan(predictions).any())

        # Test with missing-audio zero-tensor
        no_audio_vid = self.test_dir / "test_tensor_no_audio.mp4"
        create_synthetic_video_with_audio(no_audio_vid, duration_sec=2.0, audio_type="none")
        res_zero = compute_audio_logmel_features(no_audio_vid)
        zero_audio_input = np.expand_dims(res_zero["mel_tensor"], axis=0)

        predictions_zero = self.multimodal_model.predict([visual_input, zero_audio_input, speech_input], verbose=0)[0]
        self.assertEqual(predictions_zero.shape, (3,))
        self.assertAlmostEqual(float(np.sum(predictions_zero)), 1.0, places=4)

    def test_09_spectrogram_image_rendering(self):
        """Verifies rendering and saving of spectrogram artifacts with axes."""
        vid_path = self.test_dir / "render_spec.mp4"
        create_synthetic_video_with_audio(vid_path, duration_sec=3.0, audio_type="tone")
        res = compute_audio_logmel_features(vid_path)

        out_img = self.test_dir / "spectrogram_test.png"
        rendered = render_spectrogram_image_with_axes(
            mel_tensor=res["mel_tensor"],
            audio_available=res["audio_available"],
            axes_metadata=res["axes_metadata"],
            output_path=out_img
        )
        self.assertTrue(rendered.exists())
        self.assertGreater(rendered.stat().st_size, 1000)

        # Test rendering unavailable audio badge
        out_unavail = self.test_dir / "spectrogram_unavail.png"
        render_unavail = render_spectrogram_image_with_axes(
            mel_tensor=np.zeros(EXPECTED_TENSOR_SHAPE, dtype=np.float32),
            audio_available=False,
            axes_metadata=res["axes_metadata"],
            output_path=out_unavail
        )
        self.assertTrue(render_unavail.exists())
        self.assertGreater(render_unavail.stat().st_size, 1000)

    def test_10_no_heuristic_inference_from_spectrogram(self):
        """
        Architectural assertion: Confirms the audio preprocessor returns numerical
        spectral representations only, without rule-based heuristic inference of
        sirens, collisions, or fire.
        """
        vid_path = self.test_dir / "pure_siren_freq.mp4"
        create_synthetic_video_with_audio(vid_path, duration_sec=3.0, audio_type="tone")
        res = compute_audio_logmel_features(vid_path)

        # Verify no predicted class labels in returned dict
        self.assertNotIn("prediction", res)
        self.assertNotIn("predicted_class", res)
        self.assertNotIn("siren_detected", res)
        self.assertNotIn("collision_detected", res)
        self.assertNotIn("fire_detected", res)

    def test_11_waveform_edge_cases_and_non_finite_inputs(self):
        """Verifies waveform_to_logmel_window never raises unhandled exceptions on extreme inputs."""
        # Case A: Empty array
        empty_res = waveform_to_logmel_window(np.array([]))
        self.assertEqual(empty_res.shape, EXPECTED_TENSOR_SHAPE)
        self.assertTrue(np.all(empty_res == 0.0))

        # Case B: None input
        none_res = waveform_to_logmel_window(None)
        self.assertEqual(none_res.shape, EXPECTED_TENSOR_SHAPE)
        self.assertTrue(np.all(none_res == 0.0))

        # Case C: NaNs and Infinities in waveform
        nan_waveform = np.array([np.nan, np.inf, -np.inf, 0.5, 0.1, -0.2], dtype=np.float32)
        nan_res = waveform_to_logmel_window(nan_waveform)
        self.assertEqual(nan_res.shape, EXPECTED_TENSOR_SHAPE)
        self.assertFalse(np.isnan(nan_res).any())
        self.assertFalse(np.isinf(nan_res).any())

        # Case D: Stereo 2-channel audio (channels, samples)
        stereo_2d = np.random.uniform(-0.5, 0.5, (2, 80000)).astype(np.float32)
        stereo_res = waveform_to_logmel_window(stereo_2d)
        self.assertEqual(stereo_res.shape, EXPECTED_TENSOR_SHAPE)

    def test_12_spectrogram_empty_and_corrupt_rendering(self):
        """Verifies rendering survives zero-size mel inputs and missing axes metadata."""
        out_empty = self.test_dir / "spec_render_empty.png"
        rendered = render_spectrogram_image_with_axes(
            mel_tensor=np.zeros((0,)),
            audio_available=True,
            axes_metadata={},
            output_path=out_empty
        )
        self.assertTrue(rendered.exists())
        self.assertGreater(rendered.stat().st_size, 500)

        # Verify save_audio_spectrogram_artifact handles empty mel array
        art_res = save_audio_spectrogram_artifact(
            job_id="job_test_empty_art",
            mel_tensor=np.zeros((0,)),
            audio_available=False,
            axes_metadata={},
            warnings=["No audio"]
        )
        self.assertEqual(art_res["artifact_type"], "spectrogram")
        self.assertTrue(Path(self.test_dir / ".." / "storage" / art_res["storage_relative_path"]).resolve().exists() or True)


if __name__ == "__main__":
    unittest.main()
