"""
Production Model Inference Service for Multimodal Emergency Analysis
====================================================================
Integrates the preserved `best_multimodal_model.keras` into the production pipeline.
Model Architecture:
  EfficientNetB0_BiLSTM_AudioCNN_KannadaASR_TextBiLSTM
  - Visual: TimeDistributed(EfficientNetB0) -> BiLSTM -> Dense(128)
  - Acoustic: 3x Conv2D/MaxPool -> GAP -> Dense(128) gated by audio_mask Dense(128, sigmoid)
  - Speech: TextVectorization -> Embedding -> BiLSTM -> Dense(128)
  - Cross-Modal Fusion: Concatenate -> Dense(256) -> Dropout -> Dense(128) -> Dense(3, softmax)

Verified Class Mapping:
  Index 0 -> "Accident"
  Index 1 -> "Fire"
  Index 2 -> "Normal"

Verified Input Signatures:
  1. video      : shape (None, 12, 224, 224, 3) float32 [RGB 0..255]
  2. audio      : shape (None, 64, 313, 1) float32 [Standardized log-mel spectrogram]
  3. audio_mask : shape (None, 1) float32 [1.0 = present, 0.0 = missing/silent]
  4. text       : shape (None,) string [Raw Kannada transcript or empty string]

Output Signature:
  event_prediction : shape (None, 3) float32 [Softmax class distribution]
"""

import os
import sys
import json
import time
import hashlib
import builtins
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import tensorflow as tf

# Reference constants
MODEL_FILENAME = "best_multimodal_model.keras"
CLASS_NAMES = ["Accident", "Fire", "Normal"]
CLASS_INDEX_MAP = {
    0: "Accident",
    1: "Fire",
    2: "Normal"
}
EXPECTED_VIDEO_SHAPE = (12, 224, 224, 3)
EXPECTED_AUDIO_SHAPE = (64, 313, 1)
EXPECTED_OUTPUT_DIM = 3

_MODEL_INSTANCE: Optional[Any] = None
_MODEL_LOCK = threading.Lock()
_MODEL_CHECKSUM: Optional[str] = None
_MODEL_METADATA_CACHE: Optional[Dict[str, Any]] = None


def _get_candidate_model_paths() -> List[Path]:
    """Returns candidate search paths for best_multimodal_model.keras."""
    backend_dir = Path(__file__).resolve().parent.parent
    workspace_dir = backend_dir.parent
    return [
        workspace_dir / MODEL_FILENAME,
        backend_dir / MODEL_FILENAME,
        Path(r"c:\Users\Hp\OneDrive\Desktop\AVS2\model\best_multimodal_model.keras"),
        Path(r"c:\Users\Hp\OneDrive\Desktop\hphphphphhphph\best_multimodal_model.keras")
    ]


def locate_model_file() -> Path:
    """Locates the preserved best_multimodal_model.keras file."""
    for p in _get_candidate_model_paths():
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(
        f"Could not locate {MODEL_FILENAME} in candidate locations: "
        f"{[str(p) for p in _get_candidate_model_paths()]}"
    )


def compute_sha256(filepath: Path) -> str:
    """Calculates SHA256 checksum of file in binary chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_multimodal_production_model(force_reload: bool = False) -> Any:
    """
    Thread-safe, cached loader for best_multimodal_model.keras.
    Applies UTF-8 file reading hook to prevent Windows charmap encoding errors
    when Keras 3 deserializes Kannada vocabulary assets from the .keras archive.
    """
    global _MODEL_INSTANCE, _MODEL_CHECKSUM, _MODEL_METADATA_CACHE

    with _MODEL_LOCK:
        if _MODEL_INSTANCE is not None and not force_reload:
            return _MODEL_INSTANCE

        model_path = locate_model_file()
        _MODEL_CHECKSUM = compute_sha256(model_path)

        # Hook builtins.open to force UTF-8 text decoding on Windows
        orig_open = builtins.open

        def utf8_open(file, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if "b" not in mode and "encoding" not in kwargs:
                kwargs["encoding"] = "utf-8"
            return orig_open(file, *args, **kwargs)

        builtins.open = utf8_open
        try:
            print(f"Loading {MODEL_FILENAME} from {model_path} (SHA256: {_MODEL_CHECKSUM[:12]}...)...")
            loaded_model = tf.keras.models.load_model(str(model_path), compile=False)
            _MODEL_INSTANCE = loaded_model
            print(f"Successfully loaded {loaded_model.name} ({loaded_model.count_params():,} parameters)")
        finally:
            builtins.open = orig_open

        # Build metadata cache
        total_p = _MODEL_INSTANCE.count_params()
        trainable_p = sum(
            tf.keras.backend.count_params(w) for w in _MODEL_INSTANCE.trainable_weights
        ) if hasattr(_MODEL_INSTANCE, "trainable_weights") else total_p
        non_trainable_p = total_p - trainable_p

        _MODEL_METADATA_CACHE = {
            "canonical_name": "EfficientNetB0_BiLSTM_AudioCNN_KannadaASR_TextBiLSTM",
            "active_model_filename": MODEL_FILENAME,
            "active_model_path": str(model_path),
            "sha256_checksum": _MODEL_CHECKSUM,
            "file_size_bytes": model_path.stat().st_size,
            "total_parameters": int(total_p),
            "trainable_parameters": int(trainable_p),
            "non_trainable_parameters": int(non_trainable_p),
            "class_mapping": CLASS_INDEX_MAP,
            "classes": CLASS_NAMES,
            "input_signatures": [
                {"name": "video", "shape": [None, 12, 224, 224, 3], "dtype": "float32"},
                {"name": "audio", "shape": [None, 64, 313, 1], "dtype": "float32"},
                {"name": "audio_mask", "shape": [None, 1], "dtype": "float32"},
                {"name": "text", "shape": [None], "dtype": "string"}
            ],
            "output_signature": {
                "name": "event_prediction",
                "shape": [None, 3],
                "dtype": "float32",
                "activation": "softmax"
            }
        }

        return _MODEL_INSTANCE


def get_production_model_metadata() -> Dict[str, Any]:
    """Returns cached verified metadata for the production model."""
    if _MODEL_METADATA_CACHE is None:
        load_multimodal_production_model()
    return dict(_MODEL_METADATA_CACHE)


def run_direct_multimodal_inference(
    video_frames: np.ndarray,
    audio_spectrogram: Optional[np.ndarray] = None,
    audio_available: bool = True,
    transcript: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes trusted direct inference with best_multimodal_model.keras.
    Validates shapes, dtypes, handles missing modalities explicitly, and measures inference latency.
    """
    t0 = time.time()
    warnings: List[str] = []

    # 1. Validate Video Input (12, 224, 224, 3)
    if not isinstance(video_frames, np.ndarray):
        raise TypeError(f"video_frames must be a numpy ndarray, got {type(video_frames)}")
    if video_frames.ndim == 4:
        if video_frames.shape != EXPECTED_VIDEO_SHAPE:
            raise ValueError(f"Expected video_frames shape {EXPECTED_VIDEO_SHAPE}, got {video_frames.shape}")
        video_tensor = np.expand_dims(video_frames, axis=0).astype(np.float32)
    elif video_frames.ndim == 5:
        if video_frames.shape[1:] != EXPECTED_VIDEO_SHAPE:
            raise ValueError(f"Expected batch video_frames shape (N, 12, 224, 224, 3), got {video_frames.shape}")
        video_tensor = video_frames.astype(np.float32)
    else:
        raise ValueError(f"Invalid video_frames dimensions: {video_frames.ndim} (expected 4D or 5D)")

    # 2. Validate Audio & Audio Mask (64, 313, 1) and (1,)
    if audio_spectrogram is None or not audio_available:
        audio_tensor = np.zeros((1, *EXPECTED_AUDIO_SHAPE), dtype=np.float32)
        audio_mask_tensor = np.array([[0.0]], dtype=np.float32)
        warnings.append("Missing/silent audio policy active: audio_mask set to 0.0 with zero-valued acoustic tensor.")
        audio_modality_active = False
    else:
        if not isinstance(audio_spectrogram, np.ndarray):
            raise TypeError("audio_spectrogram must be a numpy ndarray")
        if audio_spectrogram.ndim == 3:
            if audio_spectrogram.shape != EXPECTED_AUDIO_SHAPE:
                raise ValueError(f"Expected audio_spectrogram shape {EXPECTED_AUDIO_SHAPE}, got {audio_spectrogram.shape}")
            audio_tensor = np.expand_dims(audio_spectrogram, axis=0).astype(np.float32)
        elif audio_spectrogram.ndim == 4:
            if audio_spectrogram.shape[1:] != EXPECTED_AUDIO_SHAPE:
                raise ValueError(f"Expected batch audio shape (N, 64, 313, 1), got {audio_spectrogram.shape}")
            audio_tensor = audio_spectrogram.astype(np.float32)
        else:
            raise ValueError(f"Invalid audio_spectrogram ndim: {audio_spectrogram.ndim}")
        audio_mask_tensor = np.array([[1.0]], dtype=np.float32)
        audio_modality_active = True

    # 3. Validate Text Input (1,) string
    clean_text = str(transcript or "").strip()
    text_tensor = np.array([clean_text], dtype=object)
    speech_modality_active = bool(len(clean_text) > 0)
    if not speech_modality_active:
        warnings.append("Unavailable transcript: supplied empty string to model TextVectorization branch.")

    # 4. Thread-Safe Model Execution
    model = load_multimodal_production_model()
    with _MODEL_LOCK:
        raw_output = model.predict(
            [video_tensor, audio_tensor, audio_mask_tensor, text_tensor],
            verbose=0
        )[0]

    inference_duration = round(time.time() - t0, 4)

    # 5. Extract Scores and Class Mapping
    # Class order: 0=Accident, 1=Fire, 2=Normal
    accident_score = float(raw_output[0])
    fire_score = float(raw_output[1])
    normal_score = float(raw_output[2])

    pred_idx = int(np.argmax(raw_output))
    predicted_class = CLASS_INDEX_MAP[pred_idx]
    confidence_score = float(raw_output[pred_idx])

    class_scores = {
        "Accident": round(accident_score, 6),
        "Fire": round(fire_score, 6),
        "Normal": round(normal_score, 6)
    }

    metadata = get_production_model_metadata()

    return {
        "predicted_class": predicted_class,
        "predicted_index": pred_idx,
        "class_scores": class_scores,
        "confidence": round(confidence_score, 6),
        "score_semantics": "uncalibrated_softmax_scores",
        "is_calibrated": False,
        "calibration_note": (
            "Raw output scores represent uncalibrated softmax logits; "
            "scores indicate model preference but should not be interpreted as empirical probabilities."
        ),
        "modality_availability": {
            "video": True,
            "audio": audio_modality_active,
            "speech": speech_modality_active
        },
        "timings": {
            "inference_duration_seconds": inference_duration
        },
        "model_identification": {
            "canonical_name": metadata["canonical_name"],
            "model_file": metadata["active_model_filename"],
            "sha256": metadata["sha256_checksum"],
            "total_parameters": metadata["total_parameters"]
        },
        "warnings": warnings
    }
