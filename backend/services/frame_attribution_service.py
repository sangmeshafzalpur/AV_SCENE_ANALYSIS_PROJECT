"""
Evidence-Based Influential Frame Attribution Service
===================================================
Provides principled temporal frame-ablation and sensitivity analysis
for the 12 extracted video frames using the verified `best_multimodal_model.keras`.

Mathematical Method:
  1. Baseline Forward Pass:
     Computes target class score S_base(c*) on unperturbed multimodal inputs.
  2. Temporal Frame Perturbation:
     For each frame t in {0..11}, frame t is ablated via a documented strategy
     (default: 'neutral_gray', or 'temporal_mean', 'adjacent_interpolation', 'zero_mask')
     while audio, audio_mask, and Kannada speech transcript remain strictly constant.
  3. Score Change / Attribution Drop:
     Delta S_t = S_base(c*) - S_perturbed^(t)(c*)
     - Positive Delta S_t: Frame t supported model confidence (ablation lowered confidence).
     - Negative Delta S_t: Frame t was unhelpful or counter-evidence (ablation raised confidence).
  4. Relative Importance Normalization:
     I_t = max(Delta S_t, 0) / sum_k(max(Delta S_k, 0) + eps)
  5. Ranking:
     Frames ranked 1..12 by descending Delta S_t. Top K (default 3) marked as keyframes.

Causation Disclaimer:
  Perturbation results indicate model sensitivity to individual temporal frame removals
  and do not constitute proof of real-world physical causation.
"""

import os
import sys
import time
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import tensorflow as tf

from services.model_inference_service import (
    load_multimodal_production_model,
    run_direct_multimodal_inference,
    CLASS_INDEX_MAP,
    CLASS_NAMES,
    EXPECTED_VIDEO_SHAPE,
    EXPECTED_AUDIO_SHAPE,
    MODEL_FILENAME
)

# Standard causation disclaimer required for all attribution outputs
CAUSATION_DISCLAIMER = (
    "Perturbation results indicate model sensitivity to individual temporal frame removals "
    "and do not constitute proof of real-world physical causation."
)

SUPPORTED_STRATEGIES = [
    "neutral_gray",
    "temporal_mean",
    "adjacent_interpolation",
    "zero_mask"
]


def perturb_frame(
    video_frames: np.ndarray,
    frame_index: int,
    strategy: str = "neutral_gray"
) -> np.ndarray:
    """
    Constructs an ablated video tensor where a single frame is perturbed
    and all other 11 frames remain strictly identical.

    Strategies:
      - 'neutral_gray': Neutral luminance (RGB 128.0), standard in occlusion sensitivity.
      - 'temporal_mean': Temporal average of the 12 video frames, preserving scene lighting.
      - 'adjacent_interpolation': Linear average of temporal neighbors (t-1, t+1).
      - 'zero_mask': Zero-valued black frame (RGB 0.0).
    """
    if not isinstance(video_frames, np.ndarray):
        raise TypeError(f"video_frames must be a numpy ndarray, got {type(video_frames)}")
    if video_frames.shape != EXPECTED_VIDEO_SHAPE:
        raise ValueError(f"Expected video_frames shape {EXPECTED_VIDEO_SHAPE}, got {video_frames.shape}")
    if not (0 <= frame_index < 12):
        raise IndexError(f"frame_index must be between 0 and 11, got {frame_index}")
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unknown perturbation strategy '{strategy}'. Supported: {SUPPORTED_STRATEGIES}")

    perturbed = video_frames.copy().astype(np.float32)

    if strategy == "neutral_gray":
        perturbed[frame_index] = np.full((224, 224, 3), 128.0, dtype=np.float32)

    elif strategy == "temporal_mean":
        mean_frame = np.mean(video_frames, axis=0, keepdims=False).astype(np.float32)
        perturbed[frame_index] = mean_frame

    elif strategy == "adjacent_interpolation":
        if frame_index == 0:
            perturbed[frame_index] = video_frames[1].astype(np.float32)
        elif frame_index == 11:
            perturbed[frame_index] = video_frames[10].astype(np.float32)
        else:
            prev_f = video_frames[frame_index - 1].astype(np.float32)
            next_f = video_frames[frame_index + 1].astype(np.float32)
            perturbed[frame_index] = (prev_f + next_f) / 2.0

    elif strategy == "zero_mask":
        perturbed[frame_index] = np.zeros((224, 224, 3), dtype=np.float32)

    return perturbed


def compute_evidence_based_frame_attributions(
    video_frames: np.ndarray,
    frame_timestamps_sec: List[float],
    audio_spectrogram: Optional[np.ndarray] = None,
    audio_available: bool = True,
    transcript: Optional[str] = None,
    target_class: Optional[str] = None,
    perturbation_strategy: str = "neutral_gray",
    top_k_keyframes: int = 3
) -> Dict[str, Any]:
    """
    Measures evidence-based frame sensitivity and score changes by perturbing one frame
    at a time while keeping other frames, audio, and speech text inputs strictly identical.

    Returns:
      Comprehensive ranked attributions including baseline score, perturbed score,
      score difference, relative importance, rank, keyframe designation, and causation disclaimer.
    """
    t_start = time.time()
    warnings: List[str] = []

    # 1. Input Validations
    if not isinstance(video_frames, np.ndarray):
        raise TypeError(f"video_frames must be a numpy ndarray, got {type(video_frames)}")
    if video_frames.shape != EXPECTED_VIDEO_SHAPE:
        raise ValueError(f"Expected video_frames shape {EXPECTED_VIDEO_SHAPE}, got {video_frames.shape}")

    if not isinstance(frame_timestamps_sec, (list, tuple)) or len(frame_timestamps_sec) != 12:
        raise ValueError(
            f"frame_timestamps_sec must be a list of 12 genuine timestamps, got {len(frame_timestamps_sec) if hasattr(frame_timestamps_sec, '__len__') else type(frame_timestamps_sec)}"
        )

    if perturbation_strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy '{perturbation_strategy}'. Choose from {SUPPORTED_STRATEGIES}")

    # 2. Baseline Model Inference
    baseline_res = run_direct_multimodal_inference(
        video_frames=video_frames,
        audio_spectrogram=audio_spectrogram,
        audio_available=audio_available,
        transcript=transcript
    )

    if target_class is None:
        target_class = baseline_res["predicted_class"]

    if target_class not in CLASS_NAMES:
        raise ValueError(f"Invalid target_class '{target_class}'. Must be one of {CLASS_NAMES}")

    target_class_idx = CLASS_NAMES.index(target_class)
    baseline_target_score = float(baseline_res["class_scores"][target_class])

    # 3. Frame-by-Frame Perturbation Analysis
    measured_frames: List[Dict[str, Any]] = []
    raw_drops: List[float] = []

    for t in range(12):
        perturbed_video = perturb_frame(video_frames, frame_index=t, strategy=perturbation_strategy)

        # Forward pass on perturbed video with strictly identical audio and text
        pert_res = run_direct_multimodal_inference(
            video_frames=perturbed_video,
            audio_spectrogram=audio_spectrogram,
            audio_available=audio_available,
            transcript=transcript
        )

        perturbed_target_score = float(pert_res["class_scores"][target_class])
        score_diff = baseline_target_score - perturbed_target_score  # Positive = frame supported baseline

        raw_drops.append(score_diff)
        measured_frames.append({
            "frame_index": t,
            "timestamp_sec": float(frame_timestamps_sec[t]),
            "baseline_score": round(baseline_target_score, 6),
            "perturbed_score": round(perturbed_target_score, 6),
            "score_difference": round(score_diff, 6),
            "predicted_class_when_ablated": pert_res["predicted_class"]
        })

    # 4. Normalized Relative Importance
    positive_drops = [max(d, 0.0) for d in raw_drops]
    sum_pos = sum(positive_drops)

    if sum_pos > 1e-6:
        normalized_scores = [round(p / sum_pos, 6) for p in positive_drops]
    else:
        # If all score drops are zero or negative, sensitivity is uniform
        normalized_scores = [round(1.0 / 12.0, 6) for _ in range(12)]
        warnings.append(
            "All temporal frame score drops were <= 0; uniform relative importance assigned."
        )

    # 5. Ranking (Descending by score_difference, ties broken by chronological index)
    # Highest drop means removing this frame most damaged target class confidence -> Rank 1
    indexed_drops = [(raw_drops[i], -i, i) for i in range(12)]
    sorted_drops = sorted(indexed_drops, reverse=True)

    rank_map: Dict[int, int] = {}
    for rank_idx, (_, _, original_idx) in enumerate(sorted_drops, start=1):
        rank_map[original_idx] = rank_idx

    top_keyframe_indices = set(orig_idx for _, _, orig_idx in sorted_drops[:top_k_keyframes])

    for i in range(12):
        measured_frames[i]["attribution_score"] = normalized_scores[i]
        measured_frames[i]["rank"] = rank_map[i]
        measured_frames[i]["is_keyframe"] = (i in top_keyframe_indices)
        measured_frames[i]["label"] = f"Rank {rank_map[i]} (Δ {raw_drops[i]:+.4f})"

    # 6. Sensitivity & Reliability Check
    score_range = max(raw_drops) - min(raw_drops)
    if score_range < 1e-4:
        reliability = "diffuse_or_modality_dominated"
        warnings.append(
            f"Low visual attribution sensitivity (score range {score_range:.6f} < 0.0001). "
            f"The model's prediction is likely diffuse or dominated by acoustic/speech modalities."
        )
    else:
        reliability = "reliable_sensitivity"

    duration_sec = round(time.time() - t_start, 4)

    return {
        "target_class": target_class,
        "target_class_index": target_class_idx,
        "baseline_score": round(baseline_target_score, 6),
        "attribution_method": f"temporal_frame_ablation ({perturbation_strategy})",
        "perturbation_strategy": perturbation_strategy,
        "causation_disclaimer": CAUSATION_DISCLAIMER,
        "reliability": reliability,
        "score_range": round(score_range, 6),
        "total_frames_evaluated": 12,
        "top_keyframes_count": top_k_keyframes,
        "timings": {
            "attribution_duration_seconds": duration_sec,
            "average_perturbation_pass_seconds": round(duration_sec / 13.0, 4)
        },
        "model_identification": {
            "model_file": MODEL_FILENAME,
            "sha256": baseline_res["model_identification"]["sha256"]
        },
        "warnings": warnings,
        "frames": measured_frames
    }


def compute_gradient_frame_attributions(
    video_frames: np.ndarray,
    frame_timestamps_sec: List[float],
    audio_spectrogram: Optional[np.ndarray] = None,
    audio_available: bool = True,
    transcript: Optional[str] = None,
    target_class: Optional[str] = None,
    top_k_keyframes: int = 3
) -> Dict[str, Any]:
    """
    Gradient-based temporal saliency attribution (Simonyan et al., 2013).
    Measures the Frobenius norm of the gradient of the target class score
    with respect to each frame: G_t = || dS(c*) / dF_t ||_F.
    """
    t_start = time.time()
    warnings: List[str] = []

    if not isinstance(video_frames, np.ndarray) or video_frames.shape != EXPECTED_VIDEO_SHAPE:
        raise ValueError(f"Expected video_frames shape {EXPECTED_VIDEO_SHAPE}")
    if len(frame_timestamps_sec) != 12:
        raise ValueError("frame_timestamps_sec must have 12 items")

    model = load_multimodal_production_model()

    # Prepare tensors
    video_tensor = tf.convert_to_tensor(np.expand_dims(video_frames, axis=0).astype(np.float32))

    if audio_spectrogram is None or not audio_available:
        audio_tensor = tf.zeros((1, *EXPECTED_AUDIO_SHAPE), dtype=tf.float32)
        mask_tensor = tf.zeros((1, 1), dtype=tf.float32)
    else:
        audio_tensor = tf.convert_to_tensor(np.expand_dims(audio_spectrogram, axis=0).astype(np.float32))
        mask_tensor = tf.ones((1, 1), dtype=tf.float32)

    clean_text = str(transcript or "").strip()
    text_tensor = tf.convert_to_tensor(np.array([clean_text], dtype=object))

    with tf.GradientTape() as tape:
        tape.watch(video_tensor)
        preds = model([video_tensor, audio_tensor, mask_tensor, text_tensor], training=False)
        if target_class is None:
            pred_idx = int(tf.argmax(preds[0]))
            target_class = CLASS_INDEX_MAP[pred_idx]
        else:
            target_class_idx = CLASS_NAMES.index(target_class)
            pred_idx = target_class_idx

        target_score = preds[0, pred_idx]

    grads = tape.gradient(target_score, video_tensor)
    if grads is None:
        raise RuntimeError("Gradient computation failed for model input tensor.")

    baseline_score = float(target_score.numpy())
    frame_grad_norms: List[float] = []

    for t in range(12):
        f_grad = grads[0, t]
        norm_val = float(tf.norm(f_grad).numpy())
        frame_grad_norms.append(norm_val)

    sum_norms = sum(frame_grad_norms)
    if sum_norms > 1e-8:
        normalized_scores = [round(n / sum_norms, 6) for n in frame_grad_norms]
    else:
        normalized_scores = [round(1.0 / 12.0, 6) for _ in range(12)]
        warnings.append("All gradient norms near zero; uniform saliency assigned.")

    indexed_norms = [(frame_grad_norms[i], -i, i) for i in range(12)]
    sorted_norms = sorted(indexed_norms, reverse=True)

    rank_map: Dict[int, int] = {}
    for rank_idx, (_, _, orig_idx) in enumerate(sorted_norms, start=1):
        rank_map[orig_idx] = rank_idx

    top_indices = set(orig_idx for _, _, orig_idx in sorted_norms[:top_k_keyframes])

    measured_frames: List[Dict[str, Any]] = []
    for i in range(12):
        measured_frames.append({
            "frame_index": i,
            "timestamp_sec": float(frame_timestamps_sec[i]),
            "baseline_score": round(baseline_score, 6),
            "gradient_norm": round(frame_grad_norms[i], 6),
            "gradient_frobenius_norm": round(frame_grad_norms[i], 6),
            "attribution_score": normalized_scores[i],
            "rank": rank_map[i],
            "is_keyframe": (i in top_indices),
            "label": f"Rank {rank_map[i]} (||grad||={frame_grad_norms[i]:.4e})"
        })

    duration_sec = round(time.time() - t_start, 4)

    return {
        "target_class": target_class,
        "target_class_index": CLASS_NAMES.index(target_class),
        "baseline_score": round(baseline_score, 6),
        "attribution_method": "gradient_temporal_saliency (Frobenius norm)",
        "causation_disclaimer": CAUSATION_DISCLAIMER,
        "reliability": "gradient_sensitivity",
        "total_frames_evaluated": 12,
        "top_keyframes_count": top_k_keyframes,
        "timings": {
            "attribution_duration_seconds": duration_sec
        },
        "model_identification": {
            "model_file": MODEL_FILENAME
        },
        "warnings": warnings,
        "frames": measured_frames
    }
