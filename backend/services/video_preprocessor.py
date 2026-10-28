import os
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image

import database
from services.storage_manager import (
    STORAGE_ROOT,
    ARTIFACTS_DIR,
    save_frame_as_artifact,
    compute_file_sha256
)

# Verified Model Constants (from training notebook Cell 2, Cell 4, Cell 8)
NUM_FRAMES = 12
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
VISUAL_FEATURE_DIM = 768
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
MIN_FILE_SIZE_BYTES = 1024                # 1 KB
MIN_REQUIRED_FREE_STORAGE_BYTES = 200 * 1024 * 1024  # 200 MB

def validate_video_file_detailed(video_path: Path) -> Dict[str, Any]:
    """
    Performs comprehensive pre-flight validation on a video file:
    1. Existence and extension validation against verified training formats.
    2. File size constraints (1 KB to 100 MB).
    3. Storage availability on the host storage disk.
    4. Media container integrity, stream readability, dimensions, FPS, and duration.
    """
    path = Path(video_path)
    warnings: List[str] = []

    # 1. Existence Check
    if not path.exists():
        return {"is_valid": False, "error": f"Video file not found at: {path}", "warnings": warnings, "metadata": {}}
    if not path.is_file():
        return {"is_valid": False, "error": f"Path is not a regular file: {path}", "warnings": warnings, "metadata": {}}

    # 2. Supported Format Check
    ext = path.suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        return {
            "is_valid": False,
            "error": f"Unsupported video format '{ext}'. Allowed extensions: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}",
            "warnings": warnings,
            "metadata": {}
        }

    # 3. File Size Constraints
    file_size = path.stat().st_size
    if file_size < MIN_FILE_SIZE_BYTES:
        return {
            "is_valid": False,
            "error": f"Video file is too small or empty ({file_size} bytes). Minimum size is {MIN_FILE_SIZE_BYTES} bytes.",
            "warnings": warnings,
            "metadata": {}
        }
    if file_size > MAX_FILE_SIZE_BYTES:
        return {
            "is_valid": False,
            "error": f"Video file size ({round(file_size / (1024 * 1024), 2)} MB) exceeds maximum limit of 100 MB.",
            "warnings": warnings,
            "metadata": {}
        }

    # 4. Storage Availability
    disk_info = shutil.disk_usage(STORAGE_ROOT)
    if disk_info.free < MIN_REQUIRED_FREE_STORAGE_BYTES:
        return {
            "is_valid": False,
            "error": f"Insufficient server storage space ({round(disk_info.free / (1024 * 1024), 1)} MB free). Minimum 200 MB required.",
            "warnings": warnings,
            "metadata": {}
        }

    # 5. Container & Readability Validation via OpenCV
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {
            "is_valid": False,
            "error": f"Corrupt or unreadable video file: OpenCV could not open container at {path.name}",
            "warnings": warnings,
            "metadata": {}
        }

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)

        # Dimension checks
        if width <= 0 or height <= 0:
            return {
                "is_valid": False,
                "error": f"Invalid video dimensions ({width}x{height}). Video must have positive width and height.",
                "warnings": warnings,
                "metadata": {}
            }

        # Frame count checks
        if total_frames <= 0:
            return {
                "is_valid": False,
                "error": "Video contains 0 readable frames.",
                "warnings": warnings,
                "metadata": {}
            }

        # Frame rate checks
        if not fps or fps <= 0 or np.isnan(fps):
            warnings.append("Video container reported invalid or 0 FPS; default 25.0 FPS will be assumed for timestamps.")
            effective_fps = 25.0
        else:
            effective_fps = float(fps)
            if effective_fps < 1.0:
                warnings.append(f"Unusually low frame rate detected: {effective_fps:.2f} FPS.")
            elif effective_fps > 120.0:
                warnings.append(f"Unusually high frame rate detected: {effective_fps:.2f} FPS.")

        duration = total_frames / effective_fps if effective_fps > 0 else 0.0
        if duration < 0.1:
            return {
                "is_valid": False,
                "error": f"Video duration ({duration:.2f}s) is too short to perform scene analysis.",
                "warnings": warnings,
                "metadata": {}
            }

        if total_frames < NUM_FRAMES:
            warnings.append(
                f"Video is short ({total_frames} frames < {NUM_FRAMES}). Verified temporal-sampling policy will sample available frames without synthesis."
            )

        # Probe first frame to verify decoder pipeline
        success, first_frame = capture.read()
        if not success or first_frame is None or first_frame.size == 0:
            return {
                "is_valid": False,
                "error": "Failed to decode initial video frame: Video stream may be corrupt or truncated.",
                "warnings": warnings,
                "metadata": {}
            }

        metadata = {
            "filename": path.name,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "fps": round(effective_fps, 2),
            "duration_seconds": round(duration, 3)
        }

        return {"is_valid": True, "error": None, "warnings": warnings, "metadata": metadata}

    finally:
        capture.release()

def get_uniform_frame_indices(total_frames: int, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """
    Selects exactly num_frames (12) approximately uniformly spaced frame indices
    according to the verified training sampling policy.
    Never fabricates frames or indices.
    """
    if total_frames <= 0:
        raise ValueError("Video contains no readable frames.")

    if total_frames == 1:
        return np.zeros(num_frames, dtype=np.int64)

    indices = np.linspace(0, total_frames - 1, num_frames)
    return np.round(indices).astype(np.int64)

def extract_video_frames_with_metadata(
    video_path: Path,
    num_frames: int = NUM_FRAMES,
    target_height: int = IMAGE_HEIGHT,
    target_width: int = IMAGE_WIDTH
) -> Dict[str, Any]:
    """
    Extracts exactly num_frames RGB frames according to verified training preprocessing:
    - Temporal uniform index selection
    - Exact frame timestamp preservation
    - BGR -> RGB color conversion
    - Resize to (224, 224) using cv2.INTER_AREA (identical to training)
    - uint8 numpy array of shape (12, 224, 224, 3)
    """
    path = Path(video_path)
    t_start = time.time()
    timings = {}

    validation = validate_video_file_detailed(path)
    if not validation["is_valid"]:
        raise ValueError(validation["error"])

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    try:
        total_frames = validation["metadata"]["total_frames"]
        fps = validation["metadata"]["fps"]
        fps_val = fps if fps > 0 else 25.0

        indices = get_uniform_frame_indices(total_frames, num_frames=num_frames)
        
        frames = []
        timestamps = []
        decoded_cache: Dict[int, Tuple[np.ndarray, float]] = {}
        
        t_decode_start = time.time()
        for frame_idx in indices:
            idx_int = int(frame_idx)
            
            if idx_int in decoded_cache:
                frame_bgr, ts_sec = decoded_cache[idx_int]
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, idx_int)
                pos_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
                success, frame_bgr = capture.read()

                if not success or frame_bgr is None:
                    # If seeking to repeat index failed on single/short frame video, reuse last successfully decoded frame
                    if decoded_cache:
                        last_key = max(decoded_cache.keys())
                        frame_bgr, ts_sec = decoded_cache[last_key]
                    else:
                        capture.release()
                        raise RuntimeError(f"Could not read genuine frame at index {frame_idx} from {path.name}")
                else:
                    if pos_msec and pos_msec >= 0:
                        ts_sec = round(pos_msec / 1000.0, 3)
                    else:
                        ts_sec = round(float(idx_int) / fps_val, 3)
                    decoded_cache[idx_int] = (frame_bgr, ts_sec)

            timestamps.append(ts_sec)

            # Color Channel Ordering: BGR -> RGB (verified training pipeline)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Image Resizing: 224x224 via cv2.INTER_AREA (verified training pipeline)
            frame_resized = cv2.resize(
                frame_rgb,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA
            )
            frames.append(frame_resized)

        timings["frame_decode_and_resize_seconds"] = round(time.time() - t_decode_start, 3)
        timings["total_extraction_seconds"] = round(time.time() - t_start, 3)

        frames_array = np.asarray(frames, dtype=np.uint8)

        assert frames_array.shape == (num_frames, target_height, target_width, 3), (
            f"Extracted frames shape {frames_array.shape} does not match expected ({num_frames}, {target_height}, {target_width}, 3)"
        )

        return {
            "frames": frames_array,
            "sampled_indices": [int(idx) for idx in indices],
            "frame_timestamps_sec": timestamps,
            "total_frames_in_video": total_frames,
            "fps": fps,
            "duration_seconds": validation["metadata"]["duration_seconds"],
            "original_width": validation["metadata"]["width"],
            "original_height": validation["metadata"]["height"],
            "target_resolution": [target_width, target_height],
            "color_format": "RGB",
            "interpolation": "INTER_AREA",
            "warnings": validation["warnings"],
            "timings": timings
        }

    finally:
        capture.release()

def compute_visual_embeddings(
    frames_rgb: np.ndarray,
    visual_processor,
    visual_model,
    device: torch.device
) -> Tuple[np.ndarray, float]:
    """
    Applies ConvNeXt-V2 feature extraction to the 12 preprocessed RGB frames.
    Produces a tensor of shape (1, 12, 768) matching the Keras model's visual_input signature.
    """
    t0 = time.time()
    pil_images = [Image.fromarray(f) for f in frames_rgb]
    inputs = visual_processor(images=pil_images, return_tensors="pt")

    target_dtype = next(visual_model.parameters()).dtype
    pixel_values = inputs["pixel_values"].to(device=device, dtype=target_dtype)

    with torch.inference_mode():
        outputs = visual_model(pixel_values=pixel_values)

    pooler_output = outputs.pooler_output.detach().float().cpu().numpy()
    visual_features = np.expand_dims(pooler_output, axis=0)  # Shape: (1, 12, 768)

    assert visual_features.shape == (1, NUM_FRAMES, VISUAL_FEATURE_DIM), (
        f"Visual features shape {visual_features.shape} does not match model signature (1, {NUM_FRAMES}, {VISUAL_FEATURE_DIM})"
    )

    inference_seconds = round(time.time() - t0, 3)
    return visual_features.astype(np.float32), inference_seconds

def generate_video_thumbnails_and_contact_sheet(
    job_id: str,
    frames_rgb: np.ndarray,
    timestamps_sec: List[float]
) -> Dict[str, Any]:
    """
    Saves all 12 extracted frames as managed artifacts, creates a primary thumbnail
    and a 12-frame contact sheet filmstrip for frontend visual inspection.
    """
    job_artifacts_dir = ARTIFACTS_DIR / job_id
    job_artifacts_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = job_artifacts_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_artifacts = []
    
    # 1. Save all 12 frames
    for idx in range(len(frames_rgb)):
        frame_rgb = frames_rgb[idx]
        ts = timestamps_sec[idx]
        filename = f"frame_{idx:02d}.jpg"
        abs_path = frames_dir / filename
        
        cv2.imwrite(str(abs_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        
        rel_path = f"artifacts/{job_id}/frames/{filename}"
        art_id = f"art_frame_{job_id}_{idx:02d}"
        file_size = abs_path.stat().st_size
        file_hash = compute_file_sha256(abs_path)

        meta = {
            "frame_index": idx,
            "timestamp_sec": round(ts, 2),
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT
        }

        database.add_artifact(
            artifact_id=art_id,
            job_id=job_id,
            artifact_type="frame",
            filename=filename,
            storage_relative_path=rel_path,
            file_size_bytes=file_size,
            mime_type="image/jpeg",
            file_hash_sha256=file_hash,
            metadata=meta
        )

        frame_artifacts.append({
            "artifact_id": art_id,
            "frame_index": idx,
            "timestamp_sec": ts,
            "filename": filename,
            "download_url": f"/api/jobs/{job_id}/artifacts/{art_id}"
        })

    # 2. Save primary representative thumbnail (middle frame)
    middle_idx = len(frames_rgb) // 2
    thumb_path = job_artifacts_dir / "thumbnail.jpg"
    cv2.imwrite(str(thumb_path), cv2.cvtColor(frames_rgb[middle_idx], cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    thumb_art_id = f"art_thumb_{job_id}"
    database.add_artifact(
        artifact_id=thumb_art_id,
        job_id=job_id,
        artifact_type="thumbnail",
        filename="thumbnail.jpg",
        storage_relative_path=f"artifacts/{job_id}/thumbnail.jpg",
        file_size_bytes=thumb_path.stat().st_size,
        mime_type="image/jpeg",
        file_hash_sha256=compute_file_sha256(thumb_path),
        metadata={"selected_frame_index": middle_idx, "timestamp_sec": timestamps_sec[middle_idx]}
    )

    # 3. Create 4x3 Contact Sheet Filmstrip
    # 4 rows, 3 columns (or 3 rows of 4)
    rows, cols = 3, 4
    pad = 4
    sheet_h = rows * IMAGE_HEIGHT + (rows + 1) * pad
    sheet_w = cols * IMAGE_WIDTH + (cols + 1) * pad
    contact_sheet = np.full((sheet_h, sheet_w, 3), 20, dtype=np.uint8)

    for i in range(min(12, len(frames_rgb))):
        r = i // cols
        c = i % cols
        y1 = pad + r * (IMAGE_HEIGHT + pad)
        x1 = pad + c * (IMAGE_WIDTH + pad)
        f_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        
        # Add timestamp watermark badge
        label = f"#{i+1} {timestamps_sec[i]:.2f}s"
        cv2.rectangle(f_bgr, (0, IMAGE_HEIGHT - 22), (90, IMAGE_HEIGHT), (0, 0, 0), -1)
        cv2.putText(f_bgr, label, (4, IMAGE_HEIGHT - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        contact_sheet[y1:y1 + IMAGE_HEIGHT, x1:x1 + IMAGE_WIDTH] = f_bgr

    sheet_path = job_artifacts_dir / "filmstrip.jpg"
    cv2.imwrite(str(sheet_path), contact_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    sheet_art_id = f"art_filmstrip_{job_id}"
    database.add_artifact(
        artifact_id=sheet_art_id,
        job_id=job_id,
        artifact_type="filmstrip",
        filename="filmstrip.jpg",
        storage_relative_path=f"artifacts/{job_id}/filmstrip.jpg",
        file_size_bytes=sheet_path.stat().st_size,
        mime_type="image/jpeg",
        file_hash_sha256=compute_file_sha256(sheet_path),
        metadata={"grid": "3x4", "total_frames": 12}
    )

    return {
        "thumbnail_artifact_id": thumb_art_id,
        "thumbnail_url": f"/api/jobs/{job_id}/artifacts/{thumb_art_id}",
        "filmstrip_artifact_id": sheet_art_id,
        "filmstrip_url": f"/api/jobs/{job_id}/artifacts/{sheet_art_id}",
        "frames": frame_artifacts
    }
