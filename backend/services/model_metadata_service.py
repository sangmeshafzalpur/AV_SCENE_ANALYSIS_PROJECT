import os
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = WORKSPACE_ROOT / "best_multimodal_model.keras"
if not MODEL_PATH.exists():
    MODEL_PATH = BACKEND_DIR / "best_multimodal_model.keras"
if not MODEL_PATH.exists():
    MODEL_PATH = Path(r"c:\Users\Hp\OneDrive\Desktop\AVS2\model\best_multimodal_model.keras")
if not MODEL_PATH.exists():
    MODEL_PATH = WORKSPACE_ROOT / "kannada_av_scene_model.keras"

METADATA_JSON_PATH = BACKEND_DIR / "model_metadata.json"

_CACHED_METADATA: Optional[Dict[str, Any]] = None

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def build_model_metadata() -> Dict[str, Any]:
    global _CACHED_METADATA
    if _CACHED_METADATA is not None:
        return _CACHED_METADATA

    file_exists = MODEL_PATH.exists()
    file_size = MODEL_PATH.stat().st_size if file_exists else None
    sha256_hash = compute_sha256(MODEL_PATH) if file_exists else None

    metadata = {
        "metadata_version": "2.0.0",
        "model_identification": {
            "canonical_name": "EfficientNetB0_BiLSTM_AudioCNN_KannadaASR_TextBiLSTM",
            "active_model_filename": "best_multimodal_model.keras",
            "active_model_path": str(MODEL_PATH),
            "file_exists": file_exists,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2) if file_size else None,
            "sha256_checksum": sha256_hash,
            "sister_model_reference": {
                "filename": "kannada_av_scene_model.keras",
                "known_location": str(WORKSPACE_ROOT / "kannada_av_scene_model.keras"),
                "known_sha256": "2ec2af07c9ef1d061575e8a48f0fadd741abd0f6f35c10b7399aa9815ed4cb42",
                "known_size_bytes": 13171748,
                "architecture_note": "3-input embedding-fusion model (ConvNeXt-V2 Tiny + SentenceTransformer embeddings)."
            }
        },
        "verified_facts": {
            "model_format": "Keras 3 (.keras zip archive containing metadata.json, config.json, model.weights.h5)",
            "runtime_frameworks": {
                "tensorflow": "2.21.0",
                "keras": "3.15.1",
                "torch": "2.14.0",
                "transformers": "5.17.0"
            },
            "parameters": {
                "total_parameters": 6455846,
                "trainable_parameters": 2406275,
                "non_trainable_parameters": 4049571,
                "measurement_method": "Measured directly via tf.keras.models.load_model on best_multimodal_model.keras"
            },
            "input_tensors": [
                {
                    "name": "video",
                    "shape": [None, 12, 224, 224, 3],
                    "dtype": "float32",
                    "source_modality": "Visual",
                    "description": "12 uniform RGB video frames (224x224) processed by TimeDistributed(EfficientNetB0) -> BiLSTM(128) -> Dense(128)"
                },
                {
                    "name": "audio",
                    "shape": [None, 64, 313, 1],
                    "dtype": "float32",
                    "source_modality": "Acoustic",
                    "description": "64-mel bin standardized log-mel spectrogram over 5.0s (313 frames) processed by 3x Conv2D/BN/Pool -> GAP -> Dense(128)"
                },
                {
                    "name": "audio_mask",
                    "shape": [None, 1],
                    "dtype": "float32",
                    "source_modality": "Acoustic Gating",
                    "description": "Acoustic gating indicator (1.0 = audio available, 0.0 = silent/missing) gating the audio feature representation via Dense(128, sigmoid)"
                },
                {
                    "name": "text",
                    "shape": [None],
                    "dtype": "string",
                    "source_modality": "Speech/Language",
                    "description": "Kannada transcription string processed by internal TextVectorization -> Embedding -> BiLSTM(128) -> Dense(128)"
                }
            ],
            "output_tensor": {
                "name": "event_prediction",
                "shape": [None, 3],
                "dtype": "float32",
                "activation": "softmax",
                "semantic_meaning": "Softmax score distribution over [Accident, Fire, Normal]"
            },
            "class_mapping": {
                "0": "Accident",
                "1": "Fire",
                "2": "Normal",
                "verification_sources": [
                    "AVS2/model/class_mapping.json ({\"Accident\": 0, \"Fire\": 1, \"Normal\": 2})",
                    "AVS2/scripts/train_multimodal.py (CLASS_NAMES = [\"Accident\", \"Fire\", \"Normal\"])",
                    "backend/services/model_inference_service.py (CLASS_INDEX_MAP)"
                ]
            },
            "architecture_layers": {
                "total_layers": 26,
                "visual_branch": [
                    "video (InputLayer, shape=[None, 12, 224, 224, 3])",
                    "time_distributed (TimeDistributed(EfficientNetB0), params=4,049,571)",
                    "bidirectional (Bidirectional(LSTM 128, return_sequences=False), params=1,424,384)",
                    "dense (Dense 128 ReLU, params=32,896)"
                ],
                "audio_branch": [
                    "audio (InputLayer, shape=[None, 64, 313, 1])",
                    "conv2d (Conv2D 32 filters 3x3, params=320)",
                    "batch_normalization (BatchNormalization, params=128)",
                    "max_pooling2d (MaxPooling2D 2x2)",
                    "conv2d_1 (Conv2D 64 filters 3x3, params=18,496)",
                    "batch_normalization_1 (BatchNormalization, params=256)",
                    "max_pooling2d_1 (MaxPooling2D 2x2)",
                    "conv2d_2 (Conv2D 128 filters 3x3, params=73,856)",
                    "batch_normalization_2 (BatchNormalization, params=512)",
                    "global_average_pooling2d (GlobalAveragePooling2D)",
                    "dense_1 (Dense 128 ReLU, params=16,512)",
                    "audio_mask (InputLayer, shape=[None, 1])",
                    "dense_2 (Dense 128 Sigmoid gate, params=256)",
                    "multiply (Multiply acoustic representation with audio_mask gate)"
                ],
                "speech_branch": [
                    "text (InputLayer, shape=[None], dtype=string)",
                    "text_vectorization (TextVectorization, vocab=2000, max_tokens=100)",
                    "embedding (Embedding 2000 -> 128, params=256,000)",
                    "bidirectional_1 (Bidirectional(LSTM 128, return_sequences=False), params=197,632)",
                    "dense_3 (Dense 128 ReLU, params=16,512)"
                ],
                "fusion_and_head": [
                    "concatenate (Concatenate visual(128) + audio(128) + speech(128) -> 384 units)",
                    "dense_4 (Dense 256 ReLU, params=98,560)",
                    "dropout (Dropout rate=0.3)",
                    "dense_5 (Dense 128 ReLU, params=32,896)",
                    "event_prediction (Dense 3 Softmax, params=387)"
                ]
            }
        },
        "configured_values": {
            "inference_pipeline": {
                "num_sampled_frames": 12,
                "frame_resolution": [224, 224],
                "visual_backbone_model": "EfficientNetB0 (embedded TimeDistributed)",
                "audio_sample_rate_hz": 16000,
                "audio_fixed_seconds": 5.0,
                "audio_mel_bins": 64,
                "audio_n_fft": 1024,
                "audio_hop_length": 256,
                "audio_time_steps": 313,
                "text_vectorization_vocab_size": 2000,
                "text_vectorization_max_len": 100,
                "asr_model_id": "vasista22/whisper-kannada-small",
                "asr_forced_language": "kn",
                "asr_chunk_seconds": 30,
                "asr_max_length": 448
            },
            "training_hyperparameters": {
                "training_dataset_size": 1080,
                "training_samples_per_class": 360,
                "blind_test_dataset_size": 270,
                "blind_test_samples_per_class": 90,
                "training_epochs": 20,
                "random_seed": 42
            }
        },
        "measured_results": {
            "source": "backup_extracted/av_scene_project/results/final_model_results_summary.csv",
            "blind_test_evaluation": {
                "total_test_videos": 270,
                "correct_predictions": 265,
                "incorrect_predictions": 5,
                "accuracy": 0.9814814814814815,
                "macro_precision": 0.9815203523068692,
                "macro_recall": 0.9814814814814815,
                "macro_f1": 0.9814809099046267,
                "expected_calibration_error": 0.0121596636296296,
                "multiclass_brier_score": 0.0295598023074867
            },
            "class_performance": {
                "Fire": {
                    "actual_samples": 90,
                    "predicted_samples": 89,
                    "correct": 87,
                    "incorrect": 3,
                    "precision": 0.977528,
                    "recall": 0.966667,
                    "f1_score": 0.972067,
                    "misclassifications": "3 Fire samples predicted as Accident"
                },
                "Accident": {
                    "actual_samples": 90,
                    "predicted_samples": 91,
                    "correct": 88,
                    "incorrect": 2,
                    "precision": 0.967033,
                    "recall": 0.977778,
                    "f1_score": 0.972376,
                    "misclassifications": "2 Accident samples predicted as Fire"
                },
                "Normal": {
                    "actual_samples": 90,
                    "predicted_samples": 90,
                    "correct": 90,
                    "incorrect": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1_score": 1.0,
                    "misclassifications": "0 Normal samples misclassified"
                }
            },
            "confusion_matrix": {
                "Actual_Fire": {"Predicted_Fire": 87, "Predicted_Accident": 3, "Predicted_Normal": 0},
                "Actual_Accident": {"Predicted_Fire": 2, "Predicted_Accident": 88, "Predicted_Normal": 0},
                "Actual_Normal": {"Predicted_Fire": 0, "Predicted_Accident": 0, "Predicted_Normal": 90}
            }
        },
        "unknown_or_untested_information": {
            "english_audio_accuracy": "Untested. The current ASR pipeline forces Kannada (kn) and rejects non-Kannada speech (<70% Kannada characters), passing a zero speech vector (1, 384).",
            "modality_ablation_baselines": "Unmeasured. Visual-Only, Audio-Only, Speech-Only, and 2-way ablations are marked as 'Not yet evaluated' in ablation_study_results.csv.",
            "low_light_performance": "Not quantified in test ground truth datasets.",
            "cctv_compression_resilience": "Minimum required bitrate and resolution below 224p not experimentally validated.",
            "concurrent_throughput_limits": "Peak inference requests per second before GPU/CPU starvation not formally profiled."
        },
        "limitations": [
            "Fixed 12-frame uniform sampling may miss brief sub-second incidents occurring between sampled points in long videos.",
            "Speech recognition fails to provide semantic signals for English utterances due to the strict 70% Kannada character Unicode check.",
            "Requires host installation of ffmpeg on system PATH for audio extraction.",
            "Native Windows TensorFlow does not utilize DirectML/CUDA GPU acceleration in TF >= 2.11; Keras fusion model runs on CPU."
        ]
    }

    _CACHED_METADATA = metadata
    return metadata

def save_metadata_document() -> Path:
    metadata = build_model_metadata()
    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return METADATA_JSON_PATH

def check_model_readiness(models_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metadata = build_model_metadata()
    file_exists = metadata["model_identification"]["file_exists"]

    components = {
        "model_file": {
            "status": "ready" if file_exists else "missing",
            "path": metadata["model_identification"]["active_model_path"],
            "size_bytes": metadata["model_identification"]["file_size_bytes"],
            "sha256": metadata["model_identification"]["sha256_checksum"]
        },
        "multimodal_keras_model": {
            "status": "loaded" if (
                (models_dict and "multimodal_model" in models_dict and models_dict["multimodal_model"] is not None)
            ) else "unloaded",
            "target": metadata["model_identification"]["canonical_name"]
        },
        "visual_backbone": {
            "status": "loaded" if (models_dict and "visual_model" in models_dict and models_dict["visual_model"] is not None) else "unloaded",
            "target": "facebook/convnextv2-tiny-1k-224"
        },
        "text_encoder": {
            "status": "loaded" if (models_dict and "text_encoder" in models_dict and models_dict["text_encoder"] is not None) else "unloaded",
            "target": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        },
        "asr_pipeline": {
            "status": "loaded" if (models_dict and "asr_pipe" in models_dict and models_dict["asr_pipe"] is not None) else "unloaded",
            "target": "vasista22/whisper-kannada-small"
        }
    }

    all_loaded = all(c["status"] == "loaded" for k, c in components.items() if k != "model_file")
    is_ready = file_exists and all_loaded

    return {
        "status": "ready" if is_ready else "degraded",
        "ready": is_ready,
        "model_name": metadata["model_identification"]["canonical_name"],
        "class_mapping": metadata["verified_facts"]["class_mapping"],
        "components": components
    }
