# Model-Intelligence Foundation & Architecture Specification
**Project:** Audio-Visual Scene Analysis for Kannada and English Videos using Deep Learning  
**Canonical Model Name:** `Kannada_AV_Multimodal_CrossModal_Model`  
**Active Artifact:** [`kannada_av_scene_model.keras`](file:///c:/Users/Hp/OneDrive/Desktop/hphphphphhphph/kannada_av_scene_model.keras)  
**Machine-Readable Metadata:** [`backend/model_metadata.json`](file:///c:/Users/Hp/OneDrive/Desktop/hphphphphhphph/backend/model_metadata.json)

---

## 1. Verified Model Facts

### 1.1 Artifact Provenance & Integrity
* **Active Filename:** `kannada_av_scene_model.keras`
* **File Size:** `13,171,748 bytes` (12.56 MB)
* **SHA-256 Checksum:** `2ec2af07c9ef1d061575e8a48f0fadd741abd0f6f35c10b7399aa9815ed4cb42`
* **Format:** Keras 3 Zip Archive (`metadata.json`, `config.json`, `model.weights.h5`)
* **External Referenced Model:** `best_multimodal_model.keras` located on Desktop (`c:\Users\Hp\OneDrive\Desktop\AVS2\model\best_multimodal_model.keras`, 65,410,165 bytes, SHA-256 `404f374484b2304e44074dd44a81a8130add1b22c22dcd7a7e21ea56f8763fde`) — an earlier 4-input prototype not utilized in current runtime.

### 1.2 Input & Output Signatures
| Tensor Name | Shape | Dtype | Modality Source | Description |
| :--- | :--- | :--- | :--- | :--- |
| `visual_input` | `(None, 12, 768)` | `float32` | Video Frames | 12 uniform video frames encoded as 768-d pooled vectors using ConvNeXt-V2 Tiny. |
| `audio_input` | `(None, 64, 313, 1)` | `float32` | Audio Spectrogram | 64 mel bins × 313 time frames normalized log-mel spectrogram across 5.0s audio. |
| `speech_input` | `(None, 384)` | `float32` | Kannada Speech | 384-d normalized sentence embedding encoded via Paraphrase Multilingual MiniLM L12 v2. |
| **Output:** `classifier` | `(None, 3)` | `float32` | Scene Prediction | Softmax probabilities corresponding to `[Fire, Accident, Normal]`. |

### 1.3 Verified Class-to-Index Mapping
* **Index `0`:** `Fire`
* **Index `1`:** `Accident`
* **Index `2`:** `Normal`

### 1.4 Measured Parameter Count & Layers
* **Total Parameters:** `1,082,755`
* **Trainable Parameters:** `1,082,307`
* **Non-trainable Parameters:** `448` (BatchNormalization moving mean/variance)
* **Total Model Layers:** `33`

---

## 2. Layer-by-Layer Architectural Specification

```
                          MULTIMODAL FUSION ARCHITECTURE
 
   [ Visual Input ]            [ Audio Input ]             [ Speech Input ]
    (None, 12, 768)          (None, 64, 313, 1)              (None, 384)
          │                           │                           │
   Dense(256)                 Conv2D(32, 3x3)              Dense(256)
          │                   BatchNorm + MaxPool                 │
   LayerNorm                          │                    LayerNorm
          │                   Conv2D(64, 3x3)                     │
   MultiHeadAttention(4)      BatchNorm + MaxPool                 │
          │                           │                           │
     Residual + LayerNorm     Conv2D(128, 3x3)                    │
          │                   BatchNorm                           │
   GlobalAvgPool1D                    │                           │
          │                   GlobalAvgPool2D                     │
   Dense(256)                         │                           │
          │                   Dense(256)                          │
          ▼                           ▼                           ▼
   Visual Token [1, 256]       Audio Token [1, 256]       Speech Token [1, 256]
          └───────────────────────────┼───────────────────────────┘
                                      ▼
                      Concatenate Tokens: (None, 3, 256)
                                      │
                     Cross-Modal MultiHeadAttention(4)
                                      │
                     Residual Add + LayerNormalization
                                      │
                           GlobalAveragePooling1D
                                      │
                                  Dense(256)
                                      │
                                 Dropout(0.3)
                                      │
                                Dense(3, Softmax)
                                      │
                                      ▼
                       [ Fire, Accident, Normal ]
```

---

## 3. Configured Preprocessing Requirements

* **Visual Preprocessing:**
  - Uniform sampling: `12` frames per video (`get_uniform_frame_indices`).
  - Spatial resolution: `224 × 224` pixels RGB (`IMAGE_HEIGHT = 224, IMAGE_WIDTH = 224`).
  - Backbone normalization: `AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")`.
* **Acoustic Preprocessing:**
  - Audio extraction: Single-channel (mono) PCM at `16,000 Hz` via `ffmpeg`.
  - Temporal duration: `5.0` seconds (`AUDIO_FIXED_SECONDS = 5.0`).
  - Multi-window averaging: If audio duration exceeds 5s, up to 4 windows are uniformly sampled and averaged.
  - Spectrogram parameters: `n_mels = 64`, `n_fft = 1024`, `hop_length = 256`.
  - Normalization: `(logmel - mean) / (std + 1e-6)`.
* **Speech Recognition & Embedding:**
  - ASR Pipeline: `vasista22/whisper-kannada-small` with forced decoder language `"kn"`.
  - Quality thresholds: `duration >= 0.5s`, `word_count >= 2`, `kannada_char_ratio >= 0.70`, `words_per_second <= 5.0`.
  - Text Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` generating 384-dimensional unit-normalized embeddings.
  - Speech Fallback: If quality check fails or audio is silent, defaults to zero vector `(1, 384)`.

---

## 4. Measured Evaluation Results (Official Ground Truth)
*Source: `backup_extracted/av_scene_project/results/final_model_results_summary.csv`*

* **Blind-Test Dataset:** 270 videos (90 Fire, 90 Accident, 90 Normal)
* **Overall Accuracy:** `98.15%` (265 / 270 correct)
* **Macro Precision:** `98.15%`
* **Macro Recall:** `98.15%`
* **Macro F1-score:** `98.15%`
* **Expected Calibration Error (ECE):** `0.01216`
* **Class Confusion Matrix:**
  - **Fire (90):** 87 Correct, 3 misclassified as Accident.
  - **Accident (90):** 88 Correct, 2 misclassified as Fire.
  - **Normal (90):** 90 Correct, 0 misclassified.

---

## 5. Explicit Unknowns & Limitations

1. **English Speech Handling:** English audio accuracy has **not been evaluated**. The current pipeline's `kannada_ratio >= 0.70` check rejects English speech and passes a zero embedding `(1, 384)`.
2. **Sub-Modality Ablation Performance:** Ablation experiments (Visual Only, Audio Only, etc.) are recorded as `"Not yet evaluated"` in the original results registry.
3. **Environmental Extremes:** Severe compression below 240p, extreme nighttime darkness, and heavy rain/fog scenarios have not been benchmarked.
