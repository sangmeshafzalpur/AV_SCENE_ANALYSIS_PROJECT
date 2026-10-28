# Audio-Visual Scene Analysis System (AV-Scene-Analysis)

An end-to-end production Multimodal Audio-Visual Scene Analysis intelligence platform powered by deep learning fusion architectures, FastAPI backend, and modern React dashboard.

---

## 🌟 Key Features

- **Multimodal Deep Learning Architecture**: Joint visual-audio feature extraction using deep neural fusion.
- **Real-Time Scene Inference**: Predicts audio-visual scene classes with high confidence calibration.
- **Audio Spectrogram & Waveform Analysis**: Visual inspection of audio frequency patterns and decibel distributions.
- **Influential Frame Attribution**: Explainable AI mechanism attributing scene classifications to critical video keyframes.
- **Kannada Speech Recognition (ASR)**: Native Kannada speech transcription and contextual scene description generation.
- **Emergency Alert Dispatcher**: Integrated Twilio SMS and WhatsApp notifications for critical/emergency scene alerts.
- **Enterprise Storage Manager**: Secure SHA-256 hashed storage layer with path traversal guards.
- **Interactive Web Dashboard**: React + Vite interface with real-time job polling, metrics monitoring, and reports.

---

## 🏗️ Architecture

```
├── backend/
│   ├── main.py                          # FastAPI API Gateway & Job Orchestrator
│   ├── database.py                      # SQLite database schema & connection pool
│   ├── best_multimodal_model.keras      # Trained production deep learning model
│   ├── model_metadata.json              # Model configuration & class mapping
│   ├── services/
│   │   ├── model_inference_service.py   # Multimodal model inference runner
│   │   ├── video_preprocessor.py        # Frame sampling & visual feature extraction
│   │   ├── audio_preprocessor.py        # Mel-spectrogram & audio feature extraction
│   │   ├── speech_recognition_service.py# Audio transcription service
│   │   ├── scene_description_service.py # Natural scene description generation
│   │   ├── frame_attribution_service.py # Explainability & frame saliency
│   │   ├── storage_manager.py           # Hardened local file & artifact storage
│   │   ├── whatsapp_service.py          # WhatsApp alert dispatcher
│   │   ├── twilio_service.py            # Twilio SMS dispatcher
│   │   └── reporting_service.py         # Multi-format report generator
│   └── tests/
│
├── frontend/
│   ├── src/
│   │   ├── pages/                       # Studio, Overview, History, Models, Monitor
│   │   ├── components/                  # Spectrogram, Attribution, Audio/Video Players
│   │   └── api.js                       # Axios API client with retry & polling
│   └── package.json
│
├── DEPLOYMENT_GUIDE.md                  # Complete deployment manual (Render + Netlify)
├── MODEL_INTELLIGENCE.md                # Model architecture & evaluation benchmarks
└── STORAGE_ARCHITECTURE.md              # Storage pipeline specifications
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ & npm
- FFmpeg installed and available on system `PATH`

### 2. Backend Setup
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🧪 Verification & Testing
Run backend test suites:
```bash
python backend/test_backend_api_engine.py
python backend/test_production_model_integration.py
```

---

## 📄 License
MIT License.