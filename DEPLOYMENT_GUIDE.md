# Audio-Visual Scene Analysis - Production Deployment Guide
==============================================================================

This guide provides step-by-step instructions to deploy the entire Audio-Visual Scene Analysis system to the cloud:
- **Backend API & ML Inference Engine:** Deployed on **[Render](https://render.com/)**
- **Frontend Dashboard & UI:** Deployed on **[Netlify](https://www.netlify.com/)**

---

## 1. Project Deployment Architecture

```
                                +-------------------------------+
                                |        Netlify Hosting        |
                                |  (Frontend Single-Page App)   |
                                |   https://your-app.netlify.app |
                                +---------------+---------------+
                                                |
                              REST API Requests |  (VITE_API_BASE_URL)
                              & Video Uploads   |
                                                v
                                +---------------+---------------+
                                |        Render Hosting         |
                                |  (FastAPI Backend + ML Engine)|
                                |  https://your-api.onrender.com |
                                +---------------+---------------+
                                                |
                         +----------------------+----------------------+
                         |                      |                      |
                         v                      v                      v
                 Keras Multimodal       FFmpeg Audio/Video      SQLite WAL Database
                   (6.45M Params)           Extractor            & Managed Storage
```

---

## 2. Directory & Configuration Structure

The project has been prepared with full monorepo support for both Render and Netlify:

```
├── .gitignore                   <- Prevents venvs and caches; tracks model weights
├── render.yaml                  <- Render Infrastructure Blueprint (1-Click Deploy)
├── netlify.toml                 <- Root Netlify Monorepo build and SPA routing config
├── requirements.txt             <- Root Python dependencies pointer
├── best_multimodal_model.keras  <- Pre-trained multimodal neural network weights (65.4 MB)
│
├── backend/                     <- Backend Web Service (Target: Render)
│   ├── main.py                  <- FastAPI API server entry point
│   ├── database.py              <- SQLite persistence with WAL mode
│   ├── migrations.py            <- Database schema migrations
│   ├── requirements.txt         <- Production Python dependencies
│   ├── render-build.sh          <- Render build script (installs ffmpeg + packages)
│   ├── Dockerfile               <- Production container configuration (Optional)
│   ├── .env.example             <- Environment variables template
│   ├── best_multimodal_model.keras <- Local model replica for subfolder builds
│   ├── services/                <- Inference, audio/video extraction, speech, reports
│   └── storage/                 <- uploads/, artifacts/, reports/, temp/
│
└── frontend/                    <- Frontend React Dashboard (Target: Netlify)
    ├── package.json             <- Vite & React dependencies
    ├── vite.config.js           <- Dev proxy configuration
    ├── netlify.toml             <- Directory-level Netlify configuration
    ├── .env.example             <- Frontend environment variables template
    ├── public/
    │   └── _redirects           <- Netlify SPA routing rewrite (/* -> /index.html 200)
    └── src/
        ├── api.js               <- Unified API client (supports VITE_API_BASE_URL)
        ├── components/          <- FrameViewer, WaveformViewer, DetectionPopup
        └── pages/               <- 8 Dashboard views (Studio, Jobs, Model, Evaluation...)
```

---

## 3. Part 1: Deploy Backend on Render

### Step 1: Push Project to GitHub
Initialize your Git repository and push it to GitHub:
```bash
git init
git add .
git commit -m "feat: complete production-ready audio-visual scene analysis system"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
git push -u origin main
```
> [!NOTE]
> `best_multimodal_model.keras` is **65.4 MB**, which is well within GitHub's standard 100 MB single-file limit without needing Git LFS.

---

### Step 2: Create a Web Service on Render

#### Option A: 1-Click Deploy using Render Blueprint (Recommended)
1. Log in to [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** > **Blueprint**.
3. Connect your GitHub repository.
4. Render automatically reads [`render.yaml`](file:///c:/Users/Hp/OneDrive/Desktop/hphphphphhphph/render.yaml) and pre-configures all build settings, environment variables, and health checks.
5. Click **Apply**.

#### Option B: Manual Web Service Setup
1. Log in to [Render Dashboard](https://dashboard.render.com/) and click **New +** > **Web Service**.
2. Select your repository.
3. Configure the following fields:
   - **Name:** `av-scene-analysis-backend` (or your choice)
   - **Region:** Choose the region closest to you (e.g., Singapore, Frankfurt, Oregon)
   - **Branch:** `main`
   - **Root Directory:** `backend`
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     chmod +x render-build.sh && ./render-build.sh
     ```
   - **Start Command:**
     ```bash
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type:** `Standard` (at least 2 GB RAM recommended for TensorFlow/Whisper/ConvNeXt)

4. Scroll to **Environment Variables** and add:
   | Key | Value | Description |
   | :--- | :--- | :--- |
   | `PYTHON_VERSION` | `3.11.9` | Ensures Python 3.11 runtime |
   | `TF_CPP_MIN_LOG_LEVEL` | `3` | Suppresses verbose TensorFlow C++ logs |
   | `ALLOWED_ORIGINS` | `*` | Or specify your Netlify domain |
   | `WHATSAPP_TEST_MODE` | `true` | Prevents unauthorized real emergency alerts |
   | `WHATSAPP_TEST_NUMBER`| `+916360553165` | Your verified test alert phone number |
   | `STORAGE_RETENTION_DAYS` | `30` | Auto-purges job storage older than 30 days |
   | `MAX_STORAGE_MB` | `5000` | Disk threshold for cleanup daemon |

5. Scroll to **Advanced** > **Health Check Path** and set:
   ```
   /api/health
   ```
6. Click **Create Web Service**.

---

### Step 3: Verify Your Render Backend
Once the build completes and status is **Live**, open your browser and navigate to:
```
https://<YOUR-RENDER-SERVICE-NAME>.onrender.com/api/health
```
You should receive a `200 OK` JSON response:
```json
{
  "status": "healthy",
  "healthy": true,
  "checks": {
    "database": "ok",
    "multimodal_model": "loaded",
    "visual_backbone": "loaded",
    "text_encoder": "loaded",
    "asr_pipeline": "loaded",
    "storage": "writable"
  }
}
```
**Copy your Render Backend URL** (e.g. `https://av-scene-analysis-backend.onrender.com`). You will need it for Netlify.

---

## 4. Part 2: Deploy Frontend on Netlify

### Step 1: Create a Site on Netlify
1. Log in to [Netlify Dashboard](https://app.netlify.com/).
2. Click **Add new site** > **Import an existing project**.
3. Select **GitHub** and authorize access to your repository.
4. Select your project repository.

---

### Step 2: Configure Build Settings
Netlify will automatically detect [`netlify.toml`](file:///c:/Users/Hp/OneDrive/Desktop/hphphphphhphph/netlify.toml), but verify the settings:

- **Base directory:** `frontend`
- **Build command:** `npm run build`
- **Publish directory:** `frontend/dist` (or `dist` if base is set to `frontend`)
- **Functions directory:** *(leave blank)*

---

### Step 3: Add Environment Variables
1. Under **Environment variables**, click **Add a variable**.
2. Set the key and value:
   - **Key:** `VITE_API_BASE_URL`
   - **Value:** `https://<YOUR-RENDER-SERVICE-NAME>.onrender.com` *(your live Render backend URL from Part 1, without trailing slash)*

---

### Step 4: Deploy
1. Click **Deploy site**.
2. Netlify will build the production bundle (`dist/`) and deploy it to a global CDN.
3. Once deployed, Netlify will provide your live URL:
   ```
   https://<YOUR-SITE-NAME>.netlify.app
   ```

---

## 5. Post-Deployment Verification Checklist

Test the complete end-to-end cloud pipeline:

1. **Dashboard Home Page:**
   - Open `https://<YOUR-SITE-NAME>.netlify.app`.
   - Verify the top navigation bar displays **System Status: Healthy** with a green pulse.
2. **Model Details Page:**
   - Navigate to `/model`.
   - Verify architecture specifications (6,455,846 parameters, 3-class Softmax, SHA256 integrity hash).
3. **Analysis Studio Video Upload:**
   - Navigate to `/studio`.
   - Upload a test video (MP4/AVI).
   - Verify:
     - Real-time progress tracker updates through 0% -> 25% -> 50% -> 75% -> 100%.
     - Predicted class badge appears (**Normal**, **Fire**, or **Accident**).
     - 12 extracted frames render in the Frame Viewer with top-3 influential highlights.
     - 64-bin Log-Mel Spectrogram renders with Time and Frequency axes.
     - Kannada ASR transcript and grounded scene description are generated.
4. **Historical Jobs Page:**
   - Navigate to `/jobs`.
   - Confirm your analyzed video appears in the table.
   - Click to download the generated JSON and Markdown reports.

---

## 6. Pro Tips & Troubleshooting

### Render Free Tier Cold Starts
If using Render's free tier, the web service spins down after 15 minutes of inactivity. The first request after spindown may take 45–60 seconds while the VM boots and loads the Keras model.
- The frontend includes built-in retry logic and connection warning indicators.
- To prevent spindown, you can use a free uptime monitoring service like [UptimeRobot](https://uptimerobot.com/) to ping `https://<YOUR-RENDER-APP>.onrender.com/api/health` every 10 minutes.

### CORS Errors
The backend [main.py](file:///c:/Users/Hp/OneDrive/Desktop/hphphphphhphph/backend/main.py#L156-L163) permits all origins (`*`) by default. If you prefer strict security, set the Render environment variable:
```env
ALLOWED_ORIGINS=https://<YOUR-SITE-NAME>.netlify.app
```

### Direct Video Artifact URL Resolution
The frontend API client ([api.js](file:///c:/Users/Hp/OneDrive/Desktop/hphphphphhphph/frontend/src/api.js#L7-L12)) dynamically maps all relative paths (`/api/jobs/.../artifacts/...`) directly to your Render backend via `VITE_API_BASE_URL`.
