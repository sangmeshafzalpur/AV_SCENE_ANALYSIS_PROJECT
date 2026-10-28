#!/usr/bin/env bash
# Exit immediately on error
set -o errexit

echo "=== 1. Upgrading Pip and Installing Python Packages ==="
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "=== 2. Ensuring FFmpeg Binary Available ==="
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not detected in system PATH. Downloading Linux static release..."
    mkdir -p bin
    curl -L -o ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    tar -xf ffmpeg.tar.xz --strip-components=1 -C bin
    rm ffmpeg.tar.xz
    export PATH="$PWD/bin:$PATH"
    echo "Static FFmpeg installed to $(pwd)/bin"
else
    echo "FFmpeg is already available: $(which ffmpeg)"
fi

echo "=== 3. Ensuring Managed Storage Directories Exist ==="
mkdir -p storage/uploads storage/artifacts storage/reports storage/temp

echo "=== 4. Verifying Pre-Trained Model Weights ==="
if [ -f "best_multimodal_model.keras" ]; then
    echo "Verified model weights: best_multimodal_model.keras"
elif [ -f "../best_multimodal_model.keras" ]; then
    echo "Copying model file from root workspace..."
    cp ../best_multimodal_model.keras ./best_multimodal_model.keras
fi

echo "=== Build Finished Successfully ==="
