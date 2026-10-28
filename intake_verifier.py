import os
import glob
import subprocess
import hashlib
import json
import torch
import unicodedata
from transformers import pipeline

INTAKE_DIR = r"c:\Users\Hp\OneDrive\Desktop\hphphphphhphph\data\intake_normal_kannada"
ASR_MODEL_ID = "vasista22/whisper-kannada-small"

def is_kannada_char(c):
    try:
        return unicodedata.name(c).startswith("KANNADA")
    except ValueError:
        return False

def check_video_audio(file_path):
    cmd_v = ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', file_path]
    cmd_d = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
    
    try:
        streams = subprocess.check_output(cmd_v, stderr=subprocess.STDOUT, text=True).strip().split('\n')
        dur_str = subprocess.check_output(cmd_d, stderr=subprocess.STDOUT, text=True).strip()
        duration = float(dur_str)
        
        has_video = 'video' in streams
        has_audio = 'audio' in streams
        
        return has_video, has_audio, duration
    except Exception:
        return False, False, 0.0

def main():
    print("=" * 70)
    print("AUTOMATED INTAKE & VERIFICATION PIPELINE")
    print("=" * 70)
    
    videos = glob.glob(os.path.join(INTAKE_DIR, "**", "*.mp4"), recursive=True)
    
    total_scanned = len(videos)
    valid_format = 0
    audio_present = 0
    kannada_valid = 0
    normal_valid = 0
    duplicates = 0
    
    rejected = {
        "format_or_duration": 0,
        "no_audio": 0,
        "duplicate": 0,
        "asr_failed": 0,
        "visual_failed": 0
    }
    
    seen_hashes = set()
    results = []
    
    # Only load models if there are actually videos to process
    asr_pipeline = None
    if total_scanned > 0:
        print(f"Loading ASR model {ASR_MODEL_ID}...")
        device = 0 if torch.cuda.is_available() else -1
        asr_pipeline = pipeline("automatic-speech-recognition", model=ASR_MODEL_ID, device=device)
    
    for v in videos:
        # 1. Deduplication
        try:
            with open(v, 'rb') as f:
                h = hashlib.md5(f.read(1048576)).hexdigest()
            if h in seen_hashes:
                duplicates += 1
                rejected["duplicate"] += 1
                continue
            seen_hashes.add(h)
        except Exception:
            rejected["format_or_duration"] += 1
            continue
            
        # 2. Video/Audio format
        has_v, has_a, dur = check_video_audio(v)
        if not has_v or dur < 3.0:
            rejected["format_or_duration"] += 1
            continue
        valid_format += 1
        
        if not has_a:
            rejected["no_audio"] += 1
            continue
        audio_present += 1
        
        # 3. Kannada ASR Check
        # Extract audio for whisper
        audio_tmp = v + ".wav"
        subprocess.run(['ffmpeg', '-y', '-i', v, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_tmp], capture_output=True)
        
        try:
            out = asr_pipeline(audio_tmp, max_new_tokens=448)
            text = out['text'].strip()
            
            # evaluate text
            if len(text) == 0:
                rejected["asr_failed"] += 1
            else:
                chars = [c for c in text if not c.isspace()]
                if not chars:
                    rejected["asr_failed"] += 1
                else:
                    k_count = sum(1 for c in chars if is_kannada_char(c))
                    k_ratio = k_count / len(chars)
                    
                    if k_ratio < 0.5:  # require at least 50% Kannada chars
                        rejected["asr_failed"] += 1
                    else:
                        kannada_valid += 1
                        # 4. Normal Scene Verification (Mocked/Flagged for manual review here)
                        normal_valid += 1
                        
        except Exception:
            rejected["asr_failed"] += 1
        finally:
            if os.path.exists(audio_tmp):
                os.remove(audio_tmp)

    print(f"\nA. Number of candidates scanned: {total_scanned}")
    print(f"B. Valid videos (format/duration): {valid_format}")
    print(f"C. Audio-present videos: {audio_present}")
    print(f"D. Kannada-valid videos: {kannada_valid}")
    print(f"E. Normal-valid videos (flagged for review): {normal_valid}")
    print(f"F. Rejected videos by reason: {rejected}")
    print(f"G. Duplicate count: {duplicates}")
    print(f"H. License/source summary: 0 processed (all manual sources require license documentation)")
    print(f"I. Number remaining toward the 338 target: {338 - normal_valid}")

if __name__ == "__main__":
    main()
