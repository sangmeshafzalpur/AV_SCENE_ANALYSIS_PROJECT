import os
import sys
import time
import platform
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from database import get_stats as get_db_stats

START_TIME = time.time()
START_ISO = datetime.now().isoformat()
BACKEND_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BACKEND_DIR / "storage"

def get_directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total

def get_system_operational_stats(models_dict: Dict[str, Any] = None, job_queue_size: int = 0) -> Dict[str, Any]:
    current_time = time.time()
    uptime_seconds = round(current_time - START_TIME, 1)

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_rss_mb = round(mem_info.rss / (1024 * 1024), 2)
    mem_vms_mb = round(mem_info.vms / (1024 * 1024), 2)

    db_stats = get_db_stats()
    storage_size_bytes = get_directory_size(STORAGE_DIR)

    models_loaded = {
        "multimodal_model": bool(models_dict and models_dict.get("multimodal_model")),
        "visual_model": bool(models_dict and models_dict.get("visual_model")),
        "text_encoder": bool(models_dict and models_dict.get("text_encoder")),
        "asr_pipe": bool(models_dict and models_dict.get("asr_pipe")),
    }
    all_ready = all(models_loaded.values())

    return {
        "status": "operational" if all_ready else "degraded",
        "started_at": START_ISO,
        "uptime_seconds": uptime_seconds,
        "uptime_human": f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s",
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "pid": os.getpid()
        },
        "resources": {
            "process_memory_rss_mb": mem_rss_mb,
            "process_memory_vms_mb": mem_vms_mb,
            "system_memory_percent": psutil.virtual_memory().percent,
            "storage_used_bytes": storage_size_bytes,
            "storage_used_mb": round(storage_size_bytes / (1024 * 1024), 2)
        },
        "models_status": {
            "all_loaded": all_ready,
            "components": models_loaded
        },
        "job_engine": {
            "queue_pending": job_queue_size,
            "database_metrics": db_stats
        }
    }
