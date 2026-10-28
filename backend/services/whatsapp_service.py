import os
import sys
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Set
import pywhatkit

# Ensure parent directory is in path for database imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whatsapp_service")

# In-memory deduplication cache
_sent_alerts: Set[str] = set()
_lock = threading.Lock()

# Thread pool executor for non-blocking asynchronous dispatch
_thread_pool = []

def get_whatsapp_config() -> Dict[str, Any]:
    """Retrieves current WhatsApp alert configuration and threshold rules."""
    return {
        "test_mode": os.getenv("WHATSAPP_TEST_MODE", "false").lower() == "true",
        "test_number": os.getenv("WHATSAPP_TEST_NUMBER", "").strip(),
        "fire_number": os.getenv("FIRE_WHATSAPP_NUMBER", "").strip(),
        "accident_number": os.getenv("ACCIDENT_WHATSAPP_NUMBER", "").strip(),
        "confidence_threshold": float(os.getenv("WHATSAPP_CONFIDENCE_THRESHOLD", "0.40")),
        "cooldown_minutes": int(os.getenv("WHATSAPP_COOLDOWN_MINUTES", "60")),
        "allowed_classes": ["Fire", "Accident"]
    }

def send_alert(
    prediction: str,
    confidence: float,
    video_identifier: str,
    job_id: Optional[str] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Sends a WhatsApp emergency warning with configurable alert rules,
    duplicate prevention, delivery status tracking, and explicit test mode safeguards.
    Clearly labels messages as AI-generated warnings.
    Notification failures never raise exceptions to avoid breaking the prediction workflow.
    """
    config = get_whatsapp_config()

    # Rule 1: Class Filter (Only Fire and Accident trigger alerts; Normal is skipped)
    if prediction not in config["allowed_classes"]:
        logger.info(f"WhatsApp Alert skipped: Normal prediction does not trigger emergency notifications ({prediction}).")
        if job_id:
            database.record_dispatch(
                job_id=job_id,
                dispatch_type="whatsapp",
                event_type=prediction,
                recipient="none",
                status="skipped_normal"
            )
        return {
            "status": "skipped_normal",
            "message": "Normal predictions do not trigger alerts.",
            "sent": False
        }

    # Rule 2: Confidence Threshold Gating
    if confidence < config["confidence_threshold"]:
        logger.info(f"WhatsApp Alert skipped: Confidence {confidence:.2%} is below threshold {config['confidence_threshold']:.2%}")
        if job_id:
            database.record_dispatch(
                job_id=job_id,
                dispatch_type="whatsapp",
                event_type=prediction,
                recipient="none",
                status="skipped_below_threshold"
            )
        return {
            "status": "skipped_below_threshold",
            "message": f"Confidence {confidence:.2%} below threshold {config['confidence_threshold']:.2%}",
            "sent": False
        }

    # Determine Recipient with Strict Test Mode Safeguards
    test_mode = config["test_mode"]
    test_number = config["test_number"]

    if test_mode:
        if not test_number:
            logger.error("WHATSAPP_TEST_MODE is enabled, but WHATSAPP_TEST_NUMBER is missing. Suppressing alert to protect real recipients.")
            if job_id:
                database.record_dispatch(
                    job_id=job_id,
                    dispatch_type="whatsapp",
                    event_type=prediction,
                    recipient="unconfigured_test_number",
                    status="skipped_no_test_number",
                    error_message="WHATSAPP_TEST_MODE active without WHATSAPP_TEST_NUMBER configured"
                )
            return {
                "status": "skipped_no_test_number",
                "message": "Test mode active without test recipient configured. Alert suppressed.",
                "sent": False
            }
        recipient = test_number
        logger.info(f"TEST MODE ACTIVE: Routing {prediction} alert strictly to test number {recipient} (NEVER to real services)")
    else:
        if prediction == "Fire":
            recipient = config["fire_number"]
        elif prediction == "Accident":
            recipient = config["accident_number"]
        else:
            recipient = None

    if not recipient:
        err_msg = f"Missing recipient phone number for {prediction} alert in environment configuration."
        logger.error(err_msg)
        if job_id:
            database.record_dispatch(
                job_id=job_id,
                dispatch_type="whatsapp",
                event_type=prediction,
                recipient="missing",
                status="failed_missing_recipient",
                error_message=err_msg
            )
        return {
            "status": "failed_missing_recipient",
            "message": err_msg,
            "sent": False
        }

    # Rule 3: Duplicate Alert Prevention
    alert_key = f"{prediction}_{video_identifier}"
    with _lock:
        is_mem_dup = alert_key in _sent_alerts

    is_db_dup = False
    try:
        is_db_dup = database.has_recent_dispatch(
            event_type=prediction,
            recipient=recipient,
            job_id=job_id,
            window_minutes=config["cooldown_minutes"]
        )
    except Exception as e:
        logger.warning(f"Database duplicate check check failed: {e}")

    if (is_mem_dup or is_db_dup) and not force:
        logger.info(f"WhatsApp Alert skipped: Duplicate alert for {alert_key} within {config['cooldown_minutes']}m window.")
        if job_id:
            database.record_dispatch(
                job_id=job_id,
                dispatch_type="whatsapp",
                event_type=prediction,
                recipient=recipient,
                status="skipped_duplicate",
                external_id=alert_key
            )
        return {
            "status": "skipped_duplicate",
            "message": f"Duplicate alert for {alert_key} suppressed.",
            "sent": False
        }

    # Rule 4: Clear AI Warning Labeling
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode_tag = "[TEST MODE - SIMULATED ALERT]\n" if test_mode else ""

    message = (
        f"{mode_tag}⚠️ AI-GENERATED WARNING (MODEL NOTIFICATION)\n\n"
        f"Event Type: {prediction.upper()}\n"
        f"Model Confidence: {confidence:.2%} (Uncalibrated Softmax Score)\n"
        f"Video Identifier: {video_identifier}\n"
        f"Incident Timestamp: {timestamp}\n\n"
        f"IMPORTANT NOTICE:\n"
        f"This is an automated notification produced by a multimodal deep learning model. "
        f"It is NOT a confirmed emergency and requires independent visual and physical verification.\n"
        f"Do not deploy first responders without secondary confirmation."
    )

    logger.info(f"Dispatching {prediction} WhatsApp alert to {recipient}...")
    try:
        pywhatkit.sendwhatmsg_instantly(
            phone_no=recipient,
            message=message,
            wait_time=15,
            tab_close=True,
            close_time=3
        )
        logger.info(f"WhatsApp alert successfully delivered to {recipient}.")
        with _lock:
            _sent_alerts.add(alert_key)

        if job_id:
            database.record_dispatch(
                job_id=job_id,
                dispatch_type="whatsapp",
                event_type=prediction,
                recipient=recipient,
                status="success",
                external_id=alert_key
            )

        return {
            "status": "success",
            "recipient": recipient,
            "prediction": prediction,
            "confidence": confidence,
            "sent": True,
            "test_mode": test_mode
        }
    except Exception as e:
        err_str = str(e)
        logger.error(f"WhatsApp sending failed for {prediction}. Error: {err_str}")
        if job_id:
            try:
                database.record_dispatch(
                    job_id=job_id,
                    dispatch_type="whatsapp",
                    event_type=prediction,
                    recipient=recipient,
                    status="failed",
                    external_id=alert_key,
                    error_message=err_str
                )
            except Exception:
                pass

        return {
            "status": "failed",
            "error": err_str,
            "recipient": recipient,
            "sent": False
        }

def send_alert_async(
    prediction: str,
    confidence: float,
    video_identifier: str,
    job_id: Optional[str] = None,
    force: bool = False
):
    """Executes send_alert in a background daemon thread to avoid blocking caller."""
    t = threading.Thread(
        target=send_alert,
        args=(prediction, confidence, video_identifier, job_id, force),
        daemon=True,
        name=f"whatsapp-worker-{job_id or 'adhoc'}"
    )
    t.start()
    return t
