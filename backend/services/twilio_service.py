import os
import logging
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from datetime import datetime

logger = logging.getLogger("twilio_service")
logging.basicConfig(level=logging.INFO)

_processed_events = set()

def get_twilio_client():
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        raise ValueError("Missing Twilio credentials in environment.")
    return Client(account_sid, auth_token)

def get_recipient_phone(event_type: str) -> str:
    test_mode = os.getenv("TWILIO_TEST_MODE", "false").lower() == "true"
    
    if test_mode:
        if event_type == "Fire":
            return os.getenv("TWILIO_TEST_FIRE_PHONE")
        elif event_type == "Accident":
            return os.getenv("TWILIO_TEST_AMBULANCE_PHONE")
    else:
        if event_type == "Fire":
            return os.getenv("FIRE_DEPARTMENT_PHONE")
        elif event_type == "Accident":
            return os.getenv("AMBULANCE_PHONE")
            
    return None

def trigger_twilio_call(event: str, confidence: float, video_identifier: str, event_id: str):
    if event not in ["Fire", "Accident"]:
        return {"success": False, "error": "Invalid event type."}

    action_key = f"call_{event_id}"
    if action_key in _processed_events:
        return {"success": False, "error": "Call already initiated for this event."}

    recipient = get_recipient_phone(event)
    if not recipient:
        return {"success": False, "error": f"Recipient phone not configured for {event}."}

    try:
        client = get_twilio_client()
        from_phone = os.getenv("TWILIO_PHONE_NUMBER")
        
        twiml_message = f'<Response><Say>Emergency alert. The video analysis system has detected a possible {event.lower()} incident. The AI confidence is {int(confidence*100)} percent. Please verify and respond immediately.</Say></Response>'
        
        call = client.calls.create(
            twiml=twiml_message,
            to=recipient,
            from_=from_phone
        )
        
        _processed_events.add(action_key)
        return {"success": True, "event": event, "call_sid": call.sid}
    except TwilioRestException as e:
        logger.error(f"Twilio Call Error: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error initiating call: {e}")
        return {"success": False, "error": "Internal server error."}

def trigger_twilio_message(event: str, confidence: float, video_identifier: str, event_id: str):
    if event not in ["Fire", "Accident"]:
        return {"success": False, "error": "Invalid event type."}

    action_key = f"msg_{event_id}"
    if action_key in _processed_events:
        return {"success": False, "error": "Message already sent for this event."}

    recipient = get_recipient_phone(event)
    if not recipient:
        return {"success": False, "error": f"Recipient phone not configured for {event}."}

    try:
        client = get_twilio_client()
        from_phone = os.getenv("TWILIO_PHONE_NUMBER")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message_body = (
            f"🚨 AI VIDEO ALERT\n\n"
            f"Event: {event}\n"
            f"Confidence: {confidence:.2%}\n"
            f"Video: {video_identifier}\n"
            f"Time: {timestamp}\n\n"
            f"The system detected a possible {event} incident.\n"
            f"Please verify the situation and respond accordingly."
        )
        
        message = client.messages.create(
            body=message_body,
            to=recipient,
            from_=from_phone
        )
        
        _processed_events.add(action_key)
        return {"success": True, "event": event, "message_sid": message.sid}
    except TwilioRestException as e:
        logger.error(f"Twilio Message Error: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return {"success": False, "error": "Internal server error."}
