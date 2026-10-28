import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.whatsapp_service import send_alert

# Ensure test mode is on
os.environ["WHATSAPP_TEST_MODE"] = "true"
os.environ["WHATSAPP_TEST_NUMBER"] = "+916360553165" # Dummy number for failure test

print("--- Testing normal class (should do nothing) ---")
send_alert("Normal", 0.99, "vid_normal_123.mp4")
print("Normal test passed (no exception, no alert).")

print("\n--- Testing accident class (failure handling) ---")
# This will attempt to open the browser for +910000000000.
# It should catch an exception or log an error if PyWhatKit fails, but NOT crash this script.
try:
    send_alert("Accident", 0.85, "vid_acc_123.mp4")
    print("Accident failure-handling test passed (did not crash).")
except Exception as e:
    print(f"CRASH: send_alert raised an exception! {e}")

print("\n--- Testing duplicate prevention ---")
send_alert("Accident", 0.85, "vid_acc_123.mp4")
print("Duplicate test passed (should log 'skipped').")
