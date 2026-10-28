import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_twilio():
    print("Testing Twilio Endpoints...")
    
    # 1. Normal Test (Should Fail / Be Rejected)
    print("\n--- Testing Normal (Should Reject) ---")
    resp = requests.post(f"{BASE_URL}/api/emergency/call", json={
        "event": "Normal",
        "confidence": 0.99,
        "video_identifier": "test.mp4",
        "event_id": "test_normal_1"
    })
    print(f"Normal Call Response: {resp.status_code}")
    assert resp.status_code == 400

    # 2. Fire Test
    print("\n--- Testing Fire Call ---")
    resp = requests.post(f"{BASE_URL}/api/emergency/call", json={
        "event": "Fire",
        "confidence": 0.97,
        "video_identifier": "fire_test.mp4",
        "event_id": "test_fire_1"
    })
    print(f"Fire Call Response: {resp.status_code} {resp.text}")

    print("\n--- Testing Fire Message ---")
    resp = requests.post(f"{BASE_URL}/api/emergency/message", json={
        "event": "Fire",
        "confidence": 0.97,
        "video_identifier": "fire_test.mp4",
        "event_id": "test_fire_1"
    })
    print(f"Fire Msg Response: {resp.status_code} {resp.text}")

    # 3. Accident Test
    print("\n--- Testing Accident Call ---")
    resp = requests.post(f"{BASE_URL}/api/emergency/call", json={
        "event": "Accident",
        "confidence": 0.96,
        "video_identifier": "acc_test.mp4",
        "event_id": "test_acc_1"
    })
    print(f"Accident Call Response: {resp.status_code} {resp.text}")
    
    print("\n--- Testing Duplicate Prevention ---")
    resp = requests.post(f"{BASE_URL}/api/emergency/call", json={
        "event": "Accident",
        "confidence": 0.96,
        "video_identifier": "acc_test.mp4",
        "event_id": "test_acc_1"
    })
    print(f"Duplicate Call Response: {resp.status_code} {resp.text}")

if __name__ == "__main__":
    test_twilio()
