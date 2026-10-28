import requests

def test_whatsapp_link():
    print("Testing WhatsApp Link Endpoint...")
    resp = requests.get("http://127.0.0.1:8000/api/whatsapp/link?event=Fire&confidence=0.95&video_identifier=video.mp4")
    print(f"Status: {resp.status_code}")
    try:
        data = resp.json()
        print(f"Response: {data}")
    except Exception as e:
        print(f"Response text: {resp.text}")

if __name__ == "__main__":
    test_whatsapp_link()
