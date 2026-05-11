import requests
import json

def test_openclaw_intent(gesture_text):
    # The correct v1 API endpoint for OpenClaw
    url = "http://localhost:18789/api/v1/messages" 
    
    # Passing your VIP token so OpenClaw accepts the command
    headers = {
        "Authorization": "Bearer ea79ae0c29794888059971d76f17b786279ba71821fb2ae1",
        "Content-Type": "application/json"
    }
    
    prompt = f"The user performed a gesture that translates to: '{gesture_text}'. What system action should be taken? Reply ONLY in JSON format like {{'action': '...', 'target': '...'}}."
    
    # The updated payload structure required by v2026.3.22
    payload = {
        "content": prompt,
        "channel": "api",
        "user_id": "yolov10-system"
    }
    
    print(f"Sending gesture intent: '{gesture_text}' to OpenClaw Sandbox...")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print("\n--- AI Response Received ---")
            print(json.dumps(result, indent=2))
            print("----------------------------\n")
            print("Success! Your OpenCV script can now map this JSON to an actual pyautogui action.")
        else:
            print(f"Agent returned an error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Failed to reach the OpenClaw container: {e}")

# Simulate your YOLOv10 script sending a command
test_openclaw_intent("open google chrome")