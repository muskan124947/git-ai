import requests
import json

# Sample GitHub issue webhook payload
sample_payload = {
    "action": "opened",
    "issue": {
        "number": 123,
        "title": "Application crashes when loading large files",
        "body": "Hi! I'm experiencing crashes when trying to load files larger than 100MB. The app freezes and then closes without any error message.\n\nSteps to reproduce:\n1. Open the application\n2. Try to load a file > 100MB\n3. App crashes\n\nExpected: File should load normally\nActual: App crashes\n\nEnvironment:\n- OS: Windows 11\n- Version: 2.1.0",
        "html_url": "https://github.com/test-user/test-repo/issues/123",
        "created_at": "2024-01-15T10:30:00Z",
        "user": {
            "login": "test-user",
            "type": "User"
        },
        "labels": [
            {"name": "bug"},
            {"name": "high-priority"}
        ]
    },
    "repository": {
        "full_name": "test-user/test-repo",
        "description": "A sample application for file processing",
        "language": "Python",
        "topics": ["file-processing", "python", "desktop-app"]
    }
}

def test_webhook():
    webhook_url = "http://localhost:8000/webhook"
    
    try:
        print("🚀 Testing Git.AI webhook with sample GitHub issue...")
        print(f"📡 Sending POST request to: {webhook_url}")
        
        response = requests.post(
            webhook_url,
            json=sample_payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("✅ Webhook call successful!")
            result = response.json()
            print(f"📝 Response: {json.dumps(result, indent=2)}")
            
            if "resolution_draft" in result:
                print(f"\n🤖 AI Generated Resolution Draft:")
                print("-" * 50)
                print(result["resolution_draft"])
                print("-" * 50)
        else:
            print(f"❌ Webhook call failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed!")
        print("Make sure your FastAPI server is running:")
        print("uvicorn main:app --reload --port 8000")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_webhook()
