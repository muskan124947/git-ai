try:
    import openai
    import os
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("\n📦 Please install dependencies first:")
    print("pip install openai python-dotenv")
    print("\nOr install from requirements.txt:")
    print("pip install -r requirements.txt")
    exit(1)

load_dotenv()

# Configure Azure OpenAI
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-12-01-preview"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

def test_azure_openai():
    # Check if environment variables are loaded
    if not openai.api_key or not openai.api_base:
        print("❌ Missing Azure OpenAI credentials!")
        print("Please check your .env file contains:")
        print("- AZURE_OPENAI_API_KEY")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_DEPLOYMENT_NAME")
        return
    
    # Debug information
    print(f"🔗 Testing connection to: {openai.api_base}")
    print(f"🔑 API Key (first 10 chars): {openai.api_key[:10]}...")
    print(f"🚀 Using deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
    print(f"📅 API Version: {openai.api_version}")
    print(f"🏷️  API Type: {openai.api_type}")
    
    # Validate endpoint format
    if not openai.api_base.endswith('/'):
        print("⚠️  Adding trailing slash to endpoint...")
        openai.api_base = openai.api_base + '/'
    
    try:
        print("\n🔄 Making API call...")
        response = openai.ChatCompletion.create(
            engine=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo-deployment"),
            messages=[
                {"role": "user", "content": "Hello! Can you help me test Azure OpenAI?"}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        print("✅ Azure OpenAI is working!")
        print(f"Response: {response.choices[0].message.content}")
        
    except openai.error.AuthenticationError as e:
        print(f"❌ Authentication Error: {str(e)}")
        print("\n🔍 Possible solutions:")
        print("1. Regenerate your API key in Azure Portal > Your Resource > Keys and Endpoint")
        print("2. Copy the KEY 1 value (not KEY 2)")
        print("3. Make sure you're using the correct resource")
        
    except openai.error.InvalidRequestError as e:
        print(f"❌ Invalid Request Error: {str(e)}")
        print("\n🔍 Possible solutions:")
        print("1. Check deployment name matches exactly in Azure OpenAI Studio")
        print("2. Ensure the model deployment status is 'Succeeded'")
        print("3. Try using 'gpt-35-turbo' instead of 'gpt-35-turbo-deployment'")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"❌ Error Type: {type(e).__name__}")
        print("\n🔍 General troubleshooting:")
        print("1. Verify your Azure subscription is active")
        print("2. Check if your resource region matches the endpoint")
        print("3. Ensure you have proper permissions on the Azure OpenAI resource")
        print("4. Try accessing Azure OpenAI Studio to confirm access")

if __name__ == "__main__":
    test_azure_openai()
