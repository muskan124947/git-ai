import os
import logging
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

def setup_logger():
    logging.basicConfig(
        filename="gitai.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    return logging.getLogger("gitai")

# --- Azure OpenAI ---
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# --- GitHub App ---
github_app_id = os.getenv("GITHUB_APP_ID")
github_installation_id = os.getenv("GITHUB_INSTALLATION_ID")

if not github_app_id:
    raise ValueError("GITHUB_APP_ID environment variable is not set or is empty")
if not github_installation_id:
    raise ValueError("GITHUB_INSTALLATION_ID environment variable is not set or is empty")

APP_ID = int(github_app_id)
INSTALLATION_ID = int(github_installation_id)
PRIVATE_KEY_PATH = os.getenv("GITHUB_PRIVATE_KEY_PATH", "git-ai-bot-test.2025-09-04.private-key.pem")
