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
APP_ID = int(os.getenv("GITHUB_APP_ID", "0"))
INSTALLATION_ID = int(os.getenv("GITHUB_INSTALLATION_ID", "0"))
PRIVATE_KEY_PATH = os.getenv("GITHUB_PRIVATE_KEY_PATH", "git-ai-bot.pem")
