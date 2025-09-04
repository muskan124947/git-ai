# main.py
from fastapi import FastAPI, Request
from ai_service import AIService
from dotenv import load_dotenv
import os
import logging
from github import GithubIntegration
from sentence_transformers import SentenceTransformer, util
from webhook_handler import process_github_webhook

# Load environment variables
load_dotenv()

app = FastAPI()
ai_service = AIService()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
def root():
    return {"message": "Git.AI backend is running!"}

# Read GitHub App credentials from environment or config
APP_ID = int(os.getenv("GITHUB_APP_ID", "1883127"))  # Set your App ID here or in .env
INSTALLATION_ID = int(os.getenv("GITHUB_INSTALLATION_ID", "83932345"))  # Set your Installation ID here or in .env
PRIVATE_KEY_PATH = os.getenv("GITHUB_PRIVATE_KEY_PATH", "git-ai-bot.pem")  # Set your .pem path here or in .env



@app.post("/webhook")
async def github_webhook(request: Request):
    """Handle GitHub webhook events."""
    return await process_github_webhook(request, ai_service, logger)
