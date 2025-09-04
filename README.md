# GitHub AI Issue Assistant Setup Guide

This guide documents all steps required to set up, run, and use the GitHub AI Issue Assistant bot that posts AI-generated resolution steps as comments on new GitHub issues.

---

## 1. Prerequisites

- Python 3.7+
- [pip](https://pip.pypa.io/en/stable/)
- [ngrok](https://ngrok.com/download)
- Azure OpenAI resource (with a deployed model)
- GitHub repository where you want to use the bot

---

## 2. Clone the Repository

```sh
git clone <your-repo-url>
cd git-ai
```

---

## 3. Python Dependencies

Install required packages:

```sh
pip install fastapi uvicorn httpx python-dotenv PyGithub openai
```

---

## 4. Azure OpenAI Setup

- Go to Azure Portal → your OpenAI resource.
- Deploy a model (e.g., GPT-4, GPT-3.5).
- Copy your **API key** and **endpoint** from the resource's "Keys and Endpoint" section.

---

## 5. GitHub App Setup (for bot comments)

1. Go to [GitHub Apps](https://github.com/settings/apps) and create a new app.
2. Set permissions:
   - Repository permissions → Issues: **Read & write**
3. Set a webhook URL (can use `https://example.com/webhook` if not needed).
4. Install the app on your target repository.
5. Download the app's private key (`.pem` file).
6. Note your **App ID** and **Installation ID**.

---

## 6. Generate GitHub App Installation Token

Create `get_github_app_token.py`:

```python
from github import GithubIntegration

APP_ID = <your-app-id>
INSTALLATION_ID = <your-installation-id>
PRIVATE_KEY_PATH = "git-ai-bot.pem"

with open(PRIVATE_KEY_PATH, "r") as key_file:
    private_key = key_file.read()

git_integration = GithubIntegration(APP_ID, private_key)
access_token = git_integration.get_access_token(INSTALLATION_ID).token
print(access_token)
```

Run:

```sh
pip install PyGithub
python get_github_app_token.py
```

Copy the printed token.

---

## 7. Configure Environment Variables

Edit `.env`:

```properties
AZURE_OPENAI_API_KEY=<your-azure-openai-key>
AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
AZURE_OPENAI_DEPLOYMENT_NAME=<your-deployment-name>
GITHUB_TOKEN=<your-github-app-installation-token>
```

---

## 8. Start FastAPI Server

```sh
uvicorn main:app --reload
```

---

## 9. Expose Local Server with ngrok

```sh
ngrok http 8000
```

Copy the HTTPS forwarding URL.

---

## 10. Set Up GitHub Webhook

- Go to your repo → Settings → Webhooks → Add webhook.
- Payload URL: `<ngrok-forwarding-url>/webhook`
- Content type: `application/json`
- Select "Issues" events.

---

## 11. Usage

- Create a new issue in your GitHub repo.
- The bot will generate resolution steps using Azure OpenAI and post them as a comment (authored by your GitHub App/bot).
- All logs are saved in `gitai.log`.

---

## 12. Troubleshooting

- Check `gitai.log` for detailed logs and errors.
- Ensure all environment variables are set correctly.
- Make sure your Azure OpenAI deployment is running.
- The GitHub App must have "Issues: Read & write" permission and be installed on the repo.

---

## 13. Customization

- Edit prompts and fallback logic in `ai_service.py`.
- Change logging settings in `main.py`.

---

## 14. Credits

- Built with FastAPI, Azure OpenAI, GitHub Apps, and ngrok.

---

For questions or help, contact the repository maintainer.