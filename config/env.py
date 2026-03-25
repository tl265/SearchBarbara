import os


def get_env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


# ---------------------------------------------------------------------------
# SMTP / email notification settings (optional)
# ---------------------------------------------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
EMAIL_FROM_ADDRESS = os.getenv("EMAIL_FROM_ADDRESS", "report@searchbarbara.com")
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "SearchBarbara")
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
