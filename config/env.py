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
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() == "true"
EMAIL_FROM_ADDRESS = os.getenv("EMAIL_FROM_ADDRESS", "report@searchbarbara.com")
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "SearchBarbara")
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"


# ---------------------------------------------------------------------------
# ASR / dictation settings (optional)
# ---------------------------------------------------------------------------
ASR_MODEL = get_env("ASR_MODEL", "gpt-4o-transcribe") or "gpt-4o-transcribe"
ASR_TIMEOUT_SEC = max(5, int(get_env("ASR_TIMEOUT_SEC", "30") or "30"))
ASR_SILENCE_TIMEOUT_MS = max(
    1000, int(get_env("ASR_SILENCE_TIMEOUT_MS", "5000") or "5000")
)
ASR_PARTIAL_INTERVAL_MS = max(
    600, int(get_env("ASR_PARTIAL_INTERVAL_MS", "600") or "600")
)
ASR_MAX_SESSION_SEC = max(
    10, int(get_env("ASR_MAX_SESSION_SEC", "120") or "120")
)
ASR_CALIBRATION_MS = max(
    0, int(get_env("ASR_CALIBRATION_MS", "150") or "150")
)
ASR_SPEECH_START_HOLD_MS = max(
    0, int(get_env("ASR_SPEECH_START_HOLD_MS", "80") or "80")
)
ASR_SPEECH_END_HOLD_MS = max(
    0, int(get_env("ASR_SPEECH_END_HOLD_MS", "700") or "700")
)
ASR_SPEECH_THRESHOLD_MULTIPLIER = min(
    10.0,
    max(
        1.1,
        float(
            get_env("ASR_SPEECH_THRESHOLD_MULTIPLIER", "2.4") or "2.4"
        ),
    ),
)
ASR_MIN_SPEECH_RMS = min(
    0.2,
    max(0.005, float(get_env("ASR_MIN_SPEECH_RMS", "0.04") or "0.04")),
)
ASR_MAX_AMBIENT_RMS = min(
    0.2,
    max(0.005, float(get_env("ASR_MAX_AMBIENT_RMS", "0.025") or "0.025")),
)
