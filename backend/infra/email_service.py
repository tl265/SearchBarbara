"""
Fire-and-forget email notification service.

Sends an email with the research report when generation completes.
All errors are caught internally so that email failures never affect
the main research workflow.
"""

import logging
import os
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

log = logging.getLogger("searchbarbara.email_service")

# ---------------------------------------------------------------------------
# Lazy-imported config (avoids circular imports at module level)
# ---------------------------------------------------------------------------

def _cfg():
    """Return SMTP config values from config.env (imported lazily)."""
    from config.env import (
        EMAIL_ENABLED,
        SMTP_HOST,
        SMTP_PORT,
        SMTP_USERNAME,
        SMTP_PASSWORD,
        SMTP_USE_TLS,
        EMAIL_FROM_ADDRESS,
        EMAIL_FROM_NAME,
    )
    return {
        "enabled": EMAIL_ENABLED,
        "host": SMTP_HOST,
        "port": SMTP_PORT,
        "username": SMTP_USERNAME,
        "password": SMTP_PASSWORD,
        "use_tls": SMTP_USE_TLS,
        "from_address": EMAIL_FROM_ADDRESS,
        "from_name": EMAIL_FROM_NAME,
    }


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "email"


def _render_html(
    report_title: str,
    preview_text: str,
    download_url: str,
) -> str:
    """Render the HTML email body from a template file, or fall back to inline."""
    template_path = _TEMPLATE_DIR / "report_ready.html"
    try:
        raw = template_path.read_text(encoding="utf-8")
        html = (
            raw.replace("{{ report_title }}", _html_escape(report_title))
            .replace("{{ preview_text }}", _html_escape(preview_text))
            .replace("{{ download_url }}", _html_escape(download_url))
        )
        return html
    except Exception as exc:
        log.warning("Email template rendering failed, using inline fallback: %s", exc)
        return _inline_fallback_html(report_title, preview_text, download_url)


def _html_escape(text: str) -> str:
    """Minimal HTML escaping for template values."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _inline_fallback_html(
    report_title: str, preview_text: str, download_url: str
) -> str:
    title = _html_escape(report_title)
    preview = _html_escape(preview_text)
    return f"""\
<html>
<body style="font-family:sans-serif;color:#333;">
<h2>Your SearchBarbara Report is Ready</h2>
<p><strong>{title}</strong></p>
<p>{preview}</p>
<p><a href="{download_url}"
       style="display:inline-block;padding:10px 20px;background:#2563eb;color:#fff;
              text-decoration:none;border-radius:6px;">
   Download Full Report
</a></p>
<p style="font-size:12px;color:#888;">
  The complete Markdown report is also attached to this email.
</p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# SMTP sending (blocking, called in daemon thread)
# ---------------------------------------------------------------------------

def _is_configured() -> bool:
    cfg = _cfg()
    return bool(cfg["enabled"] and cfg["host"])


def _build_message(
    to_addr: str,
    report_markdown: str,
    report_title: str,
    run_id: str,
) -> MIMEMultipart:
    cfg = _cfg()
    public_base = str(os.getenv("PUBLIC_BASE_URL", "")).strip().rstrip("/")
    download_url = f"{public_base}/api/runs/{run_id}/report/download"

    preview_text = report_markdown[:500].strip()
    if len(report_markdown) > 500:
        preview_text += " ..."

    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"Report Ready: {report_title[:80]}"
    msg["From"] = (
        f"{cfg['from_name']} <{cfg['from_address']}>"
        if cfg["from_name"]
        else cfg["from_address"]
    )
    msg["To"] = to_addr

    # HTML body
    html_body = _render_html(report_title, preview_text, download_url)
    html_part = MIMEText(html_body, "html", "utf-8")
    msg.attach(html_part)

    # Markdown attachment
    md_attachment = MIMEText(report_markdown, "plain", "utf-8")
    md_attachment.add_header(
        "Content-Disposition", "attachment", filename="report.md"
    )
    msg.attach(md_attachment)

    return msg


def _send_smtp(msg: MIMEMultipart) -> None:
    """Blocking SMTP send. Must be called in a background thread."""
    cfg = _cfg()
    host = cfg["host"]
    port = cfg["port"]
    try:
        if cfg["use_tls"]:
            server = smtplib.SMTP(host, port, timeout=30)
            server.ehlo()
            server.starttls()
            server.ehlo()
        else:
            server = smtplib.SMTP(host, port, timeout=30)
            server.ehlo()
        if cfg["username"] and cfg["password"]:
            server.login(cfg["username"], cfg["password"])
        server.sendmail(msg["From"], [msg["To"]], msg.as_string())
        server.quit()
        log.info("Report email sent successfully to %s", msg["To"])
    except Exception as exc:
        log.error("SMTP send failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_report_email(
    to_addr: Optional[str],
    report_markdown: str,
    report_title: str,
    run_id: str,
) -> None:
    """
    Fire-and-forget email notification.

    Immediately returns. The actual sending happens in a daemon thread.
    All exceptions are caught internally and logged — this function
    **never** raises or blocks the caller.
    """
    try:
        if not _is_configured():
            log.debug("Email notification skipped: EMAIL_ENABLED is false or SMTP not configured.")
            return
        if not to_addr or not str(to_addr).strip():
            log.warning("Email notification skipped: no recipient address available.")
            return
        to_addr = str(to_addr).strip()
        msg = _build_message(to_addr, report_markdown, report_title, run_id)

        def _worker():
            try:
                _send_smtp(msg)
            except Exception as exc:
                log.error("Email worker thread failed: %s", exc)

        t = threading.Thread(target=_worker, daemon=True, name="email-send")
        t.start()
    except Exception as exc:
        log.error("send_report_email outer error (suppressed): %s", exc)
