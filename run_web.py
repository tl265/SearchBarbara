import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from infra.observability import setup_global_logger
from infra.observability.cleanup import cleanup_old_session_logs

load_dotenv(Path(__file__).resolve().parent / ".env")

if __name__ == "__main__":
    log = setup_global_logger()
    host = os.getenv("BIND_HOST", "127.0.0.1")
    port = int(os.getenv("BIND_PORT", "8000"))
    ssl_cert = os.getenv("SSL_CERTFILE", "")
    ssl_key = os.getenv("SSL_KEYFILE", "")

    # Clean up old session logs on startup
    cleanup_old_session_logs()

    log.info("Global log: logs/searchbarbara.log")
    log.info("Session logs: logs/sessions/")
    ssl_kwargs = {}
    if ssl_cert and ssl_key:
        ssl_kwargs["ssl_certfile"] = ssl_cert
        ssl_kwargs["ssl_keyfile"] = ssl_key
        log.info("HTTPS enabled: cert=%s key=%s", ssl_cert, ssl_key)
    uvicorn.run("app.main:app", host=host, port=port, reload=False, **ssl_kwargs)
