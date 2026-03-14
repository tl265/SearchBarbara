import os
import uvicorn


if __name__ == "__main__":
    host = os.getenv("BIND_HOST", "127.0.0.1")
    port = int(os.getenv("BIND_PORT", "8000"))
    ssl_cert = os.getenv("SSL_CERTFILE", "")
    ssl_key = os.getenv("SSL_KEYFILE", "")
    # Agent debug logs are written to logs/agent_debug.log automatically
    # when deep_research_agent module is imported.
    # Set AGENT_LOG_LEVEL=INFO or DEBUG to also see them in the console.
    print(f"[info] Agent debug log: logs/agent_debug.log")
    ssl_kwargs = {}
    if ssl_cert and ssl_key:
        ssl_kwargs["ssl_certfile"] = ssl_cert
        ssl_kwargs["ssl_keyfile"] = ssl_key
        print(f"[info] HTTPS enabled: cert={ssl_cert} key={ssl_key}")
    uvicorn.run("app.main:app", host=host, port=port, reload=False, **ssl_kwargs)
