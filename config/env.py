import os


def get_env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()
