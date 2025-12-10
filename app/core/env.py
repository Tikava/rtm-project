from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_env(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file if present.
    Does nothing if file is missing.
    """
    path = Path(env_file) if env_file else Path(".env")
    if path.exists():
        load_dotenv(dotenv_path=path)
