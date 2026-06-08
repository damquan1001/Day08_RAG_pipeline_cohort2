from __future__ import annotations

import sys
from pathlib import Path


GROUP_PROJECT_DIR = Path(__file__).resolve().parent
if str(GROUP_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(GROUP_PROJECT_DIR))

from src.module_chat_ui.streamlit_app import run_app


if __name__ == "__main__":
    run_app()
