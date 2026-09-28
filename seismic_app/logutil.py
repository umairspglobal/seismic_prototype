"""Terminal logging for the Streamlit app, the API server and the pipeline.

Configured once so every stage prints progress to stderr (the terminal
where you ran `streamlit run app.py` or uvicorn). Streamlit's spinner
alone does not show what is happening underneath - model download,
weight load, tile N of M, forward pass, cache builds - so those steps
log here.

Set ``SEISMIC_LOG_FILE`` to also append every record to a file: ``1``
writes ``outputs/logs/server.log``, any other value is used as the path.
``SEISMIC_LOG_LEVEL=DEBUG`` includes per-request timings.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


_LOGGER_NAME = "seismic_app"
_configured = False
_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def _log_file_path() -> Path | None:
    raw = os.environ.get("SEISMIC_LOG_FILE", "").strip()
    if not raw or raw.lower() in ("0", "false", "no"):
        return None
    if raw.lower() in ("1", "true", "yes"):
        return Path(__file__).resolve().parents[1] / "outputs" / "logs" / "server.log"
    return Path(raw)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger that always writes to the terminal."""
    global _configured
    if not _configured:
        root = logging.getLogger(_LOGGER_NAME)
        level_name = os.environ.get("SEISMIC_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        root.setLevel(level)
        if not root.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
            root.addHandler(handler)
            log_file = _log_file_path()
            if log_file is not None:
                try:
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    file_handler = logging.FileHandler(log_file, encoding="utf-8")
                    file_handler.setLevel(level)
                    file_handler.setFormatter(
                        logging.Formatter(
                            "[%(asctime)s] %(process)d %(levelname)s %(name)s: %(message)s"
                        )
                    )
                    root.addHandler(file_handler)
                except OSError as exc:
                    print(f"Could not open log file {log_file}: {exc}", file=sys.stderr)
            root.propagate = False
        _configured = True
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)
