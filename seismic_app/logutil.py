"""Terminal logging for the Streamlit app and pipeline.

Configured once so every stage prints progress to stderr (the terminal
where you ran `streamlit run app.py`). Streamlit's spinner alone does
not show what is happening underneath - model download, weight load,
tile N of M, forward pass - so those steps log here.
"""

from __future__ import annotations

import logging
import sys


_LOGGER_NAME = "seismic_app"
_configured = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger that always writes to the terminal."""
    global _configured
    if not _configured:
        root = logging.getLogger(_LOGGER_NAME)
        root.setLevel(logging.INFO)
        if not root.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            root.addHandler(handler)
            root.propagate = False
        _configured = True
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)
