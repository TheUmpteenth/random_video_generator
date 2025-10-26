# random_video_generator/utils.py
from __future__ import annotations
import logging

VERSION = "1.1.1"

def setup_logging(verbosity: str = "info"):
    """Configure global logging level and format."""
    level = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }.get(verbosity.lower(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )