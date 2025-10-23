# random_video_generator/media_loader.py
from __future__ import annotations
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VIDEO_EXTS: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".webm")
AUDIO_EXTS: Tuple[str, ...] = (".mp3", ".wav", ".m4a", ".aac", ".ogg")

def list_files(dirpath: str, exts: Tuple[str, ...]) -> List[str]:
    p = Path(dirpath)
    if not p.exists() or not p.is_dir():
        logging.warning("Media folder missing or not a directory: %s", dirpath)
        return []
    files = [str(fp.resolve()) for fp in p.iterdir() if fp.suffix.lower() in exts]
    random.shuffle(files)
    return files
