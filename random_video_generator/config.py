# random_video_generator/config.py
from __future__ import annotations
import json
import logging
import os
from typing import Dict

def load_json_with_hint(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as e:
        hint = (
            "JSON parse error at line %d column %d: %s.\n"
            "Common cause on Windows: use forward slashes (/) in file paths\n"
            "or escape backslashes (\\\\) in strings inside config.json.\n"
            "Example: 'C:/path/to/file.png' or 'C:\\\\path\\\\to\\\\file.png'."
            % (e.lineno, e.colno, e.msg)
        )
        raise ValueError(hint) from e

def load_config(path: str) -> Dict:
    raw = load_json_with_hint(path)

    for sec in ("audio", "images", "videos", "clips", "output"):
        if sec not in raw:
            raise ValueError(f"Missing required config section: '{sec}'")

    # audio.path
    audio_path = raw["audio"].get("path")
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio path missing or not found: {audio_path}")

    # images.path
    img_dir = raw["images"].get("path")
    if not img_dir or not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images dir missing or not found: {img_dir}")

    # videos.path may be optional; warn
    vid_dir = raw["videos"].get("path")
    if not vid_dir or not os.path.isdir(vid_dir):
        logging.warning(f"Videos dir missing or not found: {vid_dir} — videos will be skipped")

    clips = raw["clips"]
    for k in ("min_still", "desired_still", "max_still"):
        if k not in clips:
            raise ValueError(f"Missing clips.{k} in config")
    min_still = float(clips["min_still"])
    desired_still = float(clips["desired_still"])
    max_still = float(clips["max_still"])
    if not (0 < min_still <= desired_still <= max_still):
        raise ValueError("clips must satisfy 0 < min_still <= desired_still <= max_still")

    start_still = clips.get("start_still")
    start_duration = float(clips.get("start_duration", 0.0)) if start_still else 0.0
    end_still = clips.get("end_still")
    end_duration = float(clips.get("end_duration", 0.0)) if end_still else 0.0

    max_videos = int(clips.get("max_videos", 0))
    if max_videos < 0:
        raise ValueError("clips.max_videos must be >= 0")

    output = raw["output"]
    out_file = output.get("file")
    if not out_file:
        raise ValueError("output.file must be set")
    width = int(output.get("width", 1080))
    height = int(output.get("height", 1920))
    fps = int(output.get("fps", 30))
    codec = output.get("codec", "libx264")
    threads = output.get("threads", 4)
    
    # Motion settings (optional)
    motion_raw = raw.get("motion", {})
    motion = {
        "enabled": bool(motion_raw.get("enabled", False)),
        "probability": float(motion_raw.get("probability", 0.7)),
        "zoom_range": motion_raw.get("zoom_range", [1.0, 1.1]),
        "zoom_direction_mode": motion_raw.get("zoom_direction_mode", "random"),
        "pan_min": float(motion_raw.get("pan_min", 0.03)),
        "pan_max": float(motion_raw.get("pan_max", 0.06)),
        "pan_direction_mode": motion_raw.get("pan_direction_mode", "random"),
        "rotation_range": float(motion_raw.get("rotation_range", 0.2)),
        "max_factor": float(motion_raw.get("max_factor", 1.1)),
    }

    normalized = {
        "audio_path": audio_path,
        "images_dir": img_dir,
        "videos_dir": vid_dir,
        "clips": {
            "min_still": min_still,
            "desired_still": desired_still,
            "max_still": max_still,
            "start_still": start_still,
            "start_duration": start_duration,
            "end_still": end_still,
            "end_duration": end_duration,
            "max_videos": max_videos,
        },
        "output": {"file": out_file, "width": width, "height": height, "fps": fps, "codec": codec, "threads": threads},
        "motion": motion,
    }
    return normalized
