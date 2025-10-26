# random_video_generator/image_processing.py
from __future__ import annotations
from PIL import Image
from typing import Tuple

def preprocess_image(src: str, dest: str, size: Tuple[int, int]) -> None:
    """
    Aspect-fill and center-crop src image to exact size, saving PNG to dest.
    Uses Pillow with Image.Resampling.LANCZOS for good quality.
    """
    target_w, target_h = size
    with Image.open(src) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        sw, sh = im.size
        scale = max(target_w / sw, target_h / sh)
        new_w = int(round(sw * scale))
        new_h = int(round(sh * scale))
        im_resized = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        im_cropped = im_resized.crop((left, top, right, bottom))
        im_cropped.save(dest, format="PNG")

def load_image(src: str, dest: str) -> None:
    """
    Aspect-fill and center-crop src image to exact size, saving PNG to dest.
    Uses Pillow with Image.Resampling.LANCZOS for good quality.
    """
    with Image.open(src) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        im.save(dest, format="PNG")