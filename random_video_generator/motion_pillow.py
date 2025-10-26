# random_video_generator/motion_pillow.py
"""
Pillow-based Ken Burns (zoom + pan + tiny rotation) renderer.

Exports:
    make_ken_burns_clip(img_path, duration, motion_cfg, size, fps) -> ImageSequenceClip

Notes:
- Works with MoviePy >= 2.2.1 by pre-rendering frames using Pillow and returning an ImageSequenceClip.
- Designed for subtle, cinematic motion. Defaults are conservative.
- `img_path` is expected to be the preprocessed image (already aspect-filled to `size`) but any image will work.
- The function always returns a clip whose duration equals `duration`.

Config (motion_cfg) expected keys (all optional — sensible defaults used):
{
  "enabled": True,
  "probability": 0.8,
  "zoom_range": [1.0, 1.12],      # start/end zoom factors (>= 1.0)
  "pan_range": 0.06,              # max pan as fraction of width/height (±)
  "direction_mode": "random",     # "in", "out", or "random"
  "rotation_range": 1.5,          # degrees, max absolute rotation
  "blur_edges": True,             # produce blurred background when zooming out (not needed for zoom>=1)
  "fps": 30,
  "blur_radius": 20               # for blurred background
}
"""

from __future__ import annotations
import logging
import random
import math
from typing import Tuple, List

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

# ImageSequenceClip import compatible with MoviePy 2.x
try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except Exception:
    # Fallback to top-level if available
    try:
        from moviepy import ImageSequenceClip  # type: ignore
    except Exception:
        raise RuntimeError("moviepy ImageSequenceClip not available; ensure moviepy is installed.")


def smoothstep(x: float) -> float:
    """Compute a smooth cubic interpolation between 0 and 1.

    Used for soft transitions, easing both in and out smoothly.
    Commonly applied to zoom or rotation progress curves.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return x * x * (3 - 2 * x)
    
def ease_in_out_cosine(t: float, edge: float = 0.1) -> float:
    """Apply cosine-based easing for natural acceleration and deceleration.

    Args:
        t (float): Normalized time value between 0 and 1.
        edge (float): Unused parameter placeholder for API consistency.
    Returns:
        float: Smoothed time value between 0 and 1.
    """
    return 0.5 * (1 - math.cos(math.pi * t))

def ease_in_out_cosine_2(t: float, edge: float = 0.1) -> float:
    """Alternative cosine-based easing function with adjustable edge softness.

    Args:
        t (float): Normalized time between 0 and 1.
        edge (float): Fraction of duration used for easing in/out.
    Returns:
        float: Adjusted time value with easing applied.
    """
    if t < edge:
        return 0.5 * (1 - math.cos(math.pi * (t / edge))) * (edge / (0.5 * math.pi))  # scale fix
    elif t > 1 - edge:
        return 1 - 0.5 * (1 - math.cos(math.pi * ((1 - t) / edge))) * (edge / (0.5 * math.pi))
    else:
        mid_t = (t - edge) / (1 - 2 * edge)
        return edge + mid_t * (1 - 2 * edge)


def make_blurred_background(pil_img: Image.Image, size: Tuple[int, int], blur_radius: int = 20) -> Image.Image:
    """
    Create a blurred, desaturated background from the image to fill behind the main
    subject when needed (e.g., if we do a zoom out or rotation).
    """
    bg = pil_img.resize(size, resample=Image.Resampling.LANCZOS)
    # Desaturate
    try:
        enhancer = ImageEnhance.Color(bg)
        bg = enhancer.enhance(0.25)
    except Exception:
        # If not available, continue
        pass
    # Slightly darken for contrast
    try:
        enhancer = ImageEnhance.Brightness(bg)
        bg = enhancer.enhance(0.9)
    except Exception:
        pass
    bg = bg.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return bg
    
def upscale(src: Image, src_w: int, src_h: int, target_w: int, target_h: int, max_factor: float):    
    """Upscale the source image if smaller than target dimensions.
    
    Ensures the image has enough resolution for cropping and zooming
    without visible pixelation.

    Returns:
        Tuple: (resized_image, new_src_w, new_src_h, target_w, target_h)
    """
    # If source is not at least as large as target, upscale to avoid extreme pixelation
    if src_w < target_w or src_h < target_h:
        # Compute scaling factor to ensure both dimensions meet or exceed target
        scale_factor = max(target_w / src_w, target_h / src_h)
        scale_factor = min(scale_factor, max_factor)
        new_w = int(round(src_w * scale_factor))
        new_h = int(round(src_h * scale_factor))
        src = src.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        src_w, src_h = src.size
    return src, src_w, src_h, target_w, target_h
        
def return_single_frame(img_path: str, size: Tuple[int, int], duration: float, fps: int, blur_radius: int, blur_edges: int, max_factor: float):
    """Create a static ImageSequenceClip with repeated identical frames.

    Used when motion is disabled or skipped probabilistically.

    Args:
        img_path (str): Path to source image.
        duration (float): Duration of the resulting clip in seconds.
        fps (int): Output frame rate.
    Returns:
        ImageSequenceClip: Static clip with repeated frames.
    """
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip as ISC
    # return a single-frame clip repeated for duration (ImageSequenceClip accepts list of arrays)
    with Image.open(img_path) as src:
        src_w, src_h = src.size
        target_w, target_h = size
        
        # Upscale images smaller than final video size 
        src, src_w, src_h, target_w, target_h = upscale(src, src_w, src_h, target_w, target_h, max_factor)

        # Background for padding/blur if needed
        bg = make_blurred_background(src, (target_w, target_h), blur_radius) if blur_edges else None

        # Center coordinates on source, then offset by pan (px,py)
        center_x = src_w // 2
        center_y = src_h // 2

        # Crop box
        x1 = center_x - target_w // 2
        y1 = center_y - target_h // 2
        x2 = x1 + target_w
        y2 = y1 + target_h
        
        # Arrange for images smaller than final video size to be centered
        paste_x = paste_y = 0
        if target_w > src_w:
            paste_x = (target_w - src_w) // 2
        if target_h > src_h:
            paste_y = (target_h - src_h) // 2

        # Clamp box to source bounds
        if x1 < 0:
            x1, x2 = 0, src_w
        if y1 < 0:
            y1, y2 = 0, src_h
        if x2 > src_w:
            x2, x1 = src_w, 0
        if y2 > src_h:
            y2, y1 = src_h, 0

        # Extract crop
        crop = src.crop((x1, y1, x2, y2))
        logging.info("paste, src_h = %d, target_h = %d, diff = %d, half_diff = %d", src_h, target_h, target_h - src_h, (target_h - src_h) // 2)
        
        # Place image on backdrop
        if bg is not None:
            composed = bg.copy()
            composed.paste(crop, (paste_x, paste_y), None)
            crop = composed
        
        frame = np.array(crop.convert("RGB"))
    frames = [frame] * max(1, int(round(duration * fps)))
    return ISC(frames, fps=fps)
    
def define_zoom(zoom_range: list, direction: str):
    """Determine zoom start and end values with consistent direction.

    Args:
        zoom_range (list): Min and max zoom factors (e.g., [1.0, 1.12]).
        direction (str): 'in', 'out', or 'random'.
    Returns:
        Tuple[str, float, float]: (direction_label, start_zoom, end_zoom)
    """
    # Determine zoom start/end (with consistent direction)
    zr0, zr1 = float(zoom_range[0]), float(zoom_range[1])
    if direction == "in":
        start_zoom, end_zoom = min(zr0, zr1), max(zr0, zr1)
        zoom_dir = "in"
    elif direction == "out":
        start_zoom, end_zoom = max(zr0, zr1), min(zr0, zr1)
        zoom_dir = "out"
    else:
        # random: choose one consistent zoom direction
        if random.random() < 0.5:
            start_zoom, end_zoom = zr0, zr1
            zoom_dir = "in"
        else:
            start_zoom, end_zoom = zr1, zr0
            zoom_dir = "out"

    logging.debug("Ken Burns zoom direction: %s (%.2f→%.2f)", zoom_dir, start_zoom, end_zoom)
    return zoom_dir, start_zoom, end_zoom
    
def define_pan(
    start_zoom: float,
    end_zoom: float,
    target_w: float,
    target_h: float,
    src_w: float,
    src_h: float,
    pan_min: float,
    pan_max: float,
    rot_range: float | None = None
) -> tuple[float, float, float, float]:
    """Compute safe pan start/end offsets to maintain visible content throughout motion.

    Ensures panning stays within valid bounds even when zooming or rotating.
    Prevents abrupt direction changes ("bounces") and edge clipping.

    Args:
        start_zoom (float): Initial zoom factor.
        end_zoom (float): Final zoom factor.
        target_w (float): Output frame width.
        target_h (float): Output frame height.
        src_w (float): Source image width.
        src_h (float): Source image height.
        rot_range (float | None): Rotation range for safety margin.

    Returns:
        Tuple[float, float, float, float]: (start_pan_x, end_pan_x, start_pan_y, end_pan_y)
    """
    # compute zoom extremes (we already have start_zoom/end_zoom)
    z_min = min(start_zoom, end_zoom)
    z_max = max(start_zoom, end_zoom)

    # Use the *largest* crop that will appear during the animation (most restrictive).
    # crop_width_max is target width divided by the smallest zoom (zoom out -> bigger crop)
    crop_w_max = target_w / z_min
    crop_h_max = target_h / z_min
    
    src_center_x = src_w / 2.0
    src_center_y = src_h / 2.0

    # Optional safety margin for rotation and rounding (reduce available pan range a bit)
    rotation_margin_px = 0.0
    if rot_range:
        # conservative padding: a small fraction of target dims
        rotation_margin_px = max(target_w, target_h) * 0.03

    # Allowed center positions (in pixels) for the *largest* crop
    # TODO: min is always bigger than max here, and it causes the pan to be 0. if we simply flip them, we get bounce. I need to rethink this.
    allowed_center_x_min = crop_w_max / 2.0 + rotation_margin_px
    allowed_center_x_max = src_w - crop_w_max / 2.0 - rotation_margin_px
    allowed_center_y_min = crop_h_max / 2.0 + rotation_margin_px
    allowed_center_y_max = src_h - crop_h_max / 2.0 - rotation_margin_px

    # Convert to allowed pan offsets from image center
    allowed_offset_x_min = allowed_center_x_min - src_center_x
    allowed_offset_x_max = allowed_center_x_max - src_center_x
    allowed_offset_y_min = allowed_center_y_min - src_center_y
    allowed_offset_y_max = allowed_center_y_max - src_center_y

    # If the allowed interval collapsed, clamp to zero (safe fallback)
    if allowed_offset_x_min > allowed_offset_x_max:
        allowed_offset_x_min = allowed_offset_x_max = 0.0
    if allowed_offset_y_min > allowed_offset_y_max:
        allowed_offset_y_min = allowed_offset_y_max = 0.0

    # Choose one coherent motion style (horizontal or vertical)
    if random.random() < 0.5:
        # horizontal travel
        pan_axis = "x"
        axis_allowed_min, axis_allowed_max = allowed_offset_x_min, allowed_offset_x_max
        other_allowed_min, other_allowed_max = allowed_offset_y_min, allowed_offset_y_max
    else:
        # vertical travel
        pan_axis = "y"
        axis_allowed_min, axis_allowed_max = allowed_offset_y_min, allowed_offset_y_max
        other_allowed_min, other_allowed_max = allowed_offset_x_min, allowed_offset_x_max

    # Determine travel distance that will fit inside allowed interval
    axis_span = axis_allowed_max - axis_allowed_min
    if axis_span <= 1.0:
        # No room to pan on the chosen axis, fallback to no pan
        travel = 0.0
    else:
        # pick travel as a fraction of available span to avoid edge hugging
        travel_frac = random.uniform(pan_min, pan_max)  # % of allowed span
        travel = axis_span * travel_frac

    # pick a center for the travel so both endpoints remain within allowed interval
    mid_min = axis_allowed_min + travel / 2.0
    mid_max = axis_allowed_max - travel / 2.0
    if mid_min > mid_max:
        # numeric edge case: shrink travel to fit exactly
        travel = axis_span
        mid_min = axis_allowed_min + travel / 2.0
        mid_max = axis_allowed_max - travel / 2.0

    mid_center = random.uniform(mid_min, mid_max)
    start_axis = mid_center - travel / 2.0
    end_axis = mid_center + travel / 2.0

    # For the orthogonal axis, pick a small constant "wobble" inside its allowed range
    if other_allowed_max - other_allowed_min <= 1.0:
        other_pos = 0.0
    else:
        other_pos = random.uniform(other_allowed_min * 0.2, other_allowed_max * 0.2)

    if pan_axis == "x":
        start_pan_x, end_pan_x = start_axis, end_axis
        start_pan_y = end_pan_y = other_pos
    else:
        start_pan_y, end_pan_y = start_axis, end_axis
        start_pan_x = end_pan_x = other_pos

    logging.debug("Ken Burns pan direction: %s (%.1f->%.1f, %.1f->%.1f)", start_pan_x, end_pan_x, start_pan_y, end_pan_y)
    
    return start_pan_x, end_pan_x, start_pan_y, end_pan_y
    
def define_rotation(rot_range: float):
    """Randomly choose a small rotational movement for subtle motion.

    Args:
        rot_range (float): Maximum rotation angle in degrees.
    Returns:
        Tuple[float, float]: (start_rotation, end_rotation)
    """
    # Rotation in degrees
    rot_dir = random.choice([-1, 1])
    rot_magnitude = random.uniform(0, rot_range)
    start_rot = rot_dir * rot_magnitude
    end_rot = rot_dir * random.uniform(rot_magnitude * 0.3, rot_magnitude)
    
    return start_rot, end_rot
    
def create_frame(
    i: int,
    n_frames: int,
    ease_fraction: float,
    start_zoom: float,
    end_zoom: float,
    start_pan_x: float,
    end_pan_x: float,
    start_pan_y: float,
    end_pan_y: float,
    start_rot: float,
    end_rot: float,
    target_w: int,
    target_h: int,
    src_w: int,
    src_h: int,
    src: Image,
    bg: Optional[Image],
    frames: List[np.ndarray]
    ) -> None:
    """Render a single frame of the Ken Burns animation.

    Computes the interpolated zoom, pan, and rotation for frame `i`,
    crops and resizes the image, and appends the resulting RGB array
    to the shared frame list.

    Args:
        i (int): Current frame index.
        n_frames (int): Total number of frames.
        ease_fraction (float): Fraction of easing used for motion curves.
        start_zoom, end_zoom (float): Zoom start/end values.
        start_pan_x, end_pan_x, start_pan_y, end_pan_y (float): Pan coordinates.
        start_rot, end_rot (float): Rotation start/end angles.
        target_w, target_h (int): Output dimensions.
        src_w, src_h (int): Source dimensions.
        src (Image): Source PIL image.
        bg (Optional[Image]): Background for compositing blurred edges.
        frames (List[np.ndarray]): List of frame arrays to append to.
    """
    t = i / max(1, n_frames - 1)  # normalized [0..1]

    # Separate easing profiles for different motions
    f_zoom = ease_in_out_cosine(t, edge=ease_fraction)   # smooth zoom
    f_rot  = ease_in_out_cosine(t, edge=ease_fraction)   # smooth rotation
    f_pan  = ease_in_out_cosine(t, edge=ease_fraction)   # linear pan across full duration

    # Interpolate params using independent progress
    z = start_zoom + (end_zoom - start_zoom) * f_zoom
    px = start_pan_x + (end_pan_x - start_pan_x) * f_pan
    py = start_pan_y + (end_pan_y - start_pan_y) * f_pan
    rot = start_rot + (end_rot - start_rot) * f_rot

    # Compute crop size (smaller when zoom>1)
    crop_w = int(round(target_w / z))
    crop_h = int(round(target_h / z))

    # Center coordinates on source, then offset by pan (px,py)
    center_x = src_w // 2 + int(round(px))
    center_y = src_h // 2 + int(round(py))

    # Crop box
    x1 = center_x - crop_w // 2
    y1 = center_y - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Clamp box to source bounds
    if x1 < 0:
        x1, x2 = 0, crop_w
    if y1 < 0:
        y1, y2 = 0, crop_h
    if x2 > src_w:
        x2, x1 = src_w, src_w - crop_w
    if y2 > src_h:
        y2, y1 = src_h, src_h - crop_h

    # Extract crop
    crop = src.crop((x1, y1, x2, y2))

    # If rotation is requested, rotate crop first to avoid clipping after resize
    if abs(rot) > 0.01:
        # Rotate with expand=False to keep same size; rotate around center
        try:
            crop = crop.rotate(rot, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=None)
        except TypeError:
            # Older Pillow versions may not support fillcolor param
            crop = crop.rotate(rot, resample=Image.Resampling.BICUBIC, expand=False)

    # Resize crop back to target size (simulate zoom)
    frame_img = crop.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

    # If rotation introduced empty corners or if we want a background (blurred), composite over bg
    if bg is not None:
        composed = bg.copy()
        composed.paste(frame_img, (0, 0), None)
        frame_img = composed

    # Convert to numpy array (RGB)
    arr = np.array(frame_img.convert("RGB"))
    frames.append(arr)
    
def export_clip(duration: float, fps: int, frames: List[np.ndarray]):
    # Build ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=fps)
    # Ensure duration exactly matches requested duration (frame rounding can cause off-by-one)
    try:
        if hasattr(clip, "with_duration"):
            clip = clip.with_duration(duration)
        else:
            clip = clip.set_duration(duration)
    except Exception:
        # ignore if neither exist
        pass

    return clip

def make_ken_burns_clip(
    img_path: str,
    duration: float,
    motion_cfg: dict,
    size: Tuple[int, int],
    fps: int,
    ):
    """
    Create an ImageSequenceClip that applies a Ken Burns effect to the image.

    Parameters
    ----------
    img_path : str
        Path to the preprocessed image (ideally already aspect-filled to `size`).
    duration : float
        Duration in seconds of the resulting clip.
    motion_cfg : dict
        Motion configuration (see module docstring for keys).
    size : (width, height)
        Output frame size in pixels.
    fps : int
        Output frames-per-second.

    Returns
    -------
    ImageSequenceClip
        A moviepy ImageSequenceClip with the animated frames.
    """
    # Defaults and extraction
    seed = motion_cfg.get("seed")
    if seed is not None:
        random.seed(seed)
        logging.info(f"Ken Burns random seed set to {seed}")
    enabled = bool(motion_cfg.get("enabled", True))
    prob = float(motion_cfg.get("probability", 0.8))
    zoom_range = motion_cfg.get("zoom_range", [1.0, 1.12])
    pan_min = float(motion_cfg.get("pan_min", 0.03))
    pan_max = float(motion_cfg.get("pan_max", 0.12))
    zoom_direction = motion_cfg.get("zoom_direction_mode", "random")
    pan_direction = motion_cfg.get("pan_direction_mode", "random")
    rot_range = float(motion_cfg.get("rotation_range", 0.0))
    blur_edges = bool(motion_cfg.get("blur_edges", True))
    blur_radius = int(motion_cfg.get("blur_radius", 20))
    ease_fraction = float(motion_cfg.get("ease_fraction", 0.12))
    max_factor = float(motion_cfg.get("max_factor", 2.2))

    # Early escape if motion disabled or randomized-out
    if not enabled:
        logging.debug("Ken Burns (Pillow): disabled in config.")
        return return_single_frame(img_path, size, duration, fps, blur_radius, blur_edges, max_factor)

    if random.random() > prob:
        logging.debug("Ken Burns (Pillow): skipped by probability (%.2f)", prob)
        return return_single_frame(img_path, size, duration, fps, blur_radius, blur_edges, max_factor)

    # Load source image with PIL
    with Image.open(img_path) as src_img:
        src = src_img.convert("RGB")
        src_w, src_h = src.size
        target_w, target_h = size

        src, src_w, src_h, target_w, target_h = upscale(src, src_w, src_h, target_w, target_h, max_factor)

        # Background for padding/blur if needed
        bg = make_blurred_background(src, (target_w, target_h), blur_radius) if blur_edges else None

        # Define Zoom
        zoom_dir, start_zoom, end_zoom = define_zoom(zoom_range, zoom_direction)

        # Define Pan (this is broken)
        start_pan_x, end_pan_x, start_pan_y, end_pan_y = define_pan(start_zoom, end_zoom, target_w, target_h, src_w, src_h, pan_min, pan_max, rot_range)

        # Define Rotation
        start_rot, end_rot = define_rotation(rot_range)

        logging.info(
            "Applying Ken Burns: zoom %.2f->%.2f, pan (%.1f, %.1f)->(%.1f, %.1f), rot %.2f->%.2f",
            start_zoom,
            end_zoom,
            start_pan_x,
            start_pan_y,
            end_pan_x,
            end_pan_y,
            start_rot,
            end_rot,
        )

        # Frame generation
        n_frames = max(1, int(round(duration * fps)))
        frames: List[np.ndarray] = []
        for i in range(n_frames):
            create_frame(
                i,
                n_frames,
                ease_fraction,
                start_zoom,
                end_zoom,
                start_pan_x,
                end_pan_x,
                start_pan_y,
                end_pan_y,
                start_rot,
                end_rot,
                target_w,
                target_h,
                src_w,
                src_h,
                src,
                bg,
                frames)

        return export_clip(duration, fps, frames)
        
def test_ken_burns_single(
    img_path: str,
    out_path: str = "kenburns_test.mp4",
    duration: float = 5.0,
    size: Tuple[int, int] = (1080, 1080),
    fps: int = 30,
    zoom_range: Tuple[float, float] = (1.0, 1.1),
    pan_range: float = 0.05,
    rotation_range: float = 1.0,
    ease_fraction: float = 0.15,
    direction_mode: str = "in",
):
    """
    Standalone test generator for a single Ken Burns-animated still.
    Outputs MP4 so you can inspect motion behavior directly.

    Example:
        test_ken_burns_single("example.jpg", "out.mp4",
                              zoom_range=(1.0,1.2), pan_range=0.1,
                              rotation_range=2.0, direction_mode="in")
    """
    motion_cfg = {
        "enabled": True,
        "probability": 1.0,
        "zoom_range": list(zoom_range),
        "pan_range": pan_range,
        "rotation_range": rotation_range,
        "direction_mode": direction_mode,
        "ease_fraction": ease_fraction,
        "blur_edges": True
    }

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info(f"Testing Ken Burns: zoom={zoom_range}, pan_range={pan_range}, rotation={rotation_range}, dir={direction_mode}")

    clip = make_ken_burns_clip(
        img_path=img_path,
        duration=duration,
        motion_cfg=motion_cfg,
        size=size,
        fps=fps,
    )

    # Save to MP4 for review
    try:
        clip.write_videofile(out_path, fps=fps, codec="libx264", audio=False, threads=2)
    except Exception as e:
        logging.error(f"Failed to write video: {e}")
    finally:
        clip.close()


if __name__ == "__main__":
    # Example quick test
    test_ken_burns_single(
        img_path="E:/Bruach/Marketing Potos/IMG-20240726-WA0003.jpg",
        out_path="C:/Users/thisg/Desktop/pytest/test_kenburns.mp4",
        zoom_range=(1.0, 1.5),
        pan_range=0.5,
        rotation_range=0,
        ease_fraction=0,
        direction_mode="random"
    )