# random_video_generator/video_composer.py
from __future__ import annotations
import logging
import math
import os
import tempfile
import shutil
from typing import List, Tuple

# MoviePy top-level imports for 2.2.1+
try:
    from moviepy import VideoFileClip, ImageClip, AudioFileClip, concatenate_videoclips
except Exception as e:
    logging.error("moviepy >= 2.2.1 is required. Install with: pip install moviepy")
    raise

from .media_loader import list_files, IMAGE_EXTS, VIDEO_EXTS
from .image_processing import preprocess_image, load_image
from .motion_pillow import make_ken_burns_clip

def compute_still_durations(remaining_time: float, min_still: float, desired_still: float, max_still: float) -> List[float]:
    if remaining_time <= 0:
        return []

    count = max(1, math.floor(remaining_time / desired_still))
    durations = [desired_still] * count
    total = sum(durations)

    if total < remaining_time:
        scale = remaining_time / total
        durations = [min(max_still, d * scale) for d in durations]
    elif total > remaining_time:
        scale = remaining_time / total
        durations = [max(min_still, d * scale) for d in durations]

    diff = remaining_time - sum(durations)
    if abs(diff) > 1e-6:
        durations[-1] += diff

    if len(durations) > 1 and durations[-1] < min_still:
        removed = durations.pop()
        durations[-1] += removed
        logging.info("Last still was shorter than min_still; merged into previous still.")

    if len(durations) == 1 and durations[0] < min_still - 1e-6:
        raise ValueError("Remaining time insufficient to satisfy minimum still duration.")

    logging.info("Computed %d still(s), total %.2f s", len(durations), sum(durations))
    return durations

def build_sequence(cfg: dict) -> Tuple[List, float, tempfile.TemporaryDirectory, AudioFileClip]:
    """
    Build the visual clip sequence (images + videos + start/end stills)
    aligned to the given audio track.

    Returns:
        (clips, audio_clip, tmpdir)
    """

    # === Load configuration ===
    audio_path, images_dir, videos_dir, clips_cfg, output_cfg = extract_config(cfg)

    # === Load and validate audio ===
    audio, audio_len = load_audio(audio_path)
    validate_audio_length(audio_len, clips_cfg)

    # === Collect images and videos ===
    image_files, video_files = gather_assets(images_dir, videos_dir, clips_cfg)

    # === Calculate timing budgets ===
    included_videos, still_durations, remaining_budget = plan_timeline(cfg, video_files, audio_len)

    # === Prepare temporary directory and clips container ===
    tmpdir = tempfile.TemporaryDirectory(prefix="rvgen_")
    tmp_path = tmpdir.name
    clips = []

    # === Add stills and videos ===
    add_start_still(clips, clips_cfg, tmp_path, output_cfg)
    add_stills(clips, cfg, still_durations, image_files, tmp_path)
    add_videos(clips, included_videos, output_cfg)
    add_end_still(clips, clips_cfg, tmp_path, output_cfg)

    # === Validate total timeline ===
    total_visual = sum(getattr(c, "duration", 0) for c in clips)
    if abs(total_visual - audio_len) > 0.5:
        logging.warning("Total visual time %.2f s differs from audio %.2f s by >0.5s", total_visual, audio_len)

    return clips, audio, tmpdir

def extract_config(cfg):
    """Unpack key configuration values."""
    return (
        cfg["audio_path"],
        cfg["images_dir"],
        cfg["videos_dir"],
        cfg["clips"],
        cfg["output"],
    )

def load_audio(audio_path):
    """Load main audio clip and return (clip, duration)."""
    audio = AudioFileClip(audio_path)
    audio_len = audio.duration
    logging.info("Audio duration: %.2f s", audio_len)
    return audio, audio_len

def validate_audio_length(audio_len, clips_cfg):
    """Ensure start + end durations fit within total audio length."""
    total_reserved = clips_cfg["start_duration"] + clips_cfg["end_duration"]
    if total_reserved > audio_len + 1e-6:
        raise ValueError("start_duration + end_duration exceed audio length.")

def gather_assets(images_dir, videos_dir, clips_cfg):
    """Collect image and video files, excluding start/end stills."""
    image_files = list_files(images_dir, IMAGE_EXTS)
    if clips_cfg.get("start_still"):
        image_files = [p for p in image_files if os.path.normcase(p) != os.path.normcase(clips_cfg["start_still"])]
    if clips_cfg.get("end_still"):
        image_files = [p for p in image_files if os.path.normcase(p) != os.path.normcase(clips_cfg["end_still"])]
    video_files = list_files(videos_dir, VIDEO_EXTS) if videos_dir and os.path.isdir(videos_dir) else []
    return image_files, video_files

def plan_timeline(cfg, video_files, audio_len):
    """Decide how much time to allocate to videos vs. stills."""
    clips_cfg = cfg["clips"]
    reserved = clips_cfg["start_duration"] + clips_cfg["end_duration"]
    budget_for_all_visuals = audio_len - reserved
    if budget_for_all_visuals <= 0:
        raise ValueError("No budget for visuals after reserving start/end durations.")

    # Choose videos that fit
    included_videos = []
    remaining_budget = budget_for_all_visuals
    max_videos = clips_cfg.get("max_videos", 0)

    if max_videos > 0 and video_files:
        for vp in video_files:
            if len(included_videos) >= max_videos:
                break
            try:
                vclip = VideoFileClip(vp)
                vdur = vclip.duration
                vclip.close()
            except Exception as e:
                logging.warning("Cannot read video '%s' — skipping (%s)", vp, e)
                continue

            if vdur <= remaining_budget - 1e-6:
                included_videos.append(vp)
                remaining_budget -= vdur
                logging.info("Include video %s (%.2f s). Remaining: %.2f s", vp, vdur, remaining_budget)
            else:
                logging.info("Skip video %s: too long (%.2f s remaining)", vp, remaining_budget)

    # Compute still durations
    total_for_stills = remaining_budget
    if total_for_stills <= 0 and not included_videos:
        raise ValueError("No time for visuals after considering videos.")

    still_durations = compute_still_durations(
        total_for_stills,
        clips_cfg["min_still"],
        clips_cfg["desired_still"],
        clips_cfg["max_still"],
    )

    if not still_durations and not included_videos:
        raise ValueError("No stills could be allocated and no videos included.")

    return included_videos, still_durations, remaining_budget

def add_start_still(clips, clips_cfg, tmp_path, output_cfg):
    """Optionally add start still."""
    start_still = clips_cfg.get("start_still")
    if not start_still:
        return
    if not os.path.exists(start_still):
        raise FileNotFoundError(f"Start still not found: {start_still}")

    target = os.path.join(tmp_path, "start.png")
    preprocess_image(start_still, target, (output_cfg["width"], output_cfg["height"]))
    ic = ImageClip(target)
    if hasattr(ic, "with_duration"):
        ic = ic.with_duration(clips_cfg["start_duration"])
    else:
        ic = ic.set_duration(clips_cfg["start_duration"])
    clips.append(ic)
    logging.info("Added start still: %s (%.2f s)", start_still, clips_cfg["start_duration"])

def add_stills(clips, cfg, still_durations, image_files, tmp_path):
    """Add all stills, optionally applying Ken Burns motion."""
    output_cfg = cfg["output"]
    motion_cfg = cfg.get("motion", {})
    num_stills_needed = len(still_durations)
    num_to_use = min(len(image_files), num_stills_needed)
    if num_stills_needed > len(image_files):
        logging.warning("Needed %d stills but only %d available.", num_stills_needed, len(image_files))

    selected_stills = image_files[:num_to_use]
    for i, (img_path, dur) in enumerate(zip(selected_stills, still_durations[:num_to_use])):
        try:
            target = os.path.join(tmp_path, f"still_{i}.png")
            if motion_cfg.get("enabled", False):
                load_image(img_path, target)
                ic = make_ken_burns_clip(
                    target, dur, motion_cfg,
                    (output_cfg["width"], output_cfg["height"]),
                    cfg["output"]["fps"]
                )
            else:
                preprocess_image(img_path, target, (output_cfg["width"], output_cfg["height"]))
                ic = ImageClip(target).with_duration(dur)
            clips.append(ic)
            logging.info("Added still: %s (%.2f s)", img_path, dur)
        except Exception as e:
            logging.warning("Skipping still %s (%s)", img_path, e)

def add_videos(clips, included_videos, output_cfg):
    """Add included video clips, resizing to fit output dimensions."""
    for vpath in included_videos:
        try:
            vclip = VideoFileClip(vpath)
        except Exception as e:
            logging.warning("Failed to open video %s — skipping (%s)", vpath, e)
            continue

        try:
            if hasattr(vclip, "without_audio"):
                vclip = vclip.without_audio()
            elif hasattr(vclip, "with_audio"):
                vclip = vclip.with_audio(None)
            elif hasattr(vclip, "set_audio"):
                vclip = vclip.set_audio(None)
        except Exception:
            logging.warning("Could not remove audio from %s; continuing.", vpath)

        # Keep existing resize code (works with 2.2.1)
        try:
            target_w, target_h = output_cfg["width"], output_cfg["height"]
            from moviepy.video.fx.Resize import Resize
            from moviepy.video.fx.Crop import Crop
            from moviepy.video.fx.Margin import Margin

            vclip = vclip.with_effects([Resize(height=target_h)])
            if vclip.w > target_w:
                x_center = vclip.w / 2
                crop_x1 = int(x_center - target_w / 2)
                crop_x2 = crop_x1 + target_w
                vclip = vclip.with_effects([Crop(x1=crop_x1, x2=crop_x2)])
            elif vclip.w < target_w:
                pad = (target_w - vclip.w) // 2
                vclip = vclip.with_effects([Margin(left=pad, right=pad, color=(0, 0, 0))])
        except Exception as e:
            logging.warning("Video resize/crop failed for %s (%s)", vpath, e)

        clips.append(vclip)
        logging.info("Included video: %s (%.2f s)", vpath, vclip.duration)

def add_end_still(clips, clips_cfg, tmp_path, output_cfg):
    """Optionally add end still."""
    end_still = clips_cfg.get("end_still")
    if not end_still:
        return
    if not os.path.exists(end_still):
        raise FileNotFoundError(f"End still not found: {end_still}")

    target = os.path.join(tmp_path, "end.png")
    preprocess_image(end_still, target, (output_cfg["width"], output_cfg["height"]))
    ic = ImageClip(target)
    if hasattr(ic, "with_duration"):
        ic = ic.with_duration(clips_cfg["end_duration"])
    else:
        ic = ic.set_duration(clips_cfg["end_duration"])
    clips.append(ic)
    logging.info("Added end still: %s (%.2f s)", end_still, clips_cfg["end_duration"])

def assemble_and_write(cfg: dict):
    clips, audio_clip, tmpdir = None, None, None
    try:
        clips, audio_clip_obj, tmp = build_sequence(cfg)
        clips, audio_clip, tmpdir = clips, audio_clip_obj, tmp

        if not clips:
            raise RuntimeError("No visual clips produced (nothing to render).")

        final = concatenate_videoclips(clips, method="compose")

        try:
            if hasattr(final, "with_duration"):
                final = final.with_duration(audio_clip.duration)
            else:
                final = final.set_duration(audio_clip.duration)
        except Exception:
            logging.debug("Could not set duration with with_duration/set_duration; proceeding.")

        try:
            if hasattr(final, "with_audio"):
                final = final.with_audio(audio_clip)
            else:
                final = final.set_audio(audio_clip)
        except Exception:
            logging.warning("Could not attach audio via with_audio/set_audio. The output may be silent.")

        out_file = cfg["output"]["file"]
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        write_kwargs = {"fps": cfg["output"]["fps"], "codec": cfg["output"]["codec"], "audio_codec": "aac", "threads": cfg["output"]["threads"]}
        logging.info("Rendering to %s ...", out_file)
        final.write_videofile(out_file, **write_kwargs)
        logging.info("Render finished: %s", out_file)
    finally:
        try:
            if tmpdir:
                tmpdir.cleanup()
        except Exception:
            logging.debug("Tempdir cleanup failed; continuing.")
