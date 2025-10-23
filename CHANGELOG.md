# Changelog

## [1.1.0] - 2025-10-23
### 🎉 Major Milestone
This release represents the first **stable, modular build** of the Random Video Generator project.  
It produces automated short-form vertical videos using image, video, and audio sources — ideal for TikTok, YouTube Shorts, and Instagram Reels.

### 🚀 Features
- **Audio-driven duration**  
  - The total video length automatically matches the selected audio clip.
- **Dynamic still durations**  
  - Stills are calculated to fill available time using min/desired/max durations.
- **Start and end stills**  
  - Configurable fixed-duration stills at beginning and end of video.
- **Video inclusion logic**  
  - Automatically includes videos that fit within time constraints; full clip only.
  - Skips videos that would exceed available time.
- **Ken Burns effect (Pillow-based)**  
  - Subtle pan/zoom/rotation animation for stills.
  - Fully configurable (zoom range, pan range, direction, easing, rotation).
  - Randomization ensures variation while maintaining smooth motion.
- **Blurred background fill for zoom-out effects.**
- **Automatic aspect correction for video clips**  
  - Uses new MoviePy 2.2.1 transform API (`with_effects` + `Resize`, `Crop`, `Margin`).
  - Prevents aspect-induced resizing of final render.
- **Configuration loader with validation and detailed JSON error hints.**
- **Structured logging system** with timestamps and severity.
- **Command-line interface** with parameter checking.
- **Fully modular codebase**  
  - Organized into `config.py`, `video_composer.py`, `image_processing.py`, `motion_pillow.py`, etc.

### 🧠 Technical Highlights
- Compatible with **Python 3.14+**
- Compatible with **MoviePy 2.2.1+**
- Uses **Pillow** for all image operations and motion frame generation
- Fallback-safe for legacy MoviePy 1.x methods (resize, crop, margin)
- Robust temporary file management and cleanup
- Easing implemented via `ease_in_out_cosine` for natural motion timing

### 🧩 Known Limitations
- Some black corners may appear during rotation-heavy Ken Burns effects.
- No transitions yet between clips (planned for v1.2.0).
- Audio ducking and text overlays not yet implemented.
- No blurred sidebar fill for padded videos (planned for v1.2.0).

### 📁 Structure
```
random_video_generator/
│
├── main.py
├── config.py
├── video_composer.py
├── image_processing.py
├── motion_pillow.py
├── media_loader.py
├── utils.py
│
└── config.json   # User configuration file
```

### 🏷 Version
`v1.1.0` — "Vertical Foundations"
