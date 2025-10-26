# Random Video Generator 🎬
**Automatic video composition tool** for short-form content platforms like TikTok, YouTube Shorts, and Instagram Reels.  
Combines stills, video clips, and audio into polished, vertical-format videos.

---

## 🚀 Features
- Generates random short-form videos using your media folders.
- Automatically syncs total length to the chosen audio track.
- Smooth Ken Burns motion for stills (Pillow-based pan, zoom, rotation).
- Start and end stills for branding or intro/outro.
- Respects min/max/desired durations for stills.
- Silences included video clips automatically.
- Maintains consistent vertical aspect (1080x1920 default).
- Configurable through a single JSON file.

---

## 🧠 Requirements
- **Python** 3.14 or newer  
- **MoviePy** 2.2.1 or newer  
- **Pillow** (installed automatically with MoviePy)

Install dependencies:
```bash
pip install moviepy pillow
```

---

## ⚙️ Usage

### 1️⃣ Prepare your folders
Organize your content like this:
```
/media/
 ├── audio/
 │    └── soundtrack.mp3
 ├── images/
 │    ├── img1.jpg
 │    ├── img2.jpg
 │    └── ...
 └── videos/
      ├── clip1.mp4
      ├── clip2.mp4
      └── ...
```

---

### 2️⃣ Create a `config.json`

## ✅ Example

```json
{
  "audio": {
    "path": "E:/Bruach/audio/music_track.mp3"
  },

  "images": {
    "path": "E:/Bruach/Marketing Photos"
  },

  "videos": {
    "path": "E:/Bruach/Clips"
  },

  "clips": {
    "min_still": 2.5,
    "desired_still": 3.0,
    "max_still": 5.0,
    "start_still": "E:/Bruach/Brand/intro.png",
    "start_duration": 2.0,
    "end_still": "E:/Bruach/Brand/call_to_action.png",
    "end_duration": 2.5,
    "max_videos": 1
  },

  "motion": {
    "enabled": true,
    "probability": 0.8,
    "zoom_range": [1.0, 1.12],
    "zoom_direction_mode": "random",
    "pan_min": 0.03,
    "pan_max": 0.06,
    "pan_direction_mode": "random",
    "rotation_range": 1.5,
    "max_factor": 1.1
  },

  "output": {
    "file": "E:/Bruach/Output/final_video.mp4",
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "codec": "libx264",
    "threads": 4
  }
}
```

---

## 🧠 Section Details

### 🎵 `audio`
| Key | Type | Description |
|-----|------|--------------|
| `path` | string | Path to the background audio track (MP3/WAV). Must exist. |

---

### 🖼️ `images`
| Key | Type | Description |
|-----|------|--------------|
| `path` | string | Directory containing still images to use in the video. Must exist. |

---

### 🎞️ `videos`
| Key | Type | Description |
|-----|------|--------------|
| `path` | string | Directory containing short video clips to include between stills. Optional — if missing, videos are skipped. |

---

### 🧩 `clips`
Controls how stills and short clips are arranged on the timeline.

| Key | Type | Description |
|-----|------|--------------|
| `min_still` | float | Minimum still duration (seconds). Must be > 0. |
| `desired_still` | float | Target still duration (seconds). |
| `max_still` | float | Maximum still duration (seconds). Must satisfy `min_still <= desired_still <= max_still`. |
| `start_still` | string | Optional image shown before the main sequence. |
| `start_duration` | float | Duration of start still (seconds). |
| `end_still` | string | Optional image shown after the main sequence. |
| `end_duration` | float | Duration of end still (seconds). |
| `max_videos` | int | Maximum number of video clips to include between stills. |

---

### 🎥 `motion` (Ken Burns effect)
Controls panning, zooming, and rotation applied to still images.

If `enabled` is `false`, stills remain static.

| Key | Type | Description | Default |
|-----|------|--------------|----------|
| `enabled` | bool | Enable/disable Ken Burns motion. | `false` |
| `probability` | float | Probability that a still will receive motion (0–1). | `0.7` |
| `zoom_range` | [float, float] | Start/end zoom factors. | `[1.0, 1.1]` |
| `zoom_direction_mode` | string | `"in"`, `"out"`, or `"random"`. | `"random"` |
| `pan_min` | float | Minimum pan offset as fraction of image width/height. | `0.03` |
| `pan_max` | float | Maximum pan offset as fraction of image width/height. | `0.06` |
| `pan_direction_mode` | string | `"horizontal"`, `"vertical"`, or `"random"`. | `"random"` |
| `rotation_range` | float | Max rotation angle (degrees). | `0.2` |
| `max_factor` | float | Optional safety upscaling factor for still images. | `1.1` |

---

### 📤 `output`
Controls final video rendering.

| Key | Type | Description | Default |
|-----|------|--------------|----------|
| `file` | string | Output video file path (MP4 recommended). | — |
| `width` | int | Output frame width (pixels). | `1080` |
| `height` | int | Output frame height (pixels). | `1920` |
| `fps` | int | Frames per second. | `30` |
| `codec` | string | Video codec for MoviePy (usually `"libx264"`). | `"libx264"` |
| `threads` | int | Number of encoding threads to use. | `4` |

---

### ⚠️ Windows JSON tip
If you’re on Windows, **escape backslashes** in paths, or use forward slashes:

✅ Correct:
```json
"C:/Brand/intro.png"
```
or
```json
"C:\\Users\\User\\Videos\\file.mp4"
```

❌ Incorrect:
```json
"C:\Users\User\Videos\file.mp4"
```


---

### 3️⃣ Run the generator
From the project root:
```bash
python -m random_video_generator.main config.json
```

If you call it with missing or invalid arguments, it will print detailed usage instructions.

---

## 🪄 Example Output
- Dynamic stills with smooth pan/zoom
- Interleaved video clips (muted)
- Perfectly timed to the selected audio

---

## 🧩 Configuration Highlights
| Key | Description |
|-----|--------------|
| `clips.min_still / desired_still / max_still` | Controls pacing of image display |
| `clips.start_still` / `end_still` | Optional fixed intro/outro images |
| `motion.zoom_range` | Controls how much zoom is applied |
| `motion.pan_range` | Maximum panning distance (as fraction of frame size) |
| `motion.ease_fraction` | Amount of smooth acceleration/deceleration |
| `output.width/height` | Defines final aspect ratio (default 1080×1920) |

---

## 🔧 Future Roadmap
| Version | Feature |
|----------|----------|
| **v1.2.0** | Better modularisation of code |
| **v1.3.0** | Transitions and basic text overlays |
| **v1.4.0** | Batching |


## 🔧 Future Wishlist
| Audio ducking and music-level automation |
| Configurable multi-template presets and branding layers |

---

## 🧑‍💻 Author Notes
Developed as a flexible automation toolkit for short-form video production, focusing on **Bruach’s content automation pipeline**.

Versioning follows `semantic versioning` (MAJOR.MINOR.PATCH).

---

## 📜 License
MIT License © 2025 Bruach / Contributors
