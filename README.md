# Fall Detection System

A fall detection system with two approaches: **Simple (YOLO only)** and **Advanced (YOLO + MediaPipe Pose)**. Both methods use YOLO v8/v11 for person detection combined with different analysis techniques.

---

## 📋 Quick Overview

| Feature | Normal Flow | MediaPipe Flow |
|---------|-----------|-----------------|
| **Models Used** | YOLO v11s only | YOLO v8n + MediaPipe Pose |
| **Detection Method** | Aspect ratio (height/width) | Body angles + aspect ratio |
| **Speed** | ⚡ Very Fast | 🚀 Fast (with frame skipping) |
| **Accuracy** | Good | Better (uses body landmarks) |
| **Complexity** | Simple | Advanced pose analysis |
| **Best For** | Quick deployment | High accuracy needed |

---

## 🎯 Two Implementation Methods

### **1. Normal Flow** - `fall_detection_model.py`
Uses **YOLO v11s** with ROI (Region of Interest) support for basic fall detection.

**How it works:**
- YOLO detects people in video
- Calculates person's aspect ratio (width/height)
- If ratio > 1.2 → **FALL DETECTED** (person wider than tall)
- Supports multiple monitoring zones (ROIs)

**Best for:** Quick setup, live camera monitoring, multi-zone surveillance

**Controls:**
- **Left Click**: Draw ROI boxes
- **Right Click**: Remove last ROI
- **Mouse Move**: See coordinates

---

### **2. MediaPipe Flow** - `fall_detection_with_mediapipe.py`
Uses **YOLO v8n + MediaPipe Pose** for advanced pose-based fall detection.

**How it works:**
1. YOLO detects person in frame
2. MediaPipe extracts 33 body landmarks (joints/keypoints)
3. Analyzes **torso angle** and **aspect ratio** together:
   - Torso nearly horizontal (angle < 45° or > 135°) = Falling
   - Wider than tall (aspect ratio > 0.8) + horizontal torso = Falling
4. Frame skipping (every 2nd frame) for speed optimization
5. Draws skeleton overlay for visualization

**Key Optimizations:**
- `static_image_mode=True` - Prevents tracking errors on cropped frames
- `model_complexity=0` - Uses lightweight pose model
- Frame caching - Reuses detections during skipped frames

**Best for:** High accuracy needed, detailed pose analysis, video recording

---

## 📁 Files in This Project

| File | Purpose |
|------|---------|
| `fall_detection_model.py` | **Normal Flow** - YOLO only with ROI support |
| `fall_detection_with_mediapipe.py` | **MediaPipe Flow** - Advanced pose detection |
| `yolo11s.pt` | YOLO v11 small weights (for normal flow) |
| `yolov8n.pt` | YOLO v8 nano weights (for MediaPipe flow) |

---

## ⚙️ Requirements

```
Python 3.8+
opencv-python (cv2)
numpy
ultralytics (YOLO)
mediapipe (for MediaPipe flow only)
cvzone (for normal flow only)
```

## 📦 Installation

```bash
# Basic installation
pip install opencv-python numpy ultralytics

# For MediaPipe flow
pip install mediapipe

# For Normal flow  
pip install cvzone
```

---

## 🚀 Usage

### **Normal Flow**
```bash
python fall_detection_model.py
```
- Draw ROI boxes with left-click before monitoring
- Right-click to undo

### **MediaPipe Flow**
```bash
python fall_detection_with_mediapipe.py
```
- Automatically processes video
- Shows body skeleton overlay
- Saves output video with detections

---

## ⭐ Important Implementation Details

### Normal Flow - Key Logic:
```python
aspect_ratio = width / height
if aspect_ratio > 1.2:  # Person is wider than tall
    FALL_DETECTED = True
```

### MediaPipe Flow - Key Logic:
```python
# Extract shoulder and hip positions
torso_angle = angle between shoulders and hips

# Fall detection combines two conditions:
if aspect_ratio > 1.2 OR (aspect_ratio > 0.8 AND horizontal_torso):
    FALL_DETECTED = True
```

### Performance Tips:
- MediaPipe uses `skip_frames=2` to process every 2nd frame (50% speed boost)
- Normal flow is real-time suitable for live monitoring
- Both models run on CPU; GPU support available with proper setup

---

## 🎨 Output Visualization

**Both flows show:**
- ✅ Green box + "Normal" → Person standing normally
- ❌ Red box + "FALL DETECTED!" → Person has fallen
- 🦴 Body skeleton (MediaPipe flow only) → Pose landmarks

---

## 📝 Notes

- Adjust `skip_frames` in MediaPipe flow to balance speed vs accuracy
- Modify detection thresholds (aspect ratio, torso angle) for your specific use case
- For best accuracy, train on domain-specific fall data
- Both models work with video files or live camera input

1. Loads YOLO v11s model for person detection
2. Processes video frames at regular intervals
3. Tracks detected persons across frames
4. Analyzes person dimensions to detect falls
5. Displays results on video frames

## Notes

- Update the video file path in the script to use your own video
- Adjust the class filter (currently set to class 0 for persons only)
- Frame processing uses skip rate for optimized performance

---

Simple fall detection system using YOLO v11 for surveillance applications.
