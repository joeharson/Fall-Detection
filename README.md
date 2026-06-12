# Fall Detection Project

A fall detection system using YOLO v11 for real-time person detection and fall identification in video streams.

## Overview

This project uses the YOLO v11 small (yolo11s) model to detect people in video frames and analyze their posture to identify falls. It supports multiple regions of interest (ROIs) for flexible monitoring.

## Features

- **Real-time Detection**: Uses YOLO v11s for fast and accurate person detection
- **Multi-ROI Support**: Define multiple regions of interest for monitoring
- **Video Processing**: Works with video files or webcam input
- **Fall Detection**: Analyzes person shape (height-to-width ratio) to detect falls

## Files

- `model.py` - Main detection script with ROI support
- `model original.py` - Original baseline model script
- `yolo11s.pt` - YOLO v11 small model weights

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Ultralytics YOLO
- cvzone

## Installation

```bash
pip install opencv-python numpy ultralytics cvzone
```

## Usage

Run the fall detection model:

```bash
python model.py
```

**Controls:**
- **Left Click**: Draw ROI boxes for monitoring areas
- **Right Click**: Remove the last ROI
- **Mouse Move**: Track coordinates in the image

## How It Works

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
