import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import numpy as np

# Support for multiple ROIs
rois = []
current_roi = None

# Function to check if a point is inside a rectangle
def is_point_in_rect(x, y, rect):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

# Variable to store the index of the marked ROI
marked_roi_index = None

# Function to select multiple ROIs
def select_multiple_rois(event, x, y, flags, param):
    global current_roi, rois, marked_roi_index
    if event == cv2.EVENT_LBUTTONDOWN:
        current_roi = [x, y, x, y]
        # Check if clicking inside an existing ROI to mark it
        for i, roi in enumerate(rois):
            if is_point_in_rect(x, y, roi):
                marked_roi_index = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if current_roi is not None:
            current_roi[2] = x
            current_roi[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        if current_roi is not None:
            current_roi[2] = x
            current_roi[3] = y
            rois.append(tuple(current_roi))
            current_roi = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Remove the last ROI if right-clicked
        if rois:
            rois.pop()

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', select_multiple_rois)
# Load the YOLOv8 model
model = YOLO("yolo11s.pt")
names=model.model.names
# Open the video file (use video file or webcam, here using webcam)
cap = cv2.VideoCapture("D:\\2)WORK AND FORUM\\TRS\\FALL DETECTION\\fall detection 1(w) best\\fall.mp4")
count=0

# Define thresholds for fall detection
fall_threshold = 0.5  # Adjust this value as needed
aspect_ratio_threshold = 0.5  # Typical aspect ratio for a fall

while True:
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))
    
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True,classes=0)

    # Draw all ROIs and highlight the marked one
    for i, roi in enumerate(rois):
        color = (0, 255, 0) if i != marked_roi_index else (0, 0, 255)
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), color, 2)

    # Filter detections based on ROIs if they are set, otherwise detect generally
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            x1, y1, x2, y2 = box
            # Check if the box is within any of the ROIs if ROIs are set
            if not rois or any(x1 >= roi[0] and y1 >= roi[1] and x2 <= roi[2] and y2 <= roi[3] for roi in rois):
                c = names[class_id]
                h = y2 - y1
                w = x2 - x1
                aspect_ratio = h / w
                thresh = h - w
                print(f"Threshold: {thresh}, Aspect Ratio: {aspect_ratio}")

                # Adjust fall detection based on aspect ratio and threshold
                if aspect_ratio < aspect_ratio_threshold and thresh <= fall_threshold * h:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                    cvzone.putTextRect(frame, f"{'Fall'}", (x1, y1), 1, 1)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                    cvzone.putTextRect(frame, f"{'Normal'}", (x1, y1), 1, 1)

    # Check for key press to remove the marked ROI
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d') and marked_roi_index is not None:
        rois.pop(marked_roi_index)
        marked_roi_index = None

    if key == ord('q'):
        break

    cv2.imshow("RGB", frame)

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

