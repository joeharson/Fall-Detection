import cv2
import math
import mediapipe as mp
from ultralytics import YOLO

class FastFallDetector:
    def __init__(self, yolo_model_path='yolov8n.pt'): # Using Nano model for max speed
        print("Loading YOLO model...")
        self.yolo_model = YOLO(yolo_model_path)
        
        print("Loading MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        
        # CRITICAL FIX FOR SCATTERING: static_image_mode=True prevents tracking errors on cropped frames
        # model_complexity=0 makes MediaPipe run significantly faster
        self.pose = self.mp_pose.Pose(
            static_image_mode=True, 
            model_complexity=0,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def calculate_angle(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    def is_falling(self, bbox, landmarks):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        aspect_ratio = width / height if height > 0 else 0
        
        if landmarks:
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_mid_x = (left_hip.x + right_hip.x) / 2
            hip_mid_y = (left_hip.y + right_hip.y) / 2
            
            torso_angle = abs(self.calculate_angle((shoulder_mid_x, shoulder_mid_y), (hip_mid_x, hip_mid_y)))
            horizontal_torso = torso_angle < 45 or torso_angle > 135
        else:
            horizontal_torso = False

        return aspect_ratio > 1.2 or (aspect_ratio > 0.8 and horizontal_torso)

    def process_video(self, input_path, output_path, skip_frames=2):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video source {input_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video at original speed... Press 'q' to stop.")
        
        frame_count = 0
        cached_detections = [] # Stores data to draw during skipped frames
        global_fall_alert = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # SPEED FIX: Only run the heavy AI models every N frames
            if frame_count % skip_frames == 0:
                cached_detections.clear()
                global_fall_alert = False
                
                # Resize frame internally for faster YOLO inference (optional, but helps)
                results = self.yolo_model(frame, classes=[0], verbose=False)
                
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Pad the crop slightly to give MediaPipe context
                        pad = 20
                        crop_x1, crop_y1 = max(0, x1 - pad), max(0, y1 - pad)
                        crop_x2, crop_y2 = min(width, x2 + pad), min(height, y2 + pad)
                        
                        person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        if person_crop.size == 0: continue

                        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                        pose_results = self.pose.process(person_rgb)

                        is_fall = self.is_falling((x1, y1, x2, y2), pose_results.pose_landmarks)
                        if is_fall: global_fall_alert = True
                        
                        # Save the detection data to draw it on current AND skipped frames
                        cached_detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'crop_coords': (crop_x1, crop_y1, crop_x2, crop_y2),
                            'landmarks': pose_results.pose_landmarks,
                            'is_fall': is_fall
                        })

            # --- DRAWING PHASE (Runs every frame to keep video smooth) ---
            for det in cached_detections:
                x1, y1, x2, y2 = det['bbox']
                crop_x1, crop_y1, crop_x2, crop_y2 = det['crop_coords']
                
                color = (0, 0, 255) if det['is_fall'] else (0, 255, 0)
                label = "FALL DETECTED!" if det['is_fall'] else "Normal"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if det['landmarks']:
                    # Draw structural skeleton
                    for connection in self.mp_pose.POSE_CONNECTIONS:
                        start_lm = det['landmarks'].landmark[connection[0]]
                        end_lm = det['landmarks'].landmark[connection[1]]
                        
                        s_x = int(start_lm.x * (crop_x2 - crop_x1)) + crop_x1
                        s_y = int(start_lm.y * (crop_y2 - crop_y1)) + crop_y1
                        e_x = int(end_lm.x * (crop_x2 - crop_x1)) + crop_x1
                        e_y = int(end_lm.y * (crop_y2 - crop_y1)) + crop_y1
                        
                        cv2.line(frame, (s_x, s_y), (e_x, e_y), (255, 255, 0), 2)

            if global_fall_alert:
                cv2.putText(frame, "ALERT: FALL DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            out.write(frame)
            cv2.imshow('Fast Fall Detection', frame)
            
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Done. Output saved to {output_path}")

if __name__ == "__main__":
    detector = FastFallDetector(yolo_model_path='yolov8n.pt') 
    
    input_video_source = 'fall.mp4' 
    output_video_destination = 'output_fast_fall_detection.mp4'
    
    # skip_frames=2 means the AI runs every 2nd frame, instantly doubling your FPS.
    # The bounding boxes will seamlessly track over the skipped frame.
    detector.process_video(input_video_source, output_video_destination, skip_frames=2)