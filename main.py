import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from yolov5 import YOLOv5
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30, n_init=3)

# Load gaze estimation model (example placeholder)
# gaze_model = ... (Load your gaze model here)

# Open the test video
video_path = r'C:\Users\negia\OneDrive\Desktop\optimized inference pipeline\ABA Therapy_ Daniel - Communication.mp4'

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # Filter for person detections
    person_detections = [det for det in detections if det[5] == 0]  # Class '0' is person

    # Prepare detections for tracker
    detection_for_tracker = []
    for det in person_detections:
        bbox = det[:4]
        conf = det[4]
        detection_for_tracker.append((bbox, conf))

    # Update tracker
    tracks = tracker.update_tracks(detection_for_tracker, frame=frame)

    # Loop through tracks to display bounding boxes, IDs, and gaze estimation
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        bbox = track.to_tlbr()

        # Example: Draw bounding box and ID
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Extract face and perform gaze estimation (placeholder)
        # face_img = extract_face(frame, bbox)
        # gaze_direction = gaze_model.predict(face_img)

        # Example: Draw gaze direction (placeholder)
        # cv2.arrowedLine(frame, (center_x, center_y), (gaze_x, gaze_y), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
