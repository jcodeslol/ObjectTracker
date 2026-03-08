import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --------------------------
# Parameters
# --------------------------
input_video_path = "input.mp4"
output_path = "output.mp4"

# --------------------------
# Load YOLOv8 model
# --------------------------
model = YOLO('yolov8n.pt')  # small, fast model

# --------------------------
# Initialize DeepSORT tracker
# --------------------------
tracker = DeepSort(max_age=30)

# --------------------------
# Open video
# --------------------------
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {input_video_path}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback if FPS is 0
print(f"Video opened: {width}x{height} at {fps} FPS")

# --------------------------
# Video writer with MP4V codec (more compatible)
# --------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for broader compatibility
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    print(f"Error: Could not open video writer for {output_path}. Try another codec or check permissions.")
    cap.release()
    exit()

# --------------------------
# Counting line setup
# --------------------------
entry_line_y = int(height / 3)
count = 0
frame_idx = 0

print("Starting video processing loop...")
while True:
    ret, frame = cap.read()
    if not ret:
        print(f"End of video or error reading frame at frame {frame_idx}.")
        break

    # Resize frame (just in case)
    frame = cv2.resize(frame, (width, height))

    # YOLO detection
    results = model(frame)
    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf

    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()

    # Prepare detections for DeepSORT
    detections = [[b.tolist(), float(s)] for b, s in zip(boxes_np, scores_np)]

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracks and count objects
    for track in tracks:
        bbox = track.to_ltrb()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
        cv2.putText(frame, f'ID:{track.track_id}', (int(bbox[0]), int(bbox[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        centroid_y = (bbox[1]+bbox[3])/2
        if centroid_y < entry_line_y and not hasattr(track, 'passed'):
            count += 1
            track.passed = True

    # Draw counting line
    cv2.line(frame, (0, entry_line_y), (width, entry_line_y), (0,255,255), 2)

    # Write frame
    out.write(frame)
    frame_idx += 1

print("Video processing loop finished. Releasing resources...")
# Release resources
cap.release()
out.release()
print(f'Tracking finished! Output saved to {output_path}')
print(f'Total count across line: {count}')