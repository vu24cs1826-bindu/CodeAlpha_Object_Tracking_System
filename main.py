import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Open video
cap = cv2.VideoCapture("vedios/test.mp4")

# Check video
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Save output video
out = cv2.VideoWriter(
    "outputs/tracked_output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# Create resizable window
cv2.namedWindow("Object Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Tracking", 700, 900)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection
    results = model(frame, conf=0.3)

    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append(
                (
                    [float(x1), float(y1),
                     float(x2 - x1), float(y2 - y1)],
                    conf,
                    str(cls)
                )
            )

    # Deep SORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Draw ID
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Save frame to output video
    out.write(frame)

    # Resize for display
    display_frame = cv2.resize(frame, (500, 800))

    # Show video
    cv2.imshow("Object Tracking", display_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Output video saved successfully!")
print("📁 Location: outputs/tracked_output.mp4")