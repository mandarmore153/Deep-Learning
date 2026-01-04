import cv2
from ultralytics import YOLO
from utils.line_counter import LineCounter

# -------------------------------
# Configuration
# -------------------------------
VIDEO_PATH = "videos/vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4"
OUTPUT_PATH = "output/result.mp4"
# MODEL_PATH = "yolov8n.pt"
MODEL_PATH = "best.pt"  # Fine-tuned model

COUNT_LINE_Y = 500  # Adjust based on video 350
DISPLAY_WIDTH = 960   # or 800
DISPLAY_HEIGHT = 540 

# Vehicle class IDs in COCO dataset
# VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck  original model
VEHICLE_CLASSES = [0, 1]  # car, truck # Adjusted for best.pt model

# -------------------------------
# Load Model
# -------------------------------
model = YOLO(MODEL_PATH)

# -------------------------------
# Video Setup
# -------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# -------------------------------
# Line Counter
# -------------------------------
line_counter = LineCounter(COUNT_LINE_Y)

# -------------------------------
# Process Video
# -------------------------------
results = model.track(
    source=VIDEO_PATH,
    tracker="bytetrack.yaml",
    persist=True,
    stream=True
)

for result in results:
    frame = result.orig_img

    # Draw counting line
    cv2.line(
        frame,
        (0, COUNT_LINE_Y),
        (width, COUNT_LINE_Y),
        (0, 255, 255),
        2
    )

    if result.boxes.id is not None:
        for box, track_id, cls in zip(
            result.boxes.xyxy,
            result.boxes.id,
            result.boxes.cls
        ):
            class_id = int(cls)

            if class_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)
            center_y = (y1 + y2) // 2
            track_id = int(track_id)

            # Count vehicle
            line_counter.count_vehicle(track_id, center_y, class_id)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Display class-wise counts
    class_counts = line_counter.get_class_counts()
    y_offset = 40
    # for class_id, count in class_counts.items():                          #Original model            
    for class_name, count in class_counts.items():
        # class_name = model.names[class_id]  # Get class name from model--- original model
        cv2.putText(
            frame,
            f"{class_name}: {count}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )
        y_offset += 40

    out.write(frame)

    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("Vehicle Counting", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
