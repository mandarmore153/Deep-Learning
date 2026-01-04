import cv2
from ultralytics import YOLO

from traffic_count import DISPLAY_HEIGHT, DISPLAY_WIDTH

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "best.pt"  # Path to your fine-tuned model
VIDEO_PATH = "videos/vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4"
OUTPUT_PATH = "output/inference_result.mp4"  # Path to save the output video

# -------------------------------
# Load Model
# -------------------------------
model = YOLO(MODEL_PATH)

# -------------------------------
# Video Setup
# -------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

# Check if the video file is loaded successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# -------------------------------
# Perform Inference
# -------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the current frame
    results = model.predict(source=frame, save=False)

    # Draw bounding boxes and labels on the frame
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            class_name = model.names[class_id]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add class label
            cv2.putText(
                frame,
                f"{class_name}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("Vehicle Counting", display_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()