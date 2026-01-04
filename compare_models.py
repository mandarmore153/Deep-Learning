from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

# Fine-tuned model
model_ft = YOLO("C:\\Users\\onkar\\Downloads\\best.pt")
metrics_ft = model_ft.val(data="car&truck-detection-1/data.yaml")

# Pretrained model
model_pre = YOLO("yolov8n.pt")
metrics_pre = model_pre.val(data="car&truck-detection-1/data.yaml")

comparison_data = {
    "Model": ["YOLOv8 Pretrained", "YOLOv8 Fine-tuned"],
    "mAP@0.5": [
        metrics_pre.box.map50,
        metrics_ft.box.map50
    ],
    "mAP@0.5:0.95": [
        metrics_pre.box.map,
        metrics_ft.box.map
    ],
    "Precision": [
        metrics_pre.box.mp,
        metrics_ft.box.mp
    ],
    "Recall": [
        metrics_pre.box.mr,
        metrics_ft.box.mr
    ]
}
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv("model_comparison.csv", index=False)

print(comparison_df)

comparison_df.set_index("Model").plot(kind="bar", figsize=(10,6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.tight_layout()
plt.show()