# 🚦 Vehicle Counting and Traffic Analysis System

A computer vision–based system for detecting, tracking, and counting vehicles in traffic video streams using YOLOv8 and ByteTrack. This project is suitable for urban planning, congestion analysis, and smart city applications.

---

## ✨ Key Features

- Vehicle detection using YOLOv8
- Multi-object tracking using ByteTrack
- Line-crossing–based vehicle counting
- Supports pretrained and fine-tuned models
- Class-wise counting (car, truck, bus, etc.)
- Annotated video output generation
- Quantitative model comparison (CSV-based)
- Modular, clean, GitHub-friendly structure

---

## 🧠 Tech Stack

- **Python 3.9+**
- **Ultralytics YOLOv8**
- **OpenCV**
- **ByteTrack**
- **NumPy**
- **Pandas** (for evaluation & comparison)
- **Matplotlib** (for visualization)

---

## 📁 Project Structure

```
Vehicle Counting & Traffic Analysis System/
│
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── traffic_count.py         # Main script for vehicle counting
├── inference.py             # Script for running inference
├── fine_tune.ipynb          # Notebook for fine-tuning the model
├── compare_models.py        # Script for model comparison
├── model_comparison.csv     # CSV file for model comparison results
│
├── utils/                   # Utility scripts
│   ├── __init__.py
│   └── line_counter.py      # Line crossing logic
│
├── car&truck-detection-1/   # Dataset folder
│   ├── data.yaml            # Dataset configuration
│   ├── test/                # Test data
│   ├── train/               # Training data
│   └── valid/               # Validation data
│
├── output/                  # Output files
│   ├── result.mp4           # Annotated output video
│   └── runs/                # YOLOv8 training and inference runs
│
├── videos/                  # Input videos
│   └── traffic.mp4          # Example input video
│
└── yolov8n.pt               # Pretrained YOLOv8 model
```

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/vehicle-counting-yolo.git
cd Vehicle Counting & Traffic Analysis System
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎥 Input Video

Place your traffic video inside the `videos/` folder:

```
videos/traffic.mp4
```
> Note: This repository includes a `.gitignore` that excludes the `videos/` directory and common video file types (e.g. .mp4, .avi, .mov, .mkv). If you need to track large video files in Git, remove them from `.gitignore` or add them explicitly.

You can download free traffic videos from:

- [Pexels](https://www.pexels.com/)
- [Pixabay](https://pixabay.com/)
- [YouTube](https://www.youtube.com/) (traffic CCTV footage)

---

## ▶️ How to Run the Project

Run the following command to start the vehicle counting system:

```bash
python traffic_count.py
```

After execution, the output video will be saved as:

```
output/result.mp4
```

The video will display:

- Bounding boxes around vehicles
- Unique IDs for each vehicle
- A counting line
- Total vehicle count

---

## 🧠 How It Works

### 🔹 1. Vehicle Detection

YOLOv8 detects vehicles frame-by-frame using a pretrained COCO dataset.

### 🔹 2. Vehicle Tracking

ByteTrack assigns a unique ID to each detected vehicle, allowing it to be tracked across frames.

### 🔹 3. Counting Logic

- A virtual horizontal line is drawn on the road.
- Each vehicle is counted only once when its centroid crosses the line.
- Tracking IDs prevent duplicate counting.

### 🔹 4. Output Generation

The processed video with overlays is automatically saved using OpenCV’s `VideoWriter`.

---

## 📊 Model Evaluation & Comparison

This project compares:

- **YOLOv8 Pretrained model**
- **YOLOv8 Fine-tuned model**

### Metrics Used

- **mAP@0.5**
- **mAP@0.5:0.95**
- **Precision**
- **Recall**

### Run Comparison

```bash
python scripts/compare_models.py
```

### Sample Results

| Model               | mAP@0.5 | Precision | Recall |
|---------------------|---------|-----------|--------|
| YOLOv8 Pretrained   | 0.004   | 0.008     | 0.026  |
| YOLOv8 Fine-tuned   | 0.916   | 0.873     | 0.859  |

📌 **Conclusion**: Fine-tuning significantly improves detection and counting accuracy.


## ⚙️ Training Environment & Hardware

Due to local machine hardware limitations, the YOLOv8 fine-tuning process was performed using **Google Colab**.

Google Colab provides access to **free GPU resources**, which significantly reduced training time and enabled efficient experimentation with model parameters.

### Training Setup
- Platform: Google Colab
- GPU: NVIDIA Tesla T4 (Colab)
- Framework: Ultralytics YOLOv8
- Dataset Size: ~1000 labeled images
- Image Size: 640 × 640
- Batch Size: 8
- Epochs: 30–50

The fine-tuned model (`best.pt`) was then downloaded from Colab and used for **local inference, tracking, counting, and evaluation**.

This approach demonstrates an efficient hybrid workflow combining **cloud-based training** with **local deployment**.


### 📈 Visualization

Performance metrics can be visualized using bar charts and comparison plots (see `compare_models.py`).

---

## 📊 Applications

- Smart traffic signal systems
- Traffic congestion analysis
- Urban planning & infrastructure optimization
- CCTV traffic monitoring
- Toll & parking analytics

---

## 🚀 Future Enhancements

- Class-wise vehicle count (cars vs trucks)
- Speed estimation
- RTSP / live CCTV stream support
- Web dashboard using Flask / Django
- Database logging & analytics
- Cloud deployment (AWS / Azure)

---

## 🧪 Sample Output

✔ Bounding boxes around vehicles  
✔ Unique ID for each vehicle  
✔ Vehicle count displayed in real time  

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

<!-- ## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details. -->