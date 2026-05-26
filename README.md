# 😊 9 Facial Expression Detection & Classification using YOLOv8

A deep learning project for **detecting and classifying 9 different facial expressions** using **YOLOv8**, OpenCV, and PyTorch.

This project includes:

- ✅ Dataset preprocessing
- ✅ Image visualization and analysis
- ✅ Corrupt image detection
- ✅ YOLOv8 training and fine-tuning
- ✅ Real-time webcam prediction
- ✅ Model evaluation and visualization
- ✅ Exporting and saving trained models

---

# 📌 Table of Contents

1. Project Overview  
2. Features  
3. Technologies Used  
4. Dataset Structure  
5. Image Preprocessing  
6. Training YOLOv8 Model  
7. Model Evaluation Metrics  
8. Real-Time Webcam Detection  
9. Prediction on Images  
10. Visualization  
11. Fine-Tuning  
12. Saving & Exporting Model  
13. Results  
14. Future Improvements  
15. Installation  
16. How to Run  

---

# 🎯 Project Overview

This project focuses on **facial expression recognition** using the **YOLOv8 object detection model**.

The model is trained to detect and classify the following **9 facial expressions**:

| Expression |
|---|
| 😀 Happy |
| 😢 Sad |
| 😠 Angry |
| 😨 Fear |
| 🤢 Disgust |
| 😴 Sleepy |
| 😐 Natural |
| 😏 Contempt |
| 😲 Surprise |

The system can:

- Detect faces in images/video
- Classify facial expressions
- Run predictions in real time using a webcam
- Evaluate model performance using mAP, Precision, and Recall

---

# 🚀 Features

✨ Detects and classifies 9 facial expressions  
✨ Real-time webcam inference  
✨ Dataset preprocessing pipeline  
✨ Automatic corrupt image detection  
✨ Image size distribution analysis  
✨ YOLOv8 training with augmentation  
✨ Model evaluation using confusion matrix  
✨ Fine-tuning support  
✨ Export trained models  
✨ Kaggle GPU training support  

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| OpenCV (`cv2`) | Image processing & webcam handling |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Pillow (PIL) | Image handling |
| PyTorch | Deep learning framework |
| Ultralytics YOLOv8 | Object detection model |
| Kaggle GPU | Model training |

---

# 📂 Dataset Structure

The dataset follows the YOLO format:

```bash
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

---

# 🖼️ Image Preprocessing

The preprocessing pipeline performs:

- ✅ Image resizing
- ✅ RGB/BGR conversion
- ✅ Optional grayscale conversion
- ✅ Image normalization
- ✅ Corrupt image detection
- ✅ Label preservation

---

## 🔍 Counting Total Images

The code collects image paths from:

- Train folder
- Validation folder
- Test folder

Supported image formats:

```python
.png
.jpg
.jpeg
```

---

## 📊 Image Size Distribution

Histograms are generated to analyze:

- Width distribution
- Height distribution

This helps identify:

- Inconsistent image sizes
- Dataset imbalance
- Need for resizing

---

## 🚨 Corrupt Image Detection

The project automatically checks for broken or unreadable images using:

```python
img.verify()
```

Benefits:

- ✅ Prevents training crashes
- ✅ Improves dataset quality
- ✅ Removes invalid images

---

## 🖼️ Visualizing Dataset Images

The first few images are displayed using Matplotlib for quick inspection.

Example:

```python
plt.imshow(img)
```

This helps verify:

- Correct labels
- Proper image loading
- Dataset quality

---

# 🔄 Image Preprocessing Function

The `preprocess_image()` function performs:

| Operation | Description |
|---|---|
| Read Image | Load image using OpenCV |
| Convert Color | BGR → RGB |
| Resize | Resize to 640×640 |
| Normalize | Scale pixels to 0–1 |
| Save Image | Store processed image |

---

# ⚡ PyTorch GPU Acceleration

The project supports:

## 🍎 Apple Silicon (MPS)

```python
torch.backends.mps.is_available()
```

Used for Mac GPU acceleration.

---

## 🎮 NVIDIA CUDA

```python
torch.cuda.is_available()
```

Used for Kaggle GPU training.

---

# 🤖 YOLOv8 Model Training

The project uses:

```python
YOLO("yolov8s.pt")
```

### Why YOLOv8s?

- ✅ Lightweight
- ✅ Faster training
- ✅ Good accuracy
- ✅ Suitable for smaller datasets

---

# 🧠 Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Image Size | 416 |
| Batch Size | 32 |
| Device | GPU |
| Augmentation | Enabled |
| Mosaic | Enabled |
| Mixup | Enabled |

---

# 🎨 Data Augmentation

The model uses multiple augmentation techniques:

| Augmentation | Purpose |
|---|---|
| Rotation | Improve robustness |
| Translation | Improve localization |
| Scaling | Handle different face sizes |
| Mosaic | Better generalization |
| Mixup | Reduce overfitting |
| Horizontal Flip | Improve variation |

---

# 📈 Model Performance

## 📊 Overall Metrics

| Metric | Score |
|---|---|
| Precision | 0.771 |
| Recall | 0.739 |
| mAP@0.5 | 0.825 ✅ |
| mAP@0.5:0.95 | 0.639 ✅ |

---

## 🧾 Metric Explanation

### 🎯 Precision
Measures how many predicted detections are correct.

### 🔍 Recall
Measures how many actual expressions were detected.

### 📈 mAP@0.5
Measures detection accuracy at IoU threshold 0.5.

### 📊 mAP@0.5:0.95
Measures stricter detection performance across multiple IoU thresholds.

---

# 😃 Per-Class Performance

| Expression | mAP@0.5 | Comment |
|---|---|---|
| 😀 Happy | 0.949 | Best performing |
| 😴 Sleepy | 0.885 | High recall |
| 😢 Sad | 0.849 | Stable |
| 🤢 Disgust | 0.802 | Balanced |
| 😨 Fear | 0.835 | Good detection |
| 😏 Contempt | 0.694 | Needs improvement |
| 😐 Natural | 0.702 | Slightly weak |

---

# 📷 Real-Time Webcam Detection

The project supports live webcam inference using OpenCV.

Features:

- ✅ Live facial expression detection
- ✅ Bounding boxes
- ✅ Confidence scores
- ✅ Real-time predictions

### Exit Key

```python
Press 'q' to quit
```

---

# 🖼️ Predicting on Images

Single image prediction:

```python
results = model("image.jpg")
```

Displays:

- ✅ Bounding boxes
- ✅ Predicted class
- ✅ Confidence score

---

# 📦 Prediction Output

The model provides:

| Output | Description |
|---|---|
| `xyxy` | Bounding box coordinates |
| `conf` | Confidence scores |
| `cls` | Predicted class IDs |

---

# 📊 Model Evaluation

Evaluation is performed using:

```python
metrics = model.val()
```

This calculates:

- ✅ Precision
- ✅ Recall
- ✅ mAP
- ✅ Validation loss

---

# 🖼️ Visualization

The project visualizes:

- ✅ Confusion matrix
- ✅ Predicted images
- ✅ Detection outputs

---

# 🔁 Fine-Tuning & Retraining

The model can be retrained using previous checkpoints:

```python
best.pt
```

Fine-tuning improvements:

- ✅ Lower learning rate
- ✅ Additional augmentation
- ✅ More epochs
- ✅ Better generalization

---

# 💾 Saving the Model

The trained model is saved as:

```bash
trained_model.pt
```

---

# 🗜️ Exporting Model

The project also:

- ✅ Creates ZIP files for trained weights
- ✅ Allows easy download from Kaggle
- ✅ Stores `best.pt` and `last.pt`

---

# 📁 Generated Files

| File | Purpose |
|---|---|
| `trained_model.pt` | Final trained model |
| `best.pt` | Best validation model |
| `last.pt` | Last training checkpoint |
| `confusion_matrix.png` | Evaluation visualization |
| `results.png` | Training results |

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone <your-repository-url>
cd facial-expression-detection
```

---

## 2️⃣ Install Dependencies

```bash
pip install ultralytics
pip install torch torchvision torchaudio
pip install opencv-python
pip install matplotlib
pip install pillow
pip install numpy
```

---

# ▶️ How to Run

---

## 🧹 Preprocess Dataset

```bash
python preprocess.py
```

---

## 🚀 Train Model

```bash
python train.py
```

---

## 📷 Webcam Detection

```bash
python webcam_detection.py
```

---

## 🖼️ Predict on Image

```bash
python predict.py
```

---

# 📌 Example Training Command

```python
model.train(
    data="data.yaml",
    epochs=10,
    imgsz=416,
    batch=32
)
```

---

# 🧪 Example Prediction Command

```python
results = model.predict(
    source="test/images",
    conf=0.25,
    save=True
)
```

---

# 📊 Example Results

- ✅ Accurate facial expression detection
- ✅ Strong performance on happy and sleepy expressions
- ✅ Real-time inference support
- ✅ Good generalization using augmentation

---

# 🔮 Future Improvements

🚀 Increase dataset size  
🚀 Improve weak classes like contempt  
🚀 Deploy as web application  
🚀 Convert to TensorRT/ONNX  
🚀 Add emotion tracking over time  
🚀 Mobile deployment  

---

# 👩‍💻 Author

**Natasha Shrestha**

Facial Expression Detection & Classification using YOLOv8

---

# ⭐ Acknowledgements

Special thanks to:

- Ultralytics YOLOv8
- PyTorch
- OpenCV
- Kaggle GPU platform

---

# 📜 License

This project is for educational and research purposes.
