# 9 Facial Expression Detection & Classification using YOLOv8

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

# Justification for Selecting YOLOv8 for Facial Expression Recognition (FER)

## Introduction

During the research and development phase of the Facial Expression Recognition (FER) project, multiple deep learning architectures and computer vision approaches were researched and evaluated before selecting the final model architecture. The primary objective of the project was to develop a system capable of detecting and classifying human facial expressions accurately and efficiently in real-time environments.

After analyzing different architectures, YOLOv8 (You Only Look Once Version 8) was selected as the primary model for the FER system. This document provides a detailed justification for choosing YOLOv8 over other alternative approaches and explains how the dataset structure, including the `.yaml` configuration file, aligned effectively with the YOLOv8 training pipeline.

---

# Research and Evaluation of Multiple Architectures

Before selecting YOLOv8, several architectures and methodologies were researched, including:

- Traditional CNN-based classification models
- ResNet architectures
- VGGNet architectures
- MobileNet
- Haar Cascade with CNN
- OpenCV-based approaches
- YOLO object detection models

Each architecture was evaluated based on the following criteria:

- Detection accuracy
- Real-time performance
- Computational efficiency
- Ease of deployment
- Dataset compatibility
- Scalability
- Training complexity
- Inference speed

The project required not only facial expression classification but also accurate face localization and detection in real-time video streams. Therefore, models that only performed image classification without detection capabilities were considered less suitable for the project objectives.

---

# Why YOLOv8 Was Selected

## 1. Real-Time Detection Capability

One of the primary reasons for selecting YOLOv8 was its excellent real-time object detection performance.

Facial Expression Recognition systems often need to:

- Detect faces instantly
- Process webcam or video frames continuously
- Predict expressions with minimal delay

YOLOv8 is specifically designed for high-speed object detection while maintaining strong accuracy. Unlike traditional CNN classifiers that require separate face detection pipelines, YOLOv8 performs:

- Face localization
- Bounding box prediction
- Expression classification

within a single unified architecture.

This significantly improved:

- Processing speed
- System efficiency
- Real-time responsiveness

which was essential for the project requirements.

---

## 2. Single-Stage Detection Architecture

YOLOv8 uses a single-stage detection mechanism, meaning detection and classification occur simultaneously while processing the image only once.

This differs from two-stage detectors such as Faster R-CNN, where:

1. Region proposals are generated
2. Classification occurs afterward

The single-stage approach provides:

- Faster inference speed
- Lower latency
- Better suitability for live applications

Since the FER project focused on real-time webcam detection, minimizing latency was extremely important.

---

## 3. Improved Accuracy and Feature Extraction

YOLOv8 provides several improvements compared to earlier YOLO versions, including:

- Better feature extraction
- Improved anchor-free detection
- Enhanced loss functions
- Better small object detection
- Improved training optimization

Facial expressions involve subtle facial movements such as:

- Smiling
- Eyebrow movement
- Eye narrowing
- Lip movement

These subtle changes require strong feature extraction capabilities. YOLOv8 demonstrated better performance in learning fine-grained facial features from the dataset compared to older object detection architectures.

---

## 4. Efficient and Simplified Training Pipeline

YOLOv8 provides a streamlined and beginner-friendly training workflow through the Ultralytics framework.

The framework simplified:

- Dataset preparation
- Model training
- Validation
- Inference
- Model exporting

This allowed faster experimentation and model iteration during development.

The command-based training structure reduced implementation complexity and enabled rapid testing of:

- Epoch values
- Image sizes
- Batch sizes
- Data augmentations
- Hyperparameters

---

## 5. Compatibility with the Dataset YAML File

Another major reason for selecting YOLOv8 was its compatibility with the dataset structure.

The FER dataset used in the project included:

- Training images
- Validation images
- Label files
- YAML configuration file

YOLOv8 natively supports datasets organized with YAML configuration files.

---

# Importance of the YAML File

The `.yaml` file plays a critical role in YOLOv8 training because it defines:

- Dataset paths
- Training image locations
- Validation image locations
- Number of classes
- Class names

Example structure:

```yaml
train: ../train/images
val: ../valid/images

nc: 7

names:
  0: angry
  1: disgust
  2: fear
  3: happy
  4: neutral
  5: sad
  6: surprise
```

The YAML file allowed YOLOv8 to:

- Automatically load the dataset
- Understand class mappings
- Configure the training pipeline correctly

This reduced manual configuration effort and improved dataset management efficiency.

---

## 6. Strong Community Support and Documentation

YOLOv8 has:

- Extensive documentation
- Large developer community
- Frequent updates
- Strong industry adoption

This was beneficial during:

- Debugging
- Training optimization
- Model evaluation
- Deployment integration

The availability of tutorials, GitHub repositories, and community discussions accelerated the learning and development process.

---

## 7. Scalability and Deployment Advantages

The project also considered future scalability and deployment requirements.

YOLOv8 supports:

- Export to ONNX
- TensorRT optimization
- Deployment with FastAPI
- Integration with OpenCV
- Edge device compatibility

These capabilities aligned with the project’s long-term objective of building a deployable FER application.

The lightweight inference capability also made YOLOv8 suitable for:

- Real-time webcam systems
- Cloud deployment
- API integration

---

## 8. Better Integration with OpenCV and FastAPI

The project architecture used:

- OpenCV for webcam frame processing
- FastAPI for backend deployment

YOLOv8 integrates effectively with both technologies.

Benefits included:

- Simple prediction APIs
- Easy frame-by-frame inference
- Faster integration into backend systems
- Simplified deployment pipeline

This reduced development complexity and improved maintainability of the application.

---

# Comparison with Other Architectures

| Architecture | Advantages | Limitations |
|---|---|---|
| Traditional CNN | Good classification accuracy | No real-time detection |
| ResNet | Deep feature extraction | Computationally expensive |
| VGGNet | Strong image learning | Large model size |
| Haar Cascade + CNN | Lightweight | Lower detection accuracy |
| Faster R-CNN | High accuracy | Slower inference |
| MobileNet | Lightweight | Lower detection precision |
| YOLOv8 | Real-time, accurate, scalable | Requires GPU for faster training |

After evaluation, YOLOv8 provided the best balance between:

- Speed
- Accuracy
- Real-time detection
- Ease of deployment
- Dataset compatibility

---

# Conclusion

YOLOv8 was selected as the primary architecture for the Facial Expression Recognition system because it effectively satisfied the project’s technical and practical requirements.

The model provided:

- Real-time face detection
- Accurate facial expression classification
- Efficient training workflow
- Seamless YAML dataset integration
- Strong scalability and deployment support

Its compatibility with OpenCV and FastAPI further strengthened its suitability for the overall system architecture.

The inclusion of the YAML dataset configuration file significantly simplified dataset management and aligned perfectly with the Ultralytics YOLOv8 training ecosystem. After comparing multiple architectures, YOLOv8 was determined to be the most efficient, scalable, and practical solution for implementing a real-time Facial Expression Recognition application.
