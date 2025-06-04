# 🖐️ Hand Pose Estimation Using Deep Learning

## 📌 Overview
This project implements real-time **hand pose estimation** using a custom dataset collected from self-recorded videos. The system predicts 2D keypoints (joints) of the hand in each frame, enabling gesture recognition, HCI applications, and real-time control interfaces.

## 🎥 Dataset
- **Source:** Collected from videos recorded by me
- **Preprocessing:** Extracted keyframes using OpenCV
- **Annotation:** Labeled finger joint positions using Label Studio
- **Format:** Keypoint annotations in CSV

## 🧠 Model Details
- **Framework:** TensorFlow / PyTorch
- **Architecture:** CNN-based keypoint regression or use of pre-trained models (e.g., BlazePose, MediaPipe Hands)
- **Training:** Supervised learning on 21 hand keypoints
- **Output:** (x, y) coordinates for each joint per frame
