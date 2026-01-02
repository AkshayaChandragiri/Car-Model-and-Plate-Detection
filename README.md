# Car Model and License Plate Detection 

A real-time deep learning-based system for **vehicle model classification** and **license plate detection & recognition** using **YOLOv5, EasyOCR, and ResNet50**.

This project detects vehicles from images/video streams, identifies the **car model**, and reads the **license plate number** accurately.

## Project Overview

The system works in three main stages:

1. **Vehicle Detection**
   - Uses YOLOv5 to detect cars from images or video frames.

2. **License Plate Detection & Recognition**
   - Detects license plates using YOLOv5.
   - Reads plate text using EasyOCR.

3. **Vehicle Model Classification**
   - Cropped vehicle images are passed to a ResNet50-based CNN.
   - Predicts the exact car make and model.

## 🧠 Technologies Used

- Python
- YOLOv5
- ResNet50
- EasyOCR
- OpenCV
- PyTorch


## 📂 Project Structure
Car-Model-and-Plate-Detection/
│
├── main.py # Main pipeline script
├── requirements.txt # Python dependencies
├── README.md
│
├── yolov5/ # Vehicle & plate detection
│
├── car_model/ # Vehicle model classification
│ ├── train_car_model.py
│ ├── car_model_inference.py
│ ├── detect_from_images.py
│ ├── check_model_architecture.py
│ ├── check_pth_keys.py
│ └── README.md
│
├── images/
│ └── sample.jpg # Sample input image
│
├── results/
│ └── output_sample.jpg # Sample output


## ▶️ How to Run

### Install dependencies
pip install -r requirements.txt

Run the full pipeline
python main.py

## Sample Output
Detected vehicle bounding box

Predicted car model

Recognized license plate number

(See results/output_sample.jpg)

## Research Publication

This project is based on a published research paper:

Title:
A novel deep learning-based method for vehicle model and number plate detection in camera-captured blurred video using YOLOv5, EasyOCR, and ResNet50

Authors:
Kavitha Soppari, Akshaya Chandragiri, Abhiram Gulab, Ganesh Vaddepalli

Journal:
World Journal of Advanced Research and Reviews (WJARR)

Year: 2025
Volume: 27 (01), Pages 487–497

DOI:
https://doi.org/10.30574/wjarr.2025.27.1.2501

## Author
Akshaya Chandragiri
B.Tech – CSE (AI & ML)
ACE Engineering College, Hyderabad
