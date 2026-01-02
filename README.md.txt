# 
 Vehicle Model and Number Plate Detection System

This project implements a real-time computer vision pipeline to detect vehicles, recognize license plates, and identify vehicle models using deep learning.

##  Features
- Vehicle detection using YOLOv5
- License plate detection and recognition using EasyOCR
- Vehicle model classification using ResNet-based CNN
- Works on images and real-time video streams

##  Technologies Used
- Python
- PyTorch
- YOLOv5
- OpenCV
- EasyOCR
- ResNet50

#Uses YOLOv5
YOLOv5 is used as the object detection backbone and should be cloned separately from the official Ultralytics repository.


##  Project Structure
## 📂 Project Structure

Vehicle_Model_and_Plate_Detection/
│
├── main.py
│   └── Entry point of the project. Runs vehicle detection, number plate detection, 
│       OCR, and vehicle model classification together.
│
├── requirements.txt
│   └── List of Python dependencies required to run the project.
│
├── yolov5/
│   └── YOLOv5 framework used for vehicle detection.
│       (Only source code is included, trained weights are excluded.)
│
├── yolov5_plate/
│   └── YOLO-based license plate detection module.
│
├── car_model/
│   └── Vehicle model classification logic (ResNet-based).
│       Includes training and inference scripts.
│
├── images/
│   └── Sample input images for testing the system.
│
├── results/
│   └── Sample output images showing detected vehicles,
│       number plates, and predicted vehicle models.
│
├── .gitignore
│   └── Prevents large files, datasets, virtual environments,
│       and model weights from being uploaded to GitHub.
│
└── README.md
    └── Project documentation.

