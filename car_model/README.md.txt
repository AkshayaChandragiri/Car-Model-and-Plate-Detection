# 🚗 Car Model Classification Module

This folder contains all scripts related to "vehicle model classification" using a deep learning CNN (ResNet-based) as part of the "Vehicle Model and Number Plate Detection System".

This module is responsible for "training", "verifying", and "running inference" on vehicle images to predict the exact car make and model.

## 🔍 Overview

After a vehicle is detected using YOLO, the cropped vehicle image is passed to this module.  
The trained CNN model then predicts the "vehicle make and model".

## 📁 Files Description

### 1. `train_car_model.py`
- Trains the vehicle model classification network  
- Loads dataset (e.g., VMMRDB)  
- Uses a ResNet-based architecture  
- Saves the trained `.pth` model file  

⚠️ Run this "only if you want to retrain" the model.


### 2. `car_model_inference.py`
- Loads the trained car model classifier  
- Takes a vehicle image as input  
- Outputs the predicted car model label  

✅ Used during "final pipeline integration".

---

### 3. `detect_from_images.py`
- Runs inference on test images  
- Verifies whether the trained model predicts correct vehicle models  
- Useful for offline testing and debugging

---

### 4. `check_model_architecture.py`
- Prints and verifies the model architecture  
- Ensures the loaded model matches the trained configuration  
- Helpful when resolving `state_dict` mismatch errors

---

### 5. `check_pth_keys.py`
- Inspects keys inside a `.pth` file  
- Helps debug issues such as:
  - Unexpected keys  
  - Missing layers  
  - Wrong classifier size  

📌 Very useful when switching between ResNet versions.

---

## ▶️ How to Use

### Train the model
```bash
python train_car_model.py

Run inference on images
python detect_from_images.py

Check model architecture
python check_model_architecture.py

Inspect trained weights
python check_pth_keys.py

