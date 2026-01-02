import torch
from torchvision import models, transforms
from PIL import Image
import os

# Set path
MODEL_PATH = 'model/car_model_classifier.pth'
CLASS_NAMES = sorted(os.listdir('vmmrdb'))  # same order as training

# Load model
resnet50 = models.resnet50(pretrained=False)
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, len(CLASS_NAMES))
resnet50.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
resnet50.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_car_model(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = resnet50(image)
        _, predicted = torch.max(output, 1)

    return CLASS_NAMES[predicted.item()]
