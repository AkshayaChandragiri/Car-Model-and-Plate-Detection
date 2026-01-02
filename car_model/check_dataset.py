from torchvision import datasets, transforms
import os

data_dir = 'vmmrdb'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

try:
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    print(f"✅ Found {len(dataset)} images across {len(dataset.classes)} classes.")
    print("First few classes:", dataset.classes[:5])
except Exception as e:
    print("❌ ERROR:", e)
