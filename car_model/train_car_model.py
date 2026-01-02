import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# -------- 1. Paths and Save Folder --------
DATA_DIR = 'vmmrdb'       # Update if your dataset is elsewhere
SAVE_DIR = 'model'        # Folder to save models
FINAL_MODEL_PATH = os.path.join(SAVE_DIR, 'car_model_classifier.pth')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# -------- 2. Hyperparameters --------
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

# -------- 3. Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# -------- 4. Transforms --------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------- 5. Dataset and Dataloader --------
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
num_classes = len(dataset.classes)
print(f"✅ Found {len(dataset)} images across {num_classes} classes.")

# -------- 6. Load ResNet50 --------
model = models.resnet50(weights='IMAGENET1K_V1')

# Freeze feature extractor layers
for param in model.parameters():
    param.requires_grad = False

# Replace the fully connected layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

print("✅ Initialized ResNet50 model (pretrained on ImageNet).")

# -------- 7. Loss and Optimizer --------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# -------- 8. Training Loop --------
print("🚀 Starting training...")
model.train()

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1} completed — Loss: {avg_loss:.4f}")

    # Save each epoch checkpoint
    epoch_path = os.path.join(SAVE_DIR, f'epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), epoch_path)
    print(f"💾 Saved checkpoint: {epoch_path}")

# -------- 9. Save Final Model --------
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"🏁 Final model saved to: {FINAL_MODEL_PATH}")
