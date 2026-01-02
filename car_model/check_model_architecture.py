import torch
from torchvision import models

model_path = 'model/car_model_classifier.pth'   # adjust if needed
state_dict  = torch.load(model_path, map_location='cpu')

# ---------- helper to decide which backbone ----------
def guess_backbone(sd):
    """Return 18 if fc.weight has 512 in_features else 50."""
    fc_w = sd[[k for k in sd if k.endswith('fc.weight')][0]]
    in_features = fc_w.shape[1]    # columns = in_features
    return 18 if in_features == 512 else 50

backbone = guess_backbone(state_dict)
print(f"🧐 Detected backbone from weight shape: ResNet{backbone}")

# ---------- build that backbone object ----------
if backbone == 18:
    net = models.resnet18()
else:
    net = models.resnet50()

# swap final layer size to match your dataset
num_classes = state_dict[[k for k in state_dict if k.endswith('fc.weight')][0]].shape[0]
net.fc = torch.nn.Linear(net.fc.in_features, num_classes)

# load the weights
net.load_state_dict(state_dict, strict=True)

print("\n✅ Full model after loading:")
print(net)
