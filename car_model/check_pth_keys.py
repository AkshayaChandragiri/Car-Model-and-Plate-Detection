import torch

# Load the .pth file
weights = torch.load('model/car_model_classifier.pth', map_location='cpu')

# Print out the keys
print("\n🔑 Top-level keys in the .pth file:\n")
print(weights.keys())

# If it's a full dict, it might have 'model_state_dict' or 'state_dict' inside
if 'model_state_dict' in weights:
    print("\n📦 Keys inside 'model_state_dict':\n")
    print(weights['model_state_dict'].keys())
elif 'state_dict' in weights:
    print("\n📦 Keys inside 'state_dict':\n")
    print(weights['state_dict'].keys())
elif isinstance(weights, dict) and all(k.startswith('model.') or k.startswith('layer') for k in weights.keys()):
    print("\n✅ The file contains raw model weights directly.")
else:
    print("\n⚠️ Unexpected format inside the .pth file.")
