# infer.py
import torch
import numpy as np
from PIL import Image
import os
from utils import ContourTransform, get_resnet34_model

# ================== CONFIG ==================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models/resnet34_USP22_HGF.pth'  # Path to the saved model
IMAGE_PATH_NHT = 'data/example_NHT_HGF.tif'  # Image to predict
IMAGE_PATH_KO = 'data/example_KO_HGF.tif'  # Image to predict


# ================== INFERENCE ==================
def predict_image(model, class_names, image_path, device=DEVICE):
    img = Image.open(image_path).convert('RGB')
    transform = ContourTransform()
    img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(prob))
        pred_class = class_names[pred_idx]

    print(f"\nPredicted class: {pred_class}")
    print("Class probabilities:")
    for cls, p in zip(class_names, prob):
        print(f"  {cls}: {p:.4f}")


# ================== LOAD MODEL ==================
def load_model(model_path, device=DEVICE):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)

    model = get_resnet34_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ Loaded model with {num_classes} classes.")
    return model, class_names


# ================== MAIN ==================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not os.path.exists(IMAGE_PATH_NHT):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH_NHT}")
    if not os.path.exists(IMAGE_PATH_KO):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH_KO}")

    model, class_names = load_model(MODEL_PATH, DEVICE)
    predict_image(model, class_names, IMAGE_PATH_NHT, DEVICE)
    predict_image(model, class_names, IMAGE_PATH_KO, DEVICE)