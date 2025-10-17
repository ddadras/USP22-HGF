from torchvision import datasets, models, transforms
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2


# # === CONTOUR FILTERING TRANSFORM ===
# class ContourTransform:
#     def __init__(self, target_size=(224, 224)):
#         self.target_size = target_size
#
#     def __call__(self, img):
#         img = np.array(img)
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         edges = cv2.Canny(blurred, 10, 60)
#         edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
#         edges = cv2.resize(edges, self.target_size)
#         edges = transforms.ToTensor()(edges)
#         edges = transforms.Normalize([0.5] * 3, [0.5] * 3)(edges)
#         return edges

# class ContourTransform:
#     def __init__(self, target_size=(270, 270), sigma=2):
#         self.target_size = target_size
#         self.sigma = sigma
#
#     def __call__(self, img):
#         img = np.array(img)
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#         # --- adaptive thresholds based on image median ---
#         v = np.median(blurred)
#         lower = int(max(10, (1.0 - self.sigma) * v))
#         upper = int(min(100, (1.0 + self.sigma) * v))
#         edges = cv2.Canny(blurred, lower, upper)
#
#         # optional: keep faint structure by blending with grayscale
#         alpha = 0.0  # how much of original gray to mix back
#         blended = cv2.addWeighted(gray, alpha, edges, 1 - alpha, 0)
#         blended = cv2.cvtColor(blended, cv2.COLOR_GRAY2RGB)
#
#         blended = cv2.resize(blended, self.target_size)
#         blended = transforms.ToTensor()(blended)
#         blended = transforms.Normalize([0.5]*3, [0.5]*3)(blended)
#         return blended

class ContourTransform:
    def __init__(self, target_size=(270, 270), replicate_channels=True):
        """
        Args:
            target_size: Output image size for CNN input.
            replicate_channels: If True, gradient map is replicated to 3 channels.
        """
        self.target_size = target_size
        self.replicate_channels = replicate_channels

    def __call__(self, img):
        """
        Args:
            img: PIL.Image or ndarray in RGB
        Returns:
            Tensor: normalized tensor ready for CNN
        """
        # --- Convert to numpy array ---
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # --- Slight blur to reduce noise ---
        blurred = cv2.GaussianBlur(gray, (3, 3), 1)

        # --- Compute Sobel gradient magnitude ---
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # --- Normalize safely ---
        max_val = grad_mag.max()
        if max_val > 0:
            grad_mag = (grad_mag / max_val * 255).astype(np.uint8)
        else:
            grad_mag = np.zeros_like(grad_mag, dtype=np.uint8)

        # --- Optional morphological closing to strengthen edges ---
        kernel = np.ones((3,3), np.uint8)
        grad_mag = cv2.morphologyEx(grad_mag, cv2.MORPH_CLOSE, kernel)

        # --- Resize to CNN input ---
        grad_mag = cv2.resize(grad_mag, self.target_size, interpolation=cv2.INTER_CUBIC)

        # --- Convert to 3-channel if needed ---
        if self.replicate_channels:
            grad_mag = cv2.cvtColor(grad_mag, cv2.COLOR_GRAY2RGB)

        # --- Convert to tensor and normalize ---
        grad_mag = transforms.ToTensor()(grad_mag)
        grad_mag = transforms.Normalize([0.5]*3, [0.5]*3)(grad_mag)

        return grad_mag

# === DATASET ===
class ContourImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.custom_transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.custom_transform:
            image = self.custom_transform(image)
        return image, target


# ================= MODEL FUNCTION =================
def get_resnet18_model(num_classes, dropout_rate=0.5):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(256, num_classes)
    )
    return model


def get_resnet34_model(num_classes, dropout_rate=0.5):
    # Load pretrained ResNet34
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # Freeze all convolutional layers initially
    for p in model.parameters():
        p.requires_grad = False

    # Replace the classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(256, num_classes)
    )
    return model