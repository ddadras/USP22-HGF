# utils.py
from torchvision import datasets, models, transforms
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import random
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    # Font
    "font.family": "Arial",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    # Lines
    "lines.linewidth": 2,
    "lines.color": "black",

    # Axes and layout
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": True,

    # Text and ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})


class ContourTransform:
    def __init__(self, target_size=(270, 270), replicate_channels=True, augment=False):
        """
        Args:
            target_size: Output image size for CNN input.
            replicate_channels: If True, gradient map is replicated to 3 channels.
            augment: If True, apply random flips, rotations, brightness/saturation adjustments.
        """
        self.target_size = target_size
        self.replicate_channels = replicate_channels
        self.augment = augment

    def __call__(self, img):
        """
        Args:
            img: PIL.Image or ndarray in RGB
        Returns:
            Tensor: normalized tensor ready for CNN
        """
        # --- Convert PIL to numpy ---
        img = np.array(img).astype(np.uint8)

        if self.augment:
            # --- Random horizontal flip ---
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
            # --- Random vertical flip ---
            if random.random() < 0.5:
                img = cv2.flip(img, 0)
            # --- Random rotation ---
            angle = random.uniform(10, 30)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            # --- Random brightness/saturation ---
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= random.uniform(0.65, 0.9)  # saturation
            hsv[..., 2] *= random.uniform(0.7, 0.9)  # brightness
            hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # --- Convert to grayscale ---
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # --- Slight blur to reduce noise ---
        blurred = cv2.GaussianBlur(gray, (3, 3), 1)

        # --- Sobel gradient magnitude ---
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # --- Normalize safely ---
        max_val = grad_mag.max()
        if max_val > 0:
            grad_mag = (grad_mag / max_val * 255).astype(np.uint8)
        else:
            grad_mag = np.zeros_like(grad_mag, dtype=np.uint8)

        # --- Morphological closing to strengthen edges ---
        kernel = np.ones((3, 3), np.uint8)
        grad_mag = cv2.morphologyEx(grad_mag, cv2.MORPH_CLOSE, kernel)

        # --- Resize to CNN input ---
        grad_mag = cv2.resize(grad_mag, self.target_size, interpolation=cv2.INTER_CUBIC)

        # --- Convert to 3 channels if needed ---
        if self.replicate_channels:
            grad_mag = cv2.cvtColor(grad_mag, cv2.COLOR_GRAY2RGB)

        # --- Convert to tensor and normalize ---
        grad_mag = transforms.ToTensor()(grad_mag)
        grad_mag = transforms.Normalize([0.5] * 3, [0.5] * 3)(grad_mag)

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


def get_resnet34_model(num_classes, dropout_rate=0.5, partial_unfreeze=True):
    # Load pretrained ResNet34
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    # Freeze all convolutional layers initially
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze the last two ResNet blocks and the classifier
    if partial_unfreeze:
        for name, param in model.named_parameters():
            if 'layer3' in name or 'layer4' in name or 'fc' in name:
                param.requires_grad = True

    # Replace the classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(256, num_classes)
    )
    return model
