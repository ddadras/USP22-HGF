import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import KFold
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# === CONFIG ===
DATA_DIR = 't2_cropped_onlybig_onlyhgf'
OUTPUT_DIR = 'results_report'
NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_EPOCHS = 10
K_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === CONTOUR FILTERING TRANSFORM ===
class ContourTransform:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    def __call__(self, img):
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges = cv2.resize(edges, self.target_size)
        edges = transforms.ToTensor()(edges)
        edges = transforms.Normalize([0.5]*3, [0.5]*3)(edges)
        return edges

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

dataset = ContourImageFolder(DATA_DIR, transform=ContourTransform())
class_names = dataset.classes

# === MODEL ===
def get_resnet18_model(num_classes=NUM_CLASSES):
    model = models.resnet18(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

# === CROSS VALIDATION ===
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_metrics = []
aggregate_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
aggregate_metrics = {cls: {"precision": [], "recall": [], "f1": []} for cls in class_names}
roc_curves = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"\n=== Fold {fold+1}/{K_FOLDS} ===")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)

    model = get_resnet18_model().to(DEVICE)
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- TRAIN ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

    # --- VALIDATION ---
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            out = model(x)
            prob = torch.softmax(out, dim=1)
            _, pred = torch.max(prob, 1)
            y_true.extend(y.numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob[:, 1].cpu().numpy())

    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

    # --- METRICS ---
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fold_metrics.append({
        "Fold": fold+1, "Accuracy": acc, "Sensitivity": sensitivity, "Specificity": specificity,
        "Balanced_Acc": balanced_acc, "MCC": mcc, "Kappa": kappa, "ROC_AUC": roc_auc, "PR_AUC": pr_auc
    })

    for cls in class_names:
        aggregate_metrics[cls]["precision"].append(report[cls]["precision"])
        aggregate_metrics[cls]["recall"].append(report[cls]["recall"])
        aggregate_metrics[cls]["f1"].append(report[cls]["f1-score"])

    # --- CONFUSION MATRIX ---
    aggregate_cm += cm
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f'Confusion Matrix - Fold {fold+1}')
    cm_path = os.path.join(OUTPUT_DIR, f'confusion_fold{fold+1}.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    # --- ROC ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_curves.append((fpr, tpr, roc_auc))
    plt.plot(fpr, tpr, label=f'Fold {fold+1} (AUC={roc_auc:.3f})')

# === ROC SUMMARY ===
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves per Fold")
plt.legend()
roc_all_path = os.path.join(OUTPUT_DIR, "roc_all_folds.png")
plt.savefig(roc_all_path, bbox_inches='tight')
plt.close()

# Mean ROC
all_fpr = np.unique(np.concatenate([fpr for fpr, _, _ in roc_curves]))
mean_tpr = np.zeros_like(all_fpr)
for fpr, tpr, _ in roc_curves:
    mean_tpr += np.interp(all_fpr, fpr, tpr)
mean_tpr /= len(roc_curves)
mean_auc = auc(all_fpr, mean_tpr)
plt.plot(all_fpr, mean_tpr, color="blue", label=f"Mean ROC (AUC={mean_auc:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("Mean ROC Curve Across Folds")
mean_roc_path = os.path.join(OUTPUT_DIR, "mean_roc.png")
plt.savefig(mean_roc_path, bbox_inches='tight')
plt.close()

# === AGGREGATED CONFUSION MATRICES ===
disp = ConfusionMatrixDisplay(aggregate_cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Aggregated Confusion Matrix (Summed)')
agg_cm_path = os.path.join(OUTPUT_DIR, "aggregated_confusion.png")
plt.savefig(agg_cm_path, bbox_inches='tight')
plt.close()

norm_cm = aggregate_cm.astype(float) / aggregate_cm.sum(axis=1, keepdims=True)
disp = ConfusionMatrixDisplay(norm_cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='.2f')
plt.title('Normalized Confusion Matrix (Average per class)')
norm_cm_path = os.path.join(OUTPUT_DIR, "normalized_confusion.png")
plt.savefig(norm_cm_path, bbox_inches='tight')
plt.close()

# === EXPORT CSV RESULTS ===
fold_df = pd.DataFrame(fold_metrics)
fold_csv = os.path.join(OUTPUT_DIR, "fold_metrics.csv")
fold_df.to_csv(fold_csv, index=False)

perclass_summary = {
    cls: {
        "Precision": np.mean(aggregate_metrics[cls]["precision"]),
        "Recall": np.mean(aggregate_metrics[cls]["recall"]),
        "F1": np.mean(aggregate_metrics[cls]["f1"]),
    } for cls in class_names
}
pd.DataFrame(perclass_summary).T.to_csv(os.path.join(OUTPUT_DIR, "per_class_metrics.csv"))

print("\n=== Average Fold Metrics ===")
print(fold_df.mean(numeric_only=True))
print("\nPer-class averages saved to per_class_metrics.csv")

# === OPTIONAL: CREATE PDF REPORT ===
pdf_path = os.path.join(OUTPUT_DIR, "classification_report.pdf")
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()
story = [Paragraph("<b>Image Classification Cross-Validation Report</b>", styles['Title']), Spacer(1, 20)]

story.append(Paragraph("<b>Average Metrics Across Folds</b>", styles['Heading2']))
story.append(Paragraph(fold_df.mean(numeric_only=True).to_string(), styles['Code']))
story.append(Spacer(1, 20))

story.append(Paragraph("<b>ROC Curves</b>", styles['Heading2']))
for img in [roc_all_path, mean_roc_path]:
    story.append(RLImage(img, width=400, height=300))
    story.append(Spacer(1, 10))

story.append(Paragraph("<b>Confusion Matrices</b>", styles['Heading2']))
for img in [agg_cm_path, norm_cm_path]:
    story.append(RLImage(img, width=400, height=300))
    story.append(Spacer(1, 10))

doc.build(story)
print(f"\n✅ All metrics, CSVs, and plots saved in: {OUTPUT_DIR}")
print(f"📄 PDF summary generated: {pdf_path}")
