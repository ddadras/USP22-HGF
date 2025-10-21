# train.py
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
from matplotlib.ticker import MaxNLocator
import pandas as pd
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import copy

from utils import *

# === CONFIG ===
DATA_DIR = 'data/TRAINING'
OUTPUT_DIR = 'results_report'
BATCH_SIZE = 64
NUM_EPOCHS = 50
K_FOLDS = 5
PATIENCE = 5
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = ContourImageFolder(DATA_DIR, transform=None)
train_transform = ContourTransform(augment=True)
val_transform = ContourTransform(augment=False)

class_names = dataset.classes
NUM_CLASSES = len(class_names)
print(f"Detected {NUM_CLASSES} classes: {class_names}")

# === CROSS VALIDATION ===
labels = [s[1] for s in dataset.samples]
kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_metrics = []
aggregate_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
aggregate_metrics = {cls: {"precision": [], "recall": [], "f1": []} for cls in class_names}
roc_curves = []
train_curves = []
val_curves = []
fig = plt.figure(figsize=(3.15, 3.15), dpi=600)

for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
    print(f"\n=== Fold {fold + 1}/{K_FOLDS} ===")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    # Assign transforms for each
    train_subset.dataset.custom_transform = train_transform
    val_subset.dataset.custom_transform = val_transform

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

    model = get_resnet34_model(NUM_CLASSES, dropout_rate=DROPOUT_RATE, partial_unfreeze=True).to(DEVICE)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    train_losses_per_fold = []
    val_losses_per_fold = []

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
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                total_val_loss += criterion(out, y).item()
        avg_val_loss = total_val_loss / len(val_loader)

        train_losses_per_fold.append(avg_train_loss)
        val_losses_per_fold.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # Load best model
    model.load_state_dict(best_model_wts)

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
            y_prob.extend(prob.cpu().numpy())

    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

    # Save fold model
    torch.save({
        'fold': fold + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': best_val_loss,
        'class_names': class_names
    }, os.path.join(OUTPUT_DIR, f"resnet34_fold{fold + 1}.pth"))

    # --- CLASSIFICATION REPORT ---
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    aggregate_cm += cm

    # --- METRICS ---
    acc = np.mean(y_true == y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Multi-class AUC / PR-AUC
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))

    try:
        if NUM_CLASSES == 2:
            # Use probability of "positive" class (class 1)
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
            pr_auc = average_precision_score(y_true, y_prob[:, 1])
        else:
            # Multi-class one-vs-rest
            roc_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
            pr_auc = average_precision_score(y_true_bin, y_prob, average="macro")
    except ValueError:
        roc_auc, pr_auc = np.nan, np.nan

    fold_metrics.append({
        "Fold": fold + 1,
        "Accuracy": acc,
        "Balanced_Acc": balanced_acc,
        "MCC": mcc,
        "Kappa": kappa,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    })

    for cls in class_names:
        aggregate_metrics[cls]["precision"].append(report[cls]["precision"])
        aggregate_metrics[cls]["recall"].append(report[cls]["recall"])
        aggregate_metrics[cls]["f1"].append(report[cls]["f1-score"])

    # --- ROC CURVES ---

    if NUM_CLASSES == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f'Fold {fold + 1} (AUC={roc_auc:.3f})')
        roc_curves.append((fpr, tpr, roc_auc))
    else:
        # Macro-average ROC
        fpr_dict, tpr_dict = {}, {}
        for i in range(NUM_CLASSES):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(NUM_CLASSES)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(NUM_CLASSES):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= NUM_CLASSES
        roc_auc_macro = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, label=f'Fold {fold + 1} (AUC={roc_auc_macro:.3f})')
        roc_curves.append((all_fpr, mean_tpr, roc_auc_macro))

    train_curves.append(train_losses_per_fold)
    val_curves.append(val_losses_per_fold)

# === ROC SUMMARY ===

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves per Fold")
plt.legend()
roc_all_path = os.path.join(OUTPUT_DIR, "roc_all_folds.png")
plt.savefig(roc_all_path, bbox_inches='tight')
plt.close()

# Mean ROC
fig = plt.figure(figsize=(3.15, 3.15), dpi=600)
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


# Mean learning curves
def mean(a):
    return sum(a) / len(a)


mean_train = list(map(mean, zip(*train_curves)))
mean_val = list(map(mean, zip(*val_curves)))
fg = plt.figure(layout='constrained', figsize=(6.3, 3.15), dpi=600)
ax = fg.gca()
ax.plot(range(1, len(mean_train) + 1), mean_train, label='Train Loss')
ax.plot(range(1, len(mean_val) + 1), mean_val, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title(f'Mean Learning Curve Across Folds')
ax.legend()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
lc_path = os.path.join(OUTPUT_DIR, f'mean_learning_curve.png')
plt.savefig(lc_path, bbox_inches='tight')
plt.close()

# === AGGREGATED CONFUSION MATRICES ===
disp = ConfusionMatrixDisplay(aggregate_cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d', colorbar=False)
plt.title('Aggregated Confusion Matrix (Summed)')
agg_cm_path = os.path.join(OUTPUT_DIR, "aggregated_confusion.png")
disp.ax_.set_xticks([])
disp.ax_.set_xticklabels([])
disp.ax_.set_xlabel('')
disp.ax_.set_ylabel('')
plt.savefig(agg_cm_path, bbox_inches='tight')
plt.close()

norm_cm = aggregate_cm.astype(float) / aggregate_cm.sum(axis=1, keepdims=True)
disp = ConfusionMatrixDisplay(norm_cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='.2f', colorbar=False)
plt.title('Normalized Confusion Matrix (Average per class)')
norm_cm_path = os.path.join(OUTPUT_DIR, "normalized_confusion.png")
disp.ax_.set_xticks([])
disp.ax_.set_xticklabels([])
disp.ax_.set_xlabel('')
disp.ax_.set_ylabel('')
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

# Per-class table
table_data = [["Class", "Precision", "Recall", "F1-score"]]
for cls in class_names:
    table_data.append([
        cls,
        f"{perclass_summary[cls]['Precision']:.4f}",
        f"{perclass_summary[cls]['Recall']:.4f}",
        f"{perclass_summary[cls]['F1']:.4f}"
    ])
table = Table(table_data, hAlign='LEFT')
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
]))
story.append(table)
story.append(Spacer(1, 20))

story.append(Paragraph("<b>Average Metrics Across Folds</b>", styles['Heading2']))
story.append(Paragraph(fold_df.mean(numeric_only=True).to_string(), styles['Code']))
story.append(Spacer(1, 20))

story.append(Paragraph("<b>ROC Curves</b>", styles['Heading2']))
for img in [roc_all_path, mean_roc_path]:
    story.append(RLImage(img, width=315, height=315))
    story.append(Spacer(1, 10))

story.append(Paragraph("<b>Learning Curve</b>", styles['Heading2']))
lc_path = os.path.join(OUTPUT_DIR, f'mean_learning_curve.png')
story.append(RLImage(lc_path, width=315, height=315))
story.append(Spacer(1, 10))

story.append(Paragraph("<b>Confusion Matrices</b>", styles['Heading2']))
for img in [agg_cm_path, norm_cm_path]:
    story.append(RLImage(img, width=630, height=315))
    story.append(Spacer(1, 10))

doc.build(story)
print(f"\n✅ All metrics, CSVs, and plots saved in: {OUTPUT_DIR}")
print(f"📄 PDF summary generated: {pdf_path}")
