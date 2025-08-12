import os
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models, utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import itertools

def plot_curve(xs, ys, xlabel, ylabel, title, out_path):
    plt.figure()
    plt.plot(xs, ys, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {title} → {out_path}")

def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved annotated confusion matrix → {out_path}")

def plot_roc(all_labels, all_probs, class_names, out_dir):
    n_classes = len(class_names)
    y_true = np.eye(n_classes)[all_labels]
    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:,i], all_probs[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves by Class")
    plt.legend(loc="lower right")
    out_path = os.path.join(out_dir, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved ROC curves → {out_path}")

def show_sample_predictions(model, loader, class_names, device, out_path):
    model.eval()
    samples = {}  
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            for img, true_lbl, pred_lbl in zip(imgs.cpu(), labels.cpu(), preds.cpu()):
                cls = int(true_lbl)
                if cls not in samples:
                    samples[cls] = (img, cls, int(pred_lbl))
                if len(samples) == len(class_names):
                    break
            if len(samples) == len(class_names):
                break

    mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
    std  = torch.tensor([0.229,0.224,0.225])[:,None,None]
    imgs, trues, preds = [], [], []
    for cls in range(len(class_names)):
        img, t, p = samples[cls]
        img = img * std + mean
        imgs.append(img)
        trues.append(t)
        preds.append(p)

    n = len(imgs)
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    grid = utils.make_grid(torch.stack(imgs), nrow=ncols, padding=4)
    npimg = grid.numpy().transpose((1,2,0))

    plt.figure(figsize=(ncols*3, nrows*3))
    plt.imshow(np.clip(npimg,0,1))
    plt.axis('off')

    _, height, width = grid.shape  
    img_h = imgs[0].shape[1]
    img_w = imgs[0].shape[2]
    for idx, (t, p) in enumerate(zip(trues, preds)):
        row = idx // ncols
        col = idx % ncols
        x = col * (img_w + 4) + img_w/2
        y = row * (img_h + 4) + img_h + 12
        plt.text(x, y,
                 f"T:{class_names[t]}\nP:{class_names[p]}",
                 ha='center', va='top', color='white',
                 bbox=dict(facecolor='black', alpha=0.6, pad=2))

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved sample predictions → {out_path}")

def show_8_with_1_miss_mixed(model, loader, class_names, device, out_path):
    model.eval()

    corrects_by_class = {}   
    all_corrects       = []  
    miss               = None

    mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
    std  = torch.tensor([0.229,0.224,0.225])[:,None,None]

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds   = outputs.argmax(dim=1)

            for img, true_lbl, pred_lbl in zip(imgs.cpu(), labels.cpu(), preds.cpu()):
                entry = (img, true_lbl.item(), pred_lbl.item())
                if miss is None and true_lbl != pred_lbl:
                    miss = entry
                if true_lbl == pred_lbl:
                    cls = int(true_lbl)
                    if cls not in corrects_by_class:
                        corrects_by_class[cls] = entry
                    all_corrects.append(entry)

    corrects = list(corrects_by_class.values())[:7]

    if len(corrects) < 7 and all_corrects:
        corrects.append(all_corrects[0])

    samples = corrects + ([miss] if miss is not None else [])

    imgs, trues, preds = [], [], []
    for img, t, p in samples:
        img = img * std + mean
        imgs.append(img)
        trues.append(t)
        preds.append(p)

    grid = utils.make_grid(torch.stack(imgs), nrow=4, padding=4)
    npimg = grid.numpy().transpose((1,2,0))

    plt.figure(figsize=(12,6))
    plt.imshow(np.clip(npimg,0,1))
    plt.axis('off')

    img_h, img_w = imgs[0].shape[1], imgs[0].shape[2]
    for idx, (t, p) in enumerate(zip(trues, preds)):
        row = idx // 4
        col = idx % 4
        x = col * (img_w + 4) + img_w/2
        y = row * (img_h + 4) + img_h + 10
        color = 'red' if t != p else 'white'
        plt.text(x, y,
                 f"T:{class_names[t]}\nP:{class_names[p]}",
                 ha='center', va='top',
                 color=color,
                 bbox=dict(facecolor='black', alpha=0.6, pad=2))

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved mixed 7 correct + 1 miss → {out_path}")



def main(args):
    device = torch.device('mps') if torch.backends.mps.is_available() else \
             torch.device('cuda') if torch.cuda.is_available() else \
             torch.device('cpu')
    print("Using device:", device)

    if os.path.exists("history.pth"):
        hist = torch.load("history.pth", weights_only=False)
        epochs = list(range(1, len(hist["train_loss"])+1))
        plot_curve(epochs, hist["train_loss"], "Epoch","Train Loss",  "Train Loss Curve",  "train_loss.png")
        plot_curve(epochs, hist["val_loss"],   "Epoch","Val Loss",    "Val Loss Curve",    "val_loss.png")
        plot_curve(epochs, hist["val_acc"],    "Epoch","Val Accuracy","Val Acc Curve",    "val_acc.png")
    else:
        print("No history.pth found; skipping train/val curves.")

    test_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    test_ds = datasets.ImageFolder(args.data_dir, transform=test_tfms)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    class_names = test_ds.classes
    print(f"Test set: {len(test_ds)} images, classes: {class_names}")

    model = models.densenet169(pretrained=False)
    ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(ftrs, len(class_names)))
    ckpt = torch.load(args.model_path, map_location='cpu')
    if "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model.to(device).eval()

    all_preds, all_labels, all_probs = [], [], []
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            test_loss += criterion(out, labels).item() * imgs.size(0)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = test_loss / len(test_ds)
    acc = (all_preds == all_labels).mean()
    print(f"\nTest Loss: {avg_loss:.4f}   Test Acc: {acc*100:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, "confusion_matrix.png")
    print("\nClassification Report:\n",
          classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    plot_roc(all_labels, all_probs, class_names, args.out_dir)

    show_sample_predictions(model, test_loader, class_names, device, "sample_predictions.png")

    show_8_with_1_miss_mixed(
    model=model,
    loader=test_loader,
    class_names=class_names,
    device=device,
    out_path=os.path.join(args.out_dir, "8_with_1_miss_mixed.png")
)



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="best_trashnet_mps.pth")
    p.add_argument("--data-dir",   required=True, help="root folder for test set")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir",    default=".",    help="where to save plots")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
