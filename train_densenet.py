import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models, utils
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    roc_curve, auc
)
from sklearn.manifold import TSNE
from tqdm import tqdm
import itertools

def train_and_save(data_dir, batch_size, num_epochs, device):
    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'val')
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(0.5,0.5,0.5,0.1),
        transforms.RandomGrayscale(0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.7),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, num_workers=4)

    model = models.densenet169(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier.in_features, len(train_ds.classes))
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {"train_loss":[], "val_loss":[], "val_acc":[]}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs+1):
        model.train(); running_loss=0.0
        for imgs,labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item()*imgs.size(0)
        avg_train_loss = running_loss/len(train_ds)

        model.eval(); rloss=0.0; all_preds, all_labels = [],[]
        with torch.no_grad():
            for imgs,labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                rloss += loss.item()*imgs.size(0)
                preds = out.argmax(1)
                all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
        avg_val_loss = rloss/len(val_ds)
        val_acc = np.mean(np.array(all_preds)==np.array(all_labels))

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {val_acc*100:.2f}%")

        if val_acc>best_val_acc:
            best_val_acc=val_acc
            torch.save(model.state_dict(), 'best_trashnet_mps.pth')
            print(f"→ New best val acc {best_val_acc*100:.2f}% (checkpoint saved)")
        scheduler.step()

    torch.save(history, 'history.pth')
    print("Training complete. history.pth and best_trashnet_mps.pth saved.")
    return history

def plot_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"] )+1))
    plt.figure(); plt.plot(epochs, history["train_loss"], marker='o', label="Train Loss")
    plt.plot(epochs, history["val_loss"],   marker='o', label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train vs Val Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png")); plt.close()
    plt.figure(); plt.plot(epochs, [a*100 for a in history["val_acc"]], marker='o', label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Validation Accuracy")
    plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "val_accuracy.png")); plt.close()
    print(f"→ Saved training curves in {out_dir}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",   required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--out-dir",    default="viz_results")
    args=p.parse_args()
    device=torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print("Using device:", device)


    history = train_and_save(args.data_dir, args.batch_size, args.epochs, device)

    os.makedirs(args.out_dir, exist_ok=True)

    plot_history(history, args.out_dir)

if __name__=="__main__":
    main()