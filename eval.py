import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import seaborn as sns

def plot_curve(x, y, xlabel, ylabel, title, out_path):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

def plot_confusion_matrix(cm, class_names, out_path="confusion_matrix.png"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved confusion matrix to {out_path}")

def show_predictions(images, preds, labels, class_names, out_path="sample_preds.png"):
    plt.figure(figsize=(12, 6))
    for i in range(min(6, len(images))):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) +
               np.array([0.485, 0.456, 0.406]))  
        img = np.clip(img, 0, 1)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(f"Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved prediction samples to {out_path}")

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main(args):
    device = torch.device("mps") if torch.backends.mps.is_available() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")

    test_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(args.data_dir, transform=test_tfms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Loaded {len(test_dataset)} test images across {num_classes} classes")

    model = CustomCNN(num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    sample_dict = defaultdict(lambda: None)


    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            for img, pred, label in zip(imgs.cpu(), preds.cpu(), labels.cpu()):
                if sample_dict[label.item()] is None:
                    sample_dict[label.item()] = (img, pred, label)

    if not all_preds:
        print("❌ No predictions were made. Check your model and data.")
        return

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(test_dataset)
    acc = (all_preds == all_labels).mean()

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {acc*100:.2f}%")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

    sample_images = [v[0] for v in sample_dict.values()]
    sample_preds  = [v[1] for v in sample_dict.values()]
    sample_labels = [v[2] for v in sample_dict.values()]
    show_predictions(sample_images, sample_preds, sample_labels, class_names)

    if os.path.exists("history.pth"):
        history = torch.load("history.pth")
        epochs = list(range(1, len(history['train_loss']) + 1))
        plot_curve(epochs, history['train_loss'], "Epoch", "Loss", "Train Loss", "train_loss.png")
        plot_curve(epochs, history['val_loss'], "Epoch", "Loss", "Validation Loss", "val_loss.png")
        plot_curve(epochs, history['val_acc'], "Epoch", "Accuracy", "Validation Accuracy", "val_acc.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to .pth model file")
    parser.add_argument("--data-dir", required=True, help="Path to test data directory (ImageFolder)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    main(args)