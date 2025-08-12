import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt  

device = torch.device('mps') if torch.backends.mps.is_available() else (
    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
)
print(f"Using device: {device}")

data_dir = '/Users/peteksener/Desktop/deep/trash/trash_dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir   = os.path.join(data_dir, 'val')
test_dir  = os.path.join(data_dir, 'test')

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.5)
])

test_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
val_ds   = datasets.ImageFolder(val_dir,   transform=test_tfms)
test_ds  = datasets.ImageFolder(test_dir,  transform=test_tfms)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
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
    
model_name = 'custom_cnn'
model = CustomCNN(num_classes=len(train_ds.classes)).to(device)


optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
class_weights = compute_class_weight('balanced', classes=np.unique(train_ds.targets), y=train_ds.targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

num_epochs = 30
best_val_acc = 0.0

history = {
    'train_loss': [],
    'val_loss': [],
    'val_acc': [],
    'val_f1': []
}

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
    for imgs, labels in train_pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        avg_loss = train_loss / ((train_pbar.n+1) * imgs.size(0))
        train_pbar.set_postfix(loss=f"{avg_loss:.4f}")
    
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]")
    with torch.no_grad():
        for imgs, labels in val_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            val_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            avg_vloss = val_loss / ((val_pbar.n+1) * imgs.size(0))
            val_pbar.set_postfix(loss=f"{avg_vloss:.4f}")

    val_acc  = accuracy_score(val_labels, val_preds)
    val_prec = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
    val_rec  = recall_score(val_labels, val_preds, average='weighted')
    val_f1   = f1_score(val_labels, val_preds, average='weighted')
    print(f"\nEpoch {epoch} Val Metrics: Acc {val_acc*100:.2f}%, Prec {val_prec*100:.2f}%, Rec {val_rec*100:.2f}%, F1 {val_f1*100:.2f}%")
    scheduler.step(val_acc)
    history['train_loss'].append(train_loss / len(train_loader.dataset))
    history['val_loss'].append(val_loss / len(val_loader.dataset))
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'trashnet_model3.pth')
        print(f"New best val accuracy: {best_val_acc*100:.2f}% -> checkpoint saved.")

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, history['train_loss'], label='Train Loss')
plt.plot(epochs, history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, history['val_acc'], label='Val Accuracy')
plt.plot(epochs, history['val_f1'], label='Val F1 Score')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Accuracy & F1 over Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()
print("✅ Saved training curves as training_curves.png")

print("\n== Final Test Evaluation ==")
model.load_state_dict(torch.load('trashnet_model3.pth'))
model.eval()

test_preds, test_labels = [], []
test_loss = 0.0
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Test Eval"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        test_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_acc  = accuracy_score(test_labels, test_preds)
test_prec = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
test_rec  = recall_score(test_labels, test_preds, average='weighted')
test_f1   = f1_score(test_labels, test_preds, average='weighted')
print(f"\nTest Metrics: Acc {test_acc*100:.2f}%, Prec {test_prec*100:.2f}%, Rec {test_rec*100:.2f}%, F1 {test_f1*100:.2f}%")
print("Classification Report:")
print(classification_report(test_labels, test_preds, target_names=train_ds.classes))
