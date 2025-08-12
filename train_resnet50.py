import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

data_dir = 'dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir   = os.path.join(data_dir, 'val')
test_dir  = os.path.join(data_dir, 'test')

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.1),
    transforms.RandomGrayscale(0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.7),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
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

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)

resnet = models.resnet50(pretrained=True)
num_features = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, len(train_ds.classes))
)
model = resnet.to(device)

optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

num_epochs = 30
best_val_acc = 0.0
train_loss_list, val_loss_list, val_acc_list = [], [], []

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
        avg_loss = train_loss / ((train_pbar.n + 1) * imgs.size(0))
        train_pbar.set_postfix(loss=f"{avg_loss:.4f}")
    scheduler.step()

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

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_trashnet_resnet50.pth')
        print(f"New best val accuracy: {best_val_acc*100:.2f}% -> checkpoint saved.")

    train_loss_list.append(train_loss / len(train_loader.dataset))
    val_loss_list.append(val_loss / len(val_loader.dataset))
    val_acc_list.append(val_acc)

plt.figure()
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png')
print("Saved: loss_curve.png")

plt.figure()
plt.plot([a * 100 for a in val_acc_list], label='Validation Accuracy (%)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('val_accuracy_curve.png')
print("Saved: val_accuracy_curve.png")

print("\n== Final Test Evaluation ==")
model.load_state_dict(torch.load('best_trashnet_resnet50.pth'))
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
