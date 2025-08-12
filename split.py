import os
import shutil
from sklearn.model_selection import train_test_split

# Paths — adjust these
raw_dir = '/Users/peteksener/Desktop/deep/dataset-resized'   # contains subfolders: glass/, paper/, etc.
out_dir = '/Users/peteksener/Desktop/deep/untitled folder'       # will get train/, val/, test/ under it

splits = ['train', 'val', 'test']
ratios = [0.7, 0.1, 0.2]   


for split in splits:
    for cls in os.listdir(raw_dir):
        os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)

for cls in os.listdir(raw_dir):
    cls_path = os.path.join(raw_dir, cls)
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.png'))]
    train_imgs, rem_imgs = train_test_split(imgs, test_size=(1 - ratios[0]), random_state=42)
    val_frac = ratios[1] / (ratios[1] + ratios[2])
    val_imgs, test_imgs = train_test_split(rem_imgs, test_size=(1 - val_frac), random_state=42)
    
    for split, split_imgs in zip(splits, [train_imgs, val_imgs, test_imgs]):
        for img in split_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(out_dir, split, cls, img)
            shutil.copy2(src, dst)

print("Done! Data split into 70% train, 10% val, 20% test.")
