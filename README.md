# ♻️ Waste Classification with Deep Learning  

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)  
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)  
![License](https://img.shields.io/badge/License-MIT-green)  
![Status](https://img.shields.io/badge/Status-Active-success)  

> Classify different types of waste into six categories using **ResNet50** and **DenseNet** on the [TrashNet Dataset](https://github.com/garythung/trashnet).  


------------------------------------------------------------
## Overview
This project leverages deep learning to automatically classify waste materials, helping improve
recycling efficiency. Models are trained on TrashNet, a dataset containing labeled waste images
across six classes:
- 🟫 Cardboard  
- 🟦 Glass  
- ⚙️ Metal  
- 📄 Paper  
- 🛍 Plastic  
- 🗑 Trash  
------------------------------------------------------------
## 📁 Project Structure
```bash
dataset/                  # Original dataset
  ├── cardboard/
  ├── glass/
  ├── metal/
  ├── paper/
  ├── plastic/
  └── trash/

dataset_split/             # Created after running split.py
  ├── train/
  ├── val/
  └── test/

train.py                   # General training script
eval.py                    # General evaluation script
train_resnet50.py          # Train ResNet50 model
eval_resnet.py             # Evaluate ResNet50 model
train_densenet.py          # Train DenseNet model
eval_densenet.py           # Evaluate DenseNet model
split.py                   # Dataset splitting utility
```
------------------------------------------------------------
## 📊 Dataset  

We use the **[TrashNet Dataset](https://github.com/garythung/trashnet)** containing categorized waste images.  

| Class       | Example         |
|-------------|-----------------|
| Cardboard   | 📦 Brown box    |
| Glass       | 🍾 Bottle       |
| Metal       | 🥫 Can          |
| Paper       | 📜 Sheet        |
| Plastic     | 🛍 Bag          |
| Trash       | 🚮 Misc waste   |
## ⚙️ Requirements  

Install the dependencies:  
```bash
pip install torch torchvision numpy pillow matplotlib scikit-learn
```

------------------------------------------------------------
## 🔄 Workflow
1️⃣ Dataset Preparation
- Place all dataset images in dataset/ within their respective folders.
- Split into train, validation, and test sets:
```bash
python split.py
```
This will create dataset_split/ with:
- Train: 70%
- Validation: 15%
- Test: 15%
2️⃣ Train the Models
- ResNet50:
```bash
python train_resnet50.py
```
- DenseNet:
```bash
python train_densenet.py
```
3️⃣ Evaluate the Models
- ResNet50:
```bash
python eval_resnet.py
```
- DenseNet:
```bash
python eval_densenet.py
```
------------------------------------------------------------
## Model Information
- ResNet50 A 50-layer residual network designed to avoid vanishing gradients.
- DenseNet A densely connected network that reuses features across layers for efficiency.
------------------------------------------------------------

## Acknowledgments
- TrashNet Dataset for providing labeled waste images.
- PyTorch for deep learning frame
