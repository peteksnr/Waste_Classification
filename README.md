# Waste Classification with Deep Learning
[Python 3.x] [PyTorch 2.x] [MIT License] [Status: Active]
Classify different types of waste into six categories using ResNet50 and DenseNet on the TrashNet
Dataset.
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
## Project Structure
dataset/ # Original dataset
cardboard/
glass/
metal/
paper/
plastic/
trash/
dataset_split/ train/
val/
test/
# Created after running split.py
train.py # General training script
eval.py # General evaluation script
train_resnet50.py # Train ResNet50 model
eval_resnet.py # Evaluate ResNet50 model
Waste Classification with Deep Learning - README
train_densenet.py # Train DenseNet model
eval_densenet.py # Evaluate DenseNet model
split.py # Dataset splitting utility
------------------------------------------------------------
## Dataset
We use the TrashNet Dataset containing categorized waste images.
| Class | Example |
|-------------|-----------------|
| Cardboard | Brown box |
| Glass | Bottle |
| Metal | Can |
| Paper | Sheet |
| Plastic | Bag |
| Trash | Misc waste |
------------------------------------------------------------
## Requirements
pip install torch torchvision numpy pillow matplotlib scikit-learn
------------------------------------------------------------
## Workflow
1. Dataset Preparation
- Place all dataset images in dataset/ within their respective folders.
- Split into train, validation, and test sets:
python split.py
This will create dataset_split/ with:
- Train: 70%
- Validation: 15%
- Test: 15%
2. Train the Models
- ResNet50:
python train_resnet50.py
- DenseNet:
python train_densenet.py
3. Evaluate the Models
- ResNet50:
python eval_resnet.py
- DenseNet:
python eval_densenet.py
------------------------------------------------------------
## Model Information
- ResNet50 A 50-layer residual network designed to avoid vanishing gradients.
- DenseNet A densely connected network that reuses features across layers for efficiency.
------------------------------------------------------------
## License
This project is licensed under the MIT License.
------------------------------------------------------------
## Acknowledgments
- TrashNet Dataset for providing labeled waste images.
- PyTorch for deep learning frame
