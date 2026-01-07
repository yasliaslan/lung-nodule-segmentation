Lung Nodule Segmentation with Node-U-Net

This repository contains the official PyTorch implementation of the lung nodule segmentation framework presented in our manuscript.
The codebase is provided for reproducibility and transparency purposes and follows the same experimental setup described in the paper.

Repository Structure
lung-nodule-segmentation/
│
├── models/
│   └── nodunet.py          
│
├── utils/
│   ├── losses.py           
│   └── metrics.py          
│
├── train.py                
├── test.py                 
├── prepare_data.py         
│
├── requirements.txt
└── README.md

Model Architecture

The proposed model is a lightweight encoder–decoder segmentation network designed for lung nodule segmentation in CT slices.

Key components include:

-MobileNetV2-based encoder adapted for single-channel (grayscale) input
-Residual dilated blocks 
-Lightweight ASPP module
-Attention-based skip connections
-Deep supervision during training
-The architecture implementation in models/nodunet.py exactly matches the model described in the manuscript.

Dataset Format

The code assumes the following directory structure:
data/
└── processed/
    ├── train/
    │   └── patient_x/
    │       ├── slices/
    │       └── masks/
    └── val/
        └── patient_y/
            ├── slices/
            └── masks/

-Input images are grayscale CT slices
-Masks are binary segmentation maps
-File naming is abstracted and does not rely on dataset-specific identifiers

Dataset paths and names have been anonymized for public release.

Training

Model training is performed using train.py.
Example command:
python train.py \
  --data_dir data/processed \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4

Training details:
-Optimizer: AdamW
-Loss: Combined BCE + Dice loss with deep supervision
-Mixed-precision training (AMP)
-Learning rate scheduling based on validation loss

All training hyperparameters follow the configuration reported in the paper.

Evaluation

Model evaluation is performed using test.py.
The script computes standard segmentation metrics including:
-Dice coefficient
-IoU
-Precision
-Sensitivity
Optional qualitative predictions can be saved for visual inspection.

Reproducibility Notes

Data augmentation and validation strategies are implemented exactly as described in the manuscript.
No dataset-specific identifiers or private file paths are included.
The repository is intended to support scientific reproducibility, not dataset redistribution.

Requirements

Dependencies are listed in requirements.txt.
Main requirements:
Python ≥ 3.8
PyTorch
Albumentations
NumPy
PIL
tqdm

License

This code is released for academic and research use only.
Please cite the corresponding paper if you use this implementation in your work.
