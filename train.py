import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from tqdm import tqdm
import os
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from collections import Counter
from torch.cuda.amp import GradScaler, autocast

# Proje içi importlar
from models.nodunet import NoduNet
from utils.losses import combined_loss

class LungSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, balance=True):
        self.balance = balance
        self.positive_samples = []
        self.negative_samples = []
        self.transform = transform
        self.samples = []

        if not os.path.exists(root_dir):
            print(f"Warning: Directory {root_dir} does not exist.")
            return

        patient_dirs = sorted(os.listdir(root_dir))
        for patient in patient_dirs:
            img_dir = os.path.join(root_dir, patient, "slices")
            msk_dir = os.path.join(root_dir, patient, "masks")
            if os.path.exists(img_dir) and os.path.exists(msk_dir):
                files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
                for f in files:
                    img_path = os.path.join(img_dir, f)
                    msk_path = os.path.join(msk_dir, f.replace("slice", "mask"))
                    if not os.path.exists(msk_path): continue
                    
                    if np.any(np.array(Image.open(msk_path)) > 0):
                        self.positive_samples.append((img_path, msk_path))
                    else:
                        self.negative_samples.append((img_path, msk_path))

        if self.balance:
            min_s = min(len(self.positive_samples), len(self.negative_samples))
            if min_s > 0:
                self.samples = self.positive_samples[:min_s] + self.negative_samples[:min_s]
            else:
                self.samples = self.positive_samples + self.negative_samples
        else:
            self.samples = self.positive_samples + self.negative_samples
        
        print(f"Dataset loaded from {root_dir}: {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, msk_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(msk_path).convert("L"))
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"].unsqueeze(0).float() / 255.0
        return image, mask

def create_weighted_sampler(dataset):
    if len(dataset) == 0: return None
    targets = [1 if np.any(np.array(Image.open(m)) > 0) else 0 for _, m in dataset.samples]
    class_counts = Counter(targets)
    if 1 not in class_counts: class_counts[1] = 1 
    if 0 not in class_counts: class_counts[0] = 1
    weights = [1.0 / class_counts[t] for t in targets]
    return WeightedRandomSampler(weights, len(weights))

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3), A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Resize(512, 512), A.Normalize(mean=[0.5], std=[0.5]), ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(512, 512), A.Normalize(mean=[0.5], std=[0.5]), ToTensorV2()
    ])

    train_dataset = LungSegmentationDataset(os.path.join(args.data_dir, "train"), train_transform, balance=True)
    val_dataset = LungSegmentationDataset(os.path.join(args.data_dir, "val"), val_transform, balance=False)
    
    if len(train_dataset) == 0:
        print("Training dataset is empty. Check data path.")
        return

    sampler = create_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = NoduNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = GradScaler()

    best_val_loss = float('inf')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = combined_loss(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with autocast():
                    out, _, _ = model(images)
                    loss = nn.functional.binary_cross_entropy_with_logits(out, masks)
                    val_loss += loss.item()
        
        avg_val = val_loss/len(val_loader)
        print(f"Val Loss: {avg_val:.4f}")
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best_model.pth"))
            print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args)