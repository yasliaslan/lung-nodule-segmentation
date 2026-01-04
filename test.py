import os
import argparse
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from models.nodunet import NoduNet
from utils.metrics import dice_score, iou_score, precision, recall, f1_score, specificity

class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        if not os.path.exists(root_dir): return
        patients = sorted(os.listdir(root_dir))
        for pid in patients:
            s_dir = os.path.join(root_dir, pid, "slices")
            m_dir = os.path.join(root_dir, pid, "masks")
            if not os.path.exists(s_dir): continue
            for f in sorted(os.listdir(s_dir)):
                self.files.append((os.path.join(s_dir, f), os.path.join(m_dir, f.replace("slice", "mask"))))

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img_p, msk_p = self.files[idx]
        image = transforms.ToTensor()(Image.open(img_p).convert("L")).float()
        mask = (transforms.ToTensor()(Image.open(msk_p).convert("L")) > 0.5).float()
        return image, mask, img_p

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    model = MobileNoduNet95().to(device)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Checkpoint not found at {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    test_ds = TestDataset(os.path.join(args.data_dir, "test"))
    if len(test_ds) == 0:
        print("Test dataset is empty.")
        return
        
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    metrics = {"Dice": [], "IoU": [], "Precision": [], "Recall": [], "F1": [], "Specificity": []}
    
    os.makedirs(args.save_dir, exist_ok=True)
    saved_count = 0

    with torch.no_grad():
        for images, masks, paths in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            # Sadece nodüllüleri değerlendirmek için: if masks.sum() == 0: continue

            outputs = model(images)
            if isinstance(outputs, tuple): outputs = outputs[0]
            preds = torch.sigmoid(outputs)

            metrics["Dice"].append(dice_score(preds, masks).item())
            metrics["IoU"].append(iou_score(preds, masks).item())
            metrics["Precision"].append(precision(preds, masks).item())
            metrics["Recall"].append(recall(preds, masks).item())
            metrics["F1"].append(f1_score(preds, masks).item())
            metrics["Specificity"].append(specificity(preds, masks).item())

            # Görselleştirme (sadece nodül varsa ve ilk 50 tane)
            if saved_count < 50 and masks.sum() > 0:
                pred_mask = (preds > 0.5).float().cpu().squeeze().numpy()
                true_mask = masks.cpu().squeeze().numpy()
                img = images.cpu().squeeze().numpy()
                
                fig, ax = plt.subplots(1, 3, figsize=(10, 3))
                ax[0].imshow(img, cmap='gray'); ax[0].set_title("Input")
                ax[1].imshow(true_mask, cmap='gray'); ax[1].set_title("Ground Truth")
                ax[2].imshow(pred_mask, cmap='gray'); ax[2].set_title("Prediction")
                for a in ax: a.axis('off')
                filename = os.path.basename(paths[0])
                plt.savefig(os.path.join(args.save_dir, f"pred_{filename}"))
                plt.close()
                saved_count += 1

    print("\n--- Results ---")
    for k, v in metrics.items():
        print(f"{k}: {np.mean(v):.4f}")

    df = pd.DataFrame(metrics)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.savefig(os.path.join(args.save_dir, "results_boxplot.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)