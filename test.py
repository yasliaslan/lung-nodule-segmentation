import os
import argparse
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from models.nodunet import NoduNet
from utils.metrics import dice_score, iou_score, precision, recall

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
                self.files.append((
                    os.path.join(s_dir, f),
                    os.path.join(m_dir, f.replace("slice", "mask"))
                ))

    def __len__(self):
        return len(self.files)

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
    
    metrics = {
        "Dice": [],
        "IoU": [],
        "Precision": [],
        "Recall": []
    }

    with torch.no_grad():
        for images, masks, _ in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            preds = torch.sigmoid(outputs)

            metrics["Dice"].append(dice_score(preds, masks).item())
            metrics["IoU"].append(iou_score(preds, masks).item())
            metrics["Precision"].append(precision(preds, masks).item())
            metrics["Recall"].append(recall(preds, masks).item())

    print("\n--- Results ---")
    for k, v in metrics.items():
        print(f"{k}: {np.mean(v):.4f}")

    df = pd.DataFrame(metrics)
    os.makedirs(args.save_dir, exist_ok=True)
    sns.boxplot(data=df).get_figure().savefig(
        os.path.join(args.save_dir, "results_boxplot.png")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
