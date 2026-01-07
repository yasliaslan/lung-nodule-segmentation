import torch

def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (2. * intersection + 1e-8) / (union + 1e-8)

def iou_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).clamp(0, 1).sum(dim=(1, 2, 3))
    return (intersection + 1e-8) / (union + 1e-8)

def precision(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    tp = (preds * targets).sum(dim=(1,2,3))
    fp = (preds * (1 - targets)).sum(dim=(1,2,3))
    return (tp + 1e-8) / (tp + fp + 1e-8)

def recall(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    tp = (preds * targets).sum(dim=(1,2,3))
    fn = ((1 - preds) * targets).sum(dim=(1,2,3))
    return (tp + 1e-8) / (tp + fn + 1e-8)
