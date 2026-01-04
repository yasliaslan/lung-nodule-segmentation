# Save as: utils/losses.py
import torch
import torch.nn as nn

def dice_loss_logits(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(torch.clamp(logits, -10, 10))
    intersection = (probs * targets).sum(dim=(1,2,3))
    union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    return 1 - (2. * intersection + smooth) / (union + smooth)

def combined_loss(outputs, targets):
    out, aux2, aux3 = outputs
    bce = nn.functional.binary_cross_entropy_with_logits(out, targets)
    dice = dice_loss_logits(out, targets).mean()
    aux2_loss = dice_loss_logits(aux2, targets).mean()
    aux3_loss = dice_loss_logits(aux3, targets).mean()
    return (0.5 * bce + 0.5 * dice) + 0.5 * aux2_loss + 0.3 * aux3_loss