"""Dice plus BCE for the legacy model's sigmoid probability output."""
import torch
from torch import nn
from torch.nn import functional as F


class DiceBCELoss(nn.Module):
    def forward(self, probability, target):
        if probability.shape != target.shape or probability.ndim != 4 or probability.shape[1] != 1:
            raise ValueError("Expected matching [B,1,H,W] probability and target")
        if not target.is_floating_point():
            raise TypeError("Segmentation targets must be floating point")
        if not torch.isfinite(probability).all() or not torch.isfinite(target).all():
            raise FloatingPointError("Non-finite probability or target")
        if torch.any((target < 0) | (target > 1)):
            raise ValueError("Segmentation targets must lie in [0,1]")
        if torch.any((probability < 0) | (probability > 1)):
            raise ValueError("The released model must output sigmoid probabilities in [0,1]")
        target = target.float()
        probability = probability.clamp(1e-7, 1 - 1e-7)
        dims = (1, 2, 3)
        intersection = (probability * target).sum(dims)
        target_sum = target.sum(dims)
        dice_score = (2 * intersection + 1e-7) / (probability.sum(dims) + target_sum + 1e-7)
        # For an empty target, penalize predicted foreground without making
        # tiny clamped probabilities look like a completely missed mask.
        dice_score = torch.where(target_sum == 0, 1 - probability.mean(dims), dice_score)
        dice_loss = 1 - dice_score.mean()
        return dice_loss + F.binary_cross_entropy(probability, target)
