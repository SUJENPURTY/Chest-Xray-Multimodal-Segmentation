"""Binary segmentation metrics on thresholded probabilities."""
import torch


def overlap_counts(probability, target, threshold=0.5):
    if probability.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    predicted = probability >= threshold
    actual = target > 0.5
    reduce = tuple(range(1, predicted.ndim))
    tp = (predicted & actual).sum(dim=reduce).float()
    fp = (predicted & ~actual).sum(dim=reduce).float()
    fn = (~predicted & actual).sum(dim=reduce).float()
    tn = (~predicted & ~actual).sum(dim=reduce).float()
    return tp, fp, fn, tn


def binary_metrics(probability, target, threshold=0.5, eps=1e-7):
    tp, fp, fn, tn = overlap_counts(probability, target, threshold)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn).clamp_min(1)
    return {"dice": dice.mean().item(), "iou": iou.mean().item(),
            "pixel_accuracy": accuracy.mean().item()}
