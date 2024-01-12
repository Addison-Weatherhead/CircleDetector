from data.generator import CircleParams
import torch
import numpy as np
from typing import Callable

def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """Calculate the intersection over union of two circles. 
    a and b each contain (x, y, radius) of a circle"""
    assert len(a) == len(b) == 3, "a and b must be of length 3"
    r1, r2 = a[2], b[2]
    d = np.linalg.norm(np.array([a[0], a[1]]) - np.array([b[0], b[1]]))
    if d > r1 + r2:
        # If the distance between the centers is greater than the sum of the radii, then the circles don't intersect
        return 0.0
    if d <= abs(r1 - r2):
        # If the distance between the centers is less than the absolute difference of the radii, then one circle is 
        # inside the other
        larger_r, smaller_r = max(r1, r2), min(r1, r2)
        return smaller_r ** 2 / larger_r ** 2
    r1_sq, r2_sq = r1**2, r2**2
    d1 = (r1_sq - r2_sq + d**2) / (2 * d)
    d2 = d - d1
    sector_area1 = r1_sq * np.arccos(d1 / r1)
    triangle_area1 = d1 * np.sqrt(r1_sq - d1**2)
    sector_area2 = r2_sq * np.arccos(d2 / r2)
    triangle_area2 = d2 * np.sqrt(r2_sq - d2**2)
    intersection = sector_area1 + sector_area2 - (triangle_area1 + triangle_area2)
    union = np.pi * (r1_sq + r2_sq) - intersection
    return intersection / union

def batch_iou(preds: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Calculate the average intersection over union of two batches of circle parameters"""
    assert len(preds) == len(params), "preds and params must have the same length"
    return torch.Tensor([iou(pred, param) for pred, param in zip(preds, params)]).mean()

def evaluate_testing_set(model: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader, loss_fn: Callable):
    """Evaluates the model on some testing set. 
        NOTE: Assumes the dataloader is set up where the batch size = the dataset size 
    """
    with torch.no_grad():
        test_batch = next(iter(test_dataloader))
        test_images, test_params = test_batch
        test_preds = model(test_images)
        test_loss = loss_fn(test_preds, test_params)
        test_iou = batch_iou(test_preds, test_params)
        return test_loss, test_iou