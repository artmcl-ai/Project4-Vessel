import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.morphology import skeletonize, skeletonize_3d


# Soft skeletonization (for soft-clDice loss)

def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, D, H, W) or (B, C, H, W)
    """
    if x.ndim == 4:  # 2D: B,C,H,W
        p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
        return torch.min(p1, p2)
    elif x.ndim == 5:  # 3D: B,C,D,H,W
        p1 = -F.max_pool3d(-x, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        p2 = -F.max_pool3d(-x, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
        p3 = -F.max_pool3d(-x, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {x.shape}")


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    elif x.ndim == 5:
        return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {x.shape}")


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skeleton(x: torch.Tensor, iters: int) -> torch.Tensor:
    """
    Differentiable soft skeletonization from the clDice paper (Shit et al., CVPR 2021).
    x: probability map in [0,1], shape (B,1,...) or (B,C,...)
    """
    img = x
    img1 = _soft_open(img)
    skel = F.relu(img - img1)

    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)

    return skel


class SoftCLDiceLoss(nn.Module):
    """
    Soft clDice loss (1 - clDice) as in Shit et al. (CVPR 2021).
    You should usually call this on a SINGLE-CHANNEL vessel mask (e.g. A∪V vs BG).
    """

    def __init__(self, iter_: int = 3, smooth: float = 1.0):
        super().__init__()
        self.iters = iter_
        self.smooth = float(smooth)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        y_true, y_pred: (B,1,D,H,W) or (B,1,H,W), values in [0,1]
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(f"SoftCLDiceLoss: shape mismatch {y_true.shape} vs {y_pred.shape}")

        skel_pred = soft_skeleton(y_pred, self.iters)
        skel_true = soft_skeleton(y_true, self.iters)

        # Sum over spatial (and channel) dims
        dims = tuple(range(1, y_true.ndim))

        tprec = ( (skel_pred * y_true).sum(dim=dims) + self.smooth ) / (
                skel_pred.sum(dim=dims) + self.smooth
        )
        tsens = ( (skel_true * y_pred).sum(dim=dims) + self.smooth ) / (
                skel_true.sum(dim=dims) + self.smooth
        )

        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        # Loss = 1 - clDice, averaged over batch
        return 1.0 - cl_dice.mean()


# Hard clDice metric (for eval/inference)

def hard_cldice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """
    Hard clDice metric using binary skeletonization.

    pred, target: boolean numpy arrays, shape (D,H,W) or (H,W)
    returns: scalar clDice in [0,1]
    """
    if pred.shape != target.shape:
        raise ValueError(f"hard_cldice: shape mismatch {pred.shape} vs {target.shape}")

    if pred.ndim == 3:
        skel_pred = skeletonize_3d(pred)
        skel_true = skeletonize_3d(target)
    elif pred.ndim == 2:
        skel_pred = skeletonize(pred)
        skel_true = skeletonize(target)
    else:
        raise ValueError(f"hard_cldice expects 2D or 3D arrays, got {pred.ndim}D")

    def _cl_score(v: np.ndarray, s: np.ndarray) -> float:
        denom = s.sum()
        if denom == 0:
            return 0.0
        return float((v & s).sum()) / float(denom)

    tprec = _cl_score(pred, skel_true)   # skeleton(gt) inside pred
    tsens = _cl_score(target, skel_pred) # skeleton(pred) inside gt

    if tprec + tsens < eps:
        return 0.0

    return float(2.0 * tprec * tsens / (tprec + tsens + eps))
