import torch, torch.nn as nn, torch.nn.functional as F

class DiceCELoss3D(nn.Module):
    def __init__(self, num_classes=3, class_weights=None, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        if class_weights is not None:
            self.register_buffer("w", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.w = None

    def forward(self, logits, target):
        # logits: (B,C,D,H,W), target: (B,D,H,W) with values in {0..C-1}
        ce = F.cross_entropy(logits, target, weight=self.w)
        probs = F.softmax(logits, dim=1)

        with torch.no_grad():
            onehot = torch.zeros_like(probs).scatter_(1, target.unsqueeze(1), 1.0)

        dims = (0,2,3,4)
        intersect = (probs * onehot).sum(dim=dims)
        pred_sum  = probs.sum(dim=dims)
        targ_sum  = onehot.sum(dim=dims)

        dice_per_class = (2*intersect + self.smooth) / (pred_sum + targ_sum + self.smooth)
        dice_loss = 1 - dice_per_class.mean()
        return ce + dice_loss


def soft_skeletonize_3d(x, iters=8):
    # x: (B,1,D,H,W) in [0,1], differentiable approx using max-pool morphological thinning
    for _ in range(iters):
        eroded = 1.0 - F.max_pool3d(1.0 - x, kernel_size=3, stride=1, padding=1)
        opened = F.max_pool3d(eroded, kernel_size=3, stride=1, padding=1)
        contour = F.relu(opened - eroded)  # soft contour
        x = F.relu(x - contour)
    return x

class SoftClDiceLoss(nn.Module):
    """
    Soft clDice from Shit et al. (CVPR'21) – differentiable centerline overlap.
    Computes union-of-vessels by taking 1 - p_bg and 1 - y_bg and compares skeleta.
    """
    def __init__(self, iters=8, eps=1e-6):
        super().__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, probs, target_onehot):
        # probs: (B,C,D,H,W) softmax; target_onehot: same shape onehot
        # Build vessel union channel: 1 - background
        p_v = 1.0 - probs[:, :1, ...]  # assumes class 0 = background
        y_v = 1.0 - target_onehot[:, :1, ...]

        p_skel = soft_skeletonize_3d(p_v.clamp(0,1), iters=self.iters)
        y_skel = soft_skeletonize_3d(y_v, iters=self.iters)

        tprec = (p_skel * y_v).sum() / (p_skel.sum() + self.eps)
        tsens = (y_skel * p_v).sum() / (y_skel.sum() + self.eps)
        cldice = (2*tprec*tsens) / (tprec + tsens + self.eps)
        return 1.0 - cldice


class CompositeLoss(nn.Module):
    def __init__(self, num_classes=3, class_weights=None, soft_cldice_weight=0.0, soft_cldice_iters=8):
        super().__init__()
        self.dicece = DiceCELoss3D(num_classes, class_weights)
        self.cl = SoftClDiceLoss(iters=soft_cldice_iters)
        self.w_cl = soft_cldice_weight

    def forward(self, logits, target):
        base = self.dicece(logits, target)
        if self.w_cl <= 0:
            return base
        probs = F.softmax(logits, dim=1)
        with torch.no_grad():
            onehot = torch.zeros_like(probs).scatter_(1, target.unsqueeze(1), 1.0)
        return base + self.w_cl * self.cl(probs, onehot)
