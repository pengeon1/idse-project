import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice(p, g, eps=1e-5):
    """p, g: (B,C,...) probabilities / binary. Returns per-channel Dice averaged over batch -> (C,)."""
    dims = (0,) + tuple(range(2, p.ndim))
    inter = (p * g).sum(dims)
    return (2 * inter + eps) / (p.sum(dims) + g.sum(dims) + eps)


def soft_edge(x, k=3):
    """Differentiable boundary band = soft dilation - soft erosion (morphological gradient)."""
    pad = k // 2
    dil = F.max_pool3d(x, k, 1, pad)
    ero = -F.max_pool3d(-x, k, 1, pad)
    return dil - ero


class DiceBoundaryLoss(nn.Module):
    """L = w_dice * (1 - Dice) + w_bce * BCE + w_bd * (1 - BoundaryDice).

    BoundaryDice is a Dice computed on soft boundary bands of prediction and target
    (cheap, needs no distance transform). Channels = WT, TC, ET with sigmoid outputs.
    """

    def __init__(self, w_dice=1.0, w_bce=1.0, w_bd=0.5, k=3):
        super().__init__()
        self.w = (w_dice, w_bce, w_bd)
        self.k = k

    def forward(self, logits, target):
        p = torch.sigmoid(logits.float())
        g = target.float()
        dice = 1 - soft_dice(p, g).mean()
        bce = F.binary_cross_entropy_with_logits(logits.float(), g)
        bd = 1 - soft_dice(soft_edge(p, self.k), soft_edge(g, self.k)).mean()
        return self.w[0] * dice + self.w[1] * bce + self.w[2] * bd
