"""Synthetic smoke tests for cfseg components.
Tests all forward passes, losses, augmentations, and spatial operations.
"""
import numpy as np
import pytest
import torch

from data import bbox_of, crop_pad, modality_dropout, to_regions
from losses import DiceBoundaryLoss, soft_dice, soft_edge
from models import CoarseNet, FineNet, pad_to_multiple, unpad


def test_finenet_shapes():
    # Forward on (1, 4, 64, 64, 64) with return_aux=True
    model = FineNet(base=8, tf_layers=1, tf_heads=4)
    model.eval()
    x = torch.randn(1, 4, 64, 64, 64)
    out, aux = model(x, return_aux=True)
    assert out.shape == (1, 3, 64, 64, 64), f"FineNet out shape {out.shape} != (1, 3, 64, 64, 64)"
    assert aux.shape == (1, 3, 16, 16, 16), f"FineNet aux shape {aux.shape} != (1, 3, 16, 16, 16)"

    # Forward on non-multiple-of-16 size via pad_to_multiple / unpad
    x_odd = torch.randn(1, 4, 65, 70, 75)
    xp, sz = pad_to_multiple(x_odd, 16)
    assert xp.shape[2] % 16 == 0 and xp.shape[3] % 16 == 0 and xp.shape[4] % 16 == 0
    out_p, aux_p = model(xp, return_aux=True)
    out = unpad(out_p, sz)
    assert out.shape == x_odd.shape[:1] + (3,) + x_odd.shape[2:], f"Unpadded out shape {out.shape} mismatch"


def test_coarsenet_shapes():
    # CoarseNet forward on non-multiple-of-8 size via pad_to_multiple
    model = CoarseNet(base=4)
    model.eval()
    x = torch.randn(1, 4, 65, 71, 77)
    xp, sz = pad_to_multiple(x, 8)
    assert xp.shape[2] % 8 == 0 and xp.shape[3] % 8 == 0 and xp.shape[4] % 8 == 0
    out_p = model(xp)
    out = unpad(out_p, sz)
    assert out.shape == (1, 1, 65, 71, 77), f"CoarseNet out shape {out.shape} mismatch"


def test_dice_boundary_loss():
    crit = DiceBoundaryLoss(w_dice=1.0, w_bce=1.0, w_bd=0.5)
    logits = torch.randn(2, 3, 32, 32, 32, requires_grad=True)
    target = (torch.rand(2, 3, 32, 32, 32) > 0.5).float()

    loss = crit(logits, target)
    assert torch.isfinite(loss), "Loss must be finite"
    loss.backward()
    assert logits.grad is not None, "Gradients should be computed"
    assert torch.isfinite(logits.grad).all(), "Gradients must be finite"

    # Loss is ~0 when logits perfectly match target
    # For large positive logits -> sigmoid(p) ~ 1.0; for large negative -> sigmoid(p) ~ 0.0
    perfect_target = torch.zeros(1, 3, 32, 32, 32)
    perfect_target[:, :, 8:24, 8:24, 8:24] = 1.0
    perfect_logits = (perfect_target * 2.0 - 1.0) * 100.0  # +100 for 1, -100 for 0

    loss_perf = crit(perfect_logits, perfect_target)
    assert loss_perf.item() < 0.05, f"Loss for perfect prediction should be close to 0, got {loss_perf.item()}"


def test_modality_dropout():
    # Never zeroes all four channels
    torch.manual_seed(42)
    for _ in range(50):
        x = torch.ones(8, 4, 16, 16, 16)
        out = modality_dropout(x, p=1.0)  # Always apply
        # Compute which channels are retained per sample
        retained = (out.abs().sum(dim=(2, 3, 4)) > 0)  # (8, 4)
        active_per_sample = retained.sum(dim=1)  # (8,)
        assert (active_per_sample >= 1).all(), "Modality dropout zeroed all channels for a sample!"


def test_crop_pad():
    arr = np.arange(4 * 40 * 40 * 40, dtype=np.float32).reshape(4, 40, 40, 40)

    # In-bounds crop
    c1 = crop_pad(arr, (10, 10, 10), (20, 20, 20))
    assert c1.shape == (4, 20, 20, 20)
    assert np.allclose(c1, arr[:, 10:30, 10:30, 10:30])

    # Out-of-bounds crop (negative start)
    c2 = crop_pad(arr, (-5, -5, -5), (20, 20, 20))
    assert c2.shape == (4, 20, 20, 20)
    # Check that padded region is 0
    assert np.allclose(c2[:, :5, :, :], 0)
    assert np.allclose(c2[:, :, :5, :], 0)
    assert np.allclose(c2[:, :, :, :5], 0)
    # Check that inside region matches
    assert np.allclose(c2[:, 5:, 5:, 5:], arr[:, :15, :15, :15])

    # Out-of-bounds crop (beyond high end)
    c3 = crop_pad(arr, (30, 30, 30), (20, 20, 20))
    assert c3.shape == (4, 20, 20, 20)
    assert np.allclose(c3[:, 10:, :, :], 0)
    assert np.allclose(c3[:, :10, :10, :10], arr[:, 30:40, 30:40, 30:40])


def test_bbox_of():
    # Empty mask
    empty_mask = np.zeros((100, 100, 100), dtype=bool)
    assert bbox_of(empty_mask) is None

    # Normal mask
    mask = np.zeros((100, 100, 100), dtype=bool)
    mask[40:60, 40:60, 40:60] = True
    lo, hi = bbox_of(mask, margin=5, min_size=0)
    assert tuple(lo) == (35, 35, 35)
    assert tuple(hi) == (65, 65, 65)

    # Minimum size logic
    small_mask = np.zeros((100, 100, 100), dtype=bool)
    small_mask[50:52, 50:52, 50:52] = True
    lo, hi = bbox_of(small_mask, margin=0, min_size=32)
    assert all((h - l) >= 32 for l, h in zip(lo, hi)), f"min_size 32 not satisfied: {hi - lo}"

    # Near volume border with min_size
    border_mask = np.zeros((100, 100, 100), dtype=bool)
    border_mask[95:99, 95:99, 95:99] = True
    lo, hi = bbox_of(border_mask, margin=0, min_size=32)
    assert all((h - l) >= 32 for l, h in zip(lo, hi)), f"min_size near high border not satisfied: {hi - lo}"
    assert all(l >= 0 and h <= 100 for l, h in zip(lo, hi))


def test_model_variants():
    # Each skip_attn option (cbam, gate, none)
    for attn in ["cbam", "gate", "none"]:
        m = FineNet(base=4, skip_attn=attn, tf_layers=1, tf_heads=4)
        m.eval()
        x = torch.randn(1, 4, 32, 32, 32)
        out = m(x)
        assert out.shape == (1, 3, 32, 32, 32), f"Failed for skip_attn={attn}"

    # tf_layers in {0, 1, 2}
    for layers in [0, 1, 2]:
        m = FineNet(base=4, skip_attn="none", tf_layers=layers, tf_heads=4)
        m.eval()
        x = torch.randn(1, 4, 32, 32, 32)
        out = m(x)
        assert out.shape == (1, 3, 32, 32, 32), f"Failed for tf_layers={layers}"

    # dilated_stages in {(3, 4), ()}
    for dil in [(3, 4), ()]:
        m = FineNet(base=4, skip_attn="none", dilated_stages=dil, tf_layers=0)
        m.eval()
        x = torch.randn(1, 4, 32, 32, 32)
        out = m(x)
        assert out.shape == (1, 3, 32, 32, 32), f"Failed for dilated_stages={dil}"
