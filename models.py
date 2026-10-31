"""Coarse-to-fine 3D brain tumor segmentation models.

Stage 1  CoarseNet : light 3D U-Net run on a downsampled volume -> whole-tumor (WT) mask.
Stage 2  FineNet   : ResNet-style 3D U-Net with
                       - CBAM / attention gates on skip connections        (Zhu et al.)
                       - kernel-sharing dilated convs in deeper encoder    (Guan et al.)
                       - 1-2 Transformer layers at the bottleneck          (Zhang et al.)
Output of FineNet is 3 logit channels: WT, TC, ET (multi-label, sigmoid).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def Norm(c):
    return nn.InstanceNorm3d(c, affine=True)


def Act():
    return nn.LeakyReLU(0.01, inplace=True)


# --------------------------------------------------------------------------- blocks
class ResBlock(nn.Module):
    """Two 3x3x3 convs with a residual connection (optionally strided)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = Norm(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = Norm(cout)
        self.act = Act()
        self.skip = None
        if cin != cout or stride != 1:
            self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), Norm(cout))

    def forward(self, x):
        y = self.act(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return self.act(y + (x if self.skip is None else self.skip(x)))


class SharedDilatedConv(nn.Module):
    """Kernel-sharing dilated convolution (our reading of Guan et al.).

    ONE 3x3x3 kernel is applied at several dilation rates, so the receptive field
    grows without adding a separate weight set per rate. Branches are fused by a 1x1
    conv and added back residually.
    """

    def __init__(self, c, rates=(1, 2, 3)):
        super().__init__()
        self.rates = rates
        self.weight = nn.Parameter(torch.empty(c, c, 3, 3, 3))
        nn.init.kaiming_normal_(self.weight, a=0.01)
        self.fuse = nn.Conv3d(c * len(rates), c, 1, bias=False)
        self.norm = Norm(c)
        self.act = Act()

    def forward(self, x):
        outs = [F.conv3d(x, self.weight, padding=r, dilation=r) for r in self.rates]
        y = self.norm(self.fuse(torch.cat(outs, dim=1)))
        return self.act(x + y)


class CBAM3D(nn.Module):
    """Channel attention then spatial attention (Woo et al.), 3D version (used by Zhu et al.)."""

    def __init__(self, c, reduction=8):
        super().__init__()
        h = max(c // reduction, 4)
        self.mlp = nn.Sequential(nn.Linear(c, h), nn.ReLU(inplace=True), nn.Linear(h, c))
        self.spatial = nn.Conv3d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        b, c = x.shape[:2]
        avg = self.mlp(x.mean(dim=(2, 3, 4)))
        mx = self.mlp(x.amax(dim=(2, 3, 4)))
        x = x * torch.sigmoid(avg + mx).view(b, c, 1, 1, 1)
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(s))


class AttentionGate3D(nn.Module):
    """Additive attention gate (Attention U-Net): decoder signal g gates skip features x."""

    def __init__(self, c_skip, c_gate, c_mid):
        super().__init__()
        self.wx = nn.Conv3d(c_skip, c_mid, 1, bias=False)
        self.wg = nn.Conv3d(c_gate, c_mid, 1, bias=False)
        self.psi = nn.Conv3d(c_mid, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, g):
        a = self.act(self.wx(x) + self.wg(g))
        return x * torch.sigmoid(self.psi(a))


class SkipAttention(nn.Module):
    """Wraps either CBAM or an attention gate for a skip connection."""

    def __init__(self, kind, c_skip, c_gate):
        super().__init__()
        self.kind = kind
        if kind == "cbam":
            self.m = CBAM3D(c_skip)
        elif kind == "gate":
            self.m = AttentionGate3D(c_skip, c_gate, max(c_skip // 2, 4))
        elif kind == "none":
            self.m = None
        else:
            raise ValueError(kind)

    def forward(self, skip, gate):
        if self.kind == "cbam":
            return self.m(skip)
        if self.kind == "gate":
            return self.m(skip, gate)
        return skip


class TransformerBottleneck(nn.Module):
    """Flatten the 1/16-resolution feature map to tokens and run 1-2 Transformer layers.

    Position information comes from a depthwise 3D conv (conditional positional
    encoding), so any crop size works at inference (no fixed positional table).
    If layers=0, acts as an identity pass-through.
    """

    def __init__(self, c, layers=2, heads=8, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.layers = layers
        if layers > 0:
            self.pos = nn.Conv3d(c, c, 3, padding=1, groups=c)
            layer = nn.TransformerEncoderLayer(
                d_model=c, nhead=heads, dim_feedforward=int(c * mlp_ratio),
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(c)

    def forward(self, x):
        if self.layers == 0:
            return x
        b, c, d, h, w = x.shape
        x = x + self.pos(x)
        t = x.flatten(2).transpose(1, 2)                  # B, N, C
        t = self.norm(self.enc(t))
        return x + t.transpose(1, 2).reshape(b, c, d, h, w)


class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out, skip_attn):
        super().__init__()
        self.up = nn.ConvTranspose3d(c_in, c_out, 2, stride=2)
        self.attn = SkipAttention(skip_attn, c_skip, c_out)
        self.conv = ResBlock(c_out + c_skip, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        skip = self.attn(skip, x)
        return self.conv(torch.cat([x, skip], dim=1))


# --------------------------------------------------------------------------- stage 1
class CoarseNet(nn.Module):
    """Small 3D U-Net (3 down-steps, 8 base channels). Input is already low-res."""

    def __init__(self, in_ch=4, base=8, out_ch=1):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e0 = ResBlock(in_ch, c[0])
        self.e1 = ResBlock(c[0], c[1], stride=2)
        self.e2 = ResBlock(c[1], c[2], stride=2)
        self.e3 = ResBlock(c[2], c[3], stride=2)
        self.u2 = UpBlock(c[3], c[2], c[2], "none")
        self.u1 = UpBlock(c[2], c[1], c[1], "none")
        self.u0 = UpBlock(c[1], c[0], c[0], "none")
        self.head = nn.Conv3d(c[0], out_ch, 1)

    def forward(self, x):
        s0 = self.e0(x); s1 = self.e1(s0); s2 = self.e2(s1); b = self.e3(s2)
        y = self.u2(b, s2); y = self.u1(y, s1); y = self.u0(y, s0)
        return self.head(y)


# --------------------------------------------------------------------------- stage 2
class FineNet(nn.Module):
    """Attention-Dilated ResNet U-Net with Transformer bottleneck.

    Encoder strides: 1,2,2,2,2  -> bottleneck at 1/16. Input spatial size must be a
    multiple of 16 (use pad_to_multiple()).
    """

    def __init__(self, in_ch=4, out_ch=3, base=16, skip_attn="cbam",
                 dilated_stages=(3, 4), tf_layers=2, tf_heads=8):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([ResBlock(in_ch, c[0])] +
                                 [ResBlock(c[i - 1], c[i], stride=2) for i in range(1, 5)])
        # dilated convs only in the deeper layers
        self.dil = nn.ModuleList([SharedDilatedConv(c[i]) if i in dilated_stages else nn.Identity()
                                  for i in range(5)])
        self.bottleneck = TransformerBottleneck(c[4], layers=tf_layers, heads=tf_heads)
        self.dec = nn.ModuleList([UpBlock(c[i + 1], c[i], c[i], skip_attn) for i in range(3, -1, -1)])
        self.head = nn.Conv3d(c[0], out_ch, 1)
        # auxiliary head (deep supervision on the 1/4-res decoder output)
        self.aux = nn.Conv3d(c[2], out_ch, 1)

    def forward(self, x, return_aux=False):
        skips = []
        for e, d in zip(self.enc, self.dil):
            x = d(e(x))
            skips.append(x)
        x = self.bottleneck(skips.pop())                  # 1/16
        aux = None
        for i, up in enumerate(self.dec):
            x = up(x, skips.pop())
            if i == 1:
                aux = self.aux(x)
        out = self.head(x)
        return (out, aux) if return_aux else out


def pad_to_multiple(x, m=16):
    """Pad (B,C,D,H,W) at the end of each spatial dim; returns padded tensor and original size."""
    size = x.shape[2:]
    pads = []
    for s in reversed(size):
        pads += [0, (-s) % m]
    return F.pad(x, pads), size


def unpad(x, size):
    return x[..., :size[0], :size[1], :size[2]]


if __name__ == "__main__":
    f = FineNet()
    print("FineNet params: %.2fM" % (sum(p.numel() for p in f.parameters()) / 1e6))
    c = CoarseNet()
    print("CoarseNet params: %.2fM" % (sum(p.numel() for p in c.parameters()) / 1e6))
