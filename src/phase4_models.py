"""
phase4_models.py  (Phase 4)
---------------------------
The four networks:
  G : A(mixed) -> B(clean)      generator we actually want (the clutter remover)
  F : B(clean) -> A(mixed)      reverse generator (enables cycle-consistency)
  D_A, D_B : PatchGAN discriminators judging real vs fake in each domain.

Generators are ResNet-based (the standard CycleGAN generator of Zhu et al.). We use
InstanceNorm and reflection padding. Kept moderately sized (ngf=32, 6 residual
blocks) so ~50k iterations are feasible on an Apple-Silicon GPU (MPS).
"""
import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim))

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, ngf=32, n_blocks=6):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7),
                  nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        # downsample x2
        c = ngf
        for _ in range(2):
            layers += [nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(c * 2), nn.ReLU(True)]
            c *= 2
        # residual blocks
        for _ in range(n_blocks):
            layers += [ResnetBlock(c)]
        # upsample x2
        for _ in range(2):
            layers += [nn.ConvTranspose2d(c, c // 2, 3, stride=2, padding=1,
                                          output_padding=1),
                       nn.InstanceNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def asym_channels(x, m=8):
    """Left-right symmetry-breaking channels. x [B,C,256,F] with 256 = 16x16 antenna
    pairs -> |orientation| S - P S P where P(k)=(m-k)%16. Differentiable. This pre-
    exposes a one-sided bleed by cancelling the symmetric clutter (the Phase-2 idea,
    now fed straight to the generator)."""
    B, C, P, F = x.shape
    g = x.reshape(B, C, 16, 16, F)
    idx = [(m - k) % 16 for k in range(16)]
    gm = g[:, :, idx][:, :, :, idx]              # mirror both antenna indices
    return (g - gm).reshape(B, C, 256, F)


class SymGenerator(nn.Module):
    """Wraps a base generator. If use_asym, it concatenates the symmetry-asymmetry
    channels to the input (so a 2-ch input becomes 4-ch) before the base network.
    Keeps the training loop identical -- G(x) just works."""
    def __init__(self, base, use_asym=False, m=8):
        super().__init__()
        self.base = base; self.use_asym = use_asym; self.m = m

    def forward(self, x):
        if self.use_asym:
            x = torch.cat([x, asym_channels(x, self.m)], dim=1)
        return self.base(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN: outputs a map of real/fake scores over overlapping patches.
    Spectral normalization (Miyato et al., also used by the paper) bounds each
    conv layer's Lipschitz constant, which stops the discriminator from becoming
    'too strong' and blowing the adversarial game up to NaN."""
    def __init__(self, in_ch=2, ndf=64, n_layers=3, spectral=True):
        super().__init__()
        sn = nn.utils.spectral_norm if spectral else (lambda m: m)
        layers = [sn(nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1)),
                  nn.LeakyReLU(0.2, True)]
        c = ndf
        for n in range(1, n_layers):
            nc = min(c * 2, ndf * 8)
            layers += [sn(nn.Conv2d(c, nc, 4, stride=2, padding=1)),
                       nn.InstanceNorm2d(nc), nn.LeakyReLU(0.2, True)]
            c = nc
        nc = min(c * 2, ndf * 8)
        layers += [sn(nn.Conv2d(c, nc, 4, stride=1, padding=1)),
                   nn.InstanceNorm2d(nc), nn.LeakyReLU(0.2, True)]
        layers += [sn(nn.Conv2d(nc, 1, 4, stride=1, padding=1))]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(net, gain=0.02):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # spectral_norm renames the trainable weight to 'weight_orig'
            w = getattr(m, "weight_orig", None)
            w = w if w is not None else m.weight
            with torch.no_grad():
                nn.init.normal_(w, 0.0, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return net


class ImagePool:
    """Keep a history of generated images to stabilise the discriminator (Shrivastava et al.)."""
    def __init__(self, size=50):
        self.size = size; self.items = []

    def query(self, images):
        if self.size == 0:
            return images
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.items) < self.size:
                self.items.append(img); out.append(img)
            else:
                if torch.rand(1).item() < 0.5:
                    i = torch.randint(0, self.size, (1,)).item()
                    out.append(self.items[i].clone()); self.items[i] = img
                else:
                    out.append(img)
        return torch.cat(out, 0)
