"""PatchGAN discriminator for V24.1 (Spec 101 Slice 7 / Spec 100).

A small PatchGAN that takes the 33×33 quincunx WDL prior (or a 257×257
upsampled version) and outputs an N×N patch of real/fake logits. Following
the standard 70×70 PatchGAN from pix2pix, adapted for our small input.

~250K params. Cheap to train.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class WDLDiscriminator(nn.Module):
    """PatchGAN discriminator for WDL prior quality (Spec 101 Slice 7).

    Takes a WDL prior (either the 33×33 quincunx or the 257×257 upsampled
    version) and outputs a patch map of real/fake logits.

    Architecture (following pix2pix 70×70 PatchGAN, adapted for small input):
        Conv(stride=2) → Conv(stride=2) → Conv → Conv
        LeakyReLU(0.2) activations
        No BatchNorm on the first layer (pix2pix convention)

    Args:
        in_channels: Number of input channels (1 for prior-only, 4 for
            prior + minimap context). Default 1.
        base: Base channel count (default 64, following pix2pix).
        n_layers: Number of downsampling layers (default 3).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.base = int(base)
        self.n_layers = int(n_layers)

        layers: list[nn.Module] = []
        # First layer: no BatchNorm, LeakyReLU.
        layers.extend([
            nn.Conv2d(in_channels, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        # Middle layers: BatchNorm + LeakyReLU, stride=2 for first n_layers-1.
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            stride = 2 if n < n_layers - 1 else 1
            layers.extend([
                nn.Conv2d(base * nf_mult_prev, base * nf_mult, 4,
                          stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(base * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # Final layer: 1-channel output (real/fake logits).
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.extend([
            nn.Conv2d(base * nf_mult_prev, base * nf_mult, 4,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * nf_mult, 1, 4, stride=1, padding=1),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns patch logits (B, 1, H', W') where H' < H, W' < W.

        ``x`` is (B, in_channels, H, W) — the WDL prior (real or generated).
        For the 33×33 quincunx, input is (B, 1, 33, 33).
        For the 257×257 upsampled prior, input is (B, 1, 257, 257).
        """
        return self.model(x)


def gan_step(
    model_D: WDLDiscriminator,
    model_G: nn.Module,
    real_prior: torch.Tensor,
    generator_input: torch.Tensor,
    opt_D: torch.optim.Optimizer,
    opt_G: torch.optim.Optimizer,
    lambda_adv: float = 0.1,
    l1_loss_fn=None,
    l1_targets: tuple | None = None,
    l1_weights: tuple | None = None,
) -> dict[str, float]:
    """One GAN training step: D step then G step.

    Args:
        model_D: Discriminator.
        model_G: Generator (Stage A model).
        real_prior: Real WDL prior (B, 1, H, W) for the D step.
        generator_input: Input to the generator (B, C, H, W).
        opt_D: Discriminator optimizer.
        opt_G: Generator optimizer.
        lambda_adv: Adversarial loss weight.
        l1_loss_fn: Optional L1 loss function for the G step.
        l1_targets: Optional (target_outer, target_inner) for L1 loss.
        l1_weights: Optional (weight_outer, weight_inner) for L1 loss.

    Returns:
        Dict with d_loss, g_adv_loss, and optionally g_l1_loss.
    """
    bce = nn.BCEWithLogitsLoss()
    real_label = torch.ones(real_prior.shape[0], 1, device=real_prior.device)
    fake_label = torch.zeros(real_prior.shape[0], 1, device=real_prior.device)

    # --- D step ---
    opt_D.zero_grad(set_to_none=True)
    with torch.no_grad():
        gen_outer, gen_inner = model_G(generator_input)
        # Render the generated prior as a 33×33 quincunx for the discriminator.
        gen_prior = _render_quincunx_33(gen_outer, gen_inner).unsqueeze(1)

    # Real: D should output 1.
    d_real_logits = model_D(real_prior)
    d_real_loss = bce(d_real_logits.mean(dim=[2, 3]), real_label)
    # Fake: D should output 0.
    d_fake_logits = model_D(gen_prior.detach())
    d_fake_loss = bce(d_fake_logits.mean(dim=[2, 3]), fake_label)
    d_loss = (d_real_loss + d_fake_loss) * 0.5
    d_loss.backward()
    opt_D.step()

    # --- G step ---
    opt_G.zero_grad(set_to_none=True)
    gen_outer, gen_inner = model_G(generator_input)
    gen_prior = _render_quincunx_33(gen_outer, gen_inner).unsqueeze(1)
    # G wants D to output 1 for its generated prior.
    g_adv_logits = model_D(gen_prior)
    g_adv_loss = bce(g_adv_logits.mean(dim=[2, 3]), real_label)

    g_loss = lambda_adv * g_adv_loss
    g_l1_val = 0.0
    if l1_loss_fn is not None and l1_targets is not None and l1_weights is not None:
        target_outer, target_inner = l1_targets
        weight_outer, weight_inner = l1_weights
        g_l1 = l1_loss_fn(gen_outer, gen_inner, target_outer, target_inner,
                         weight_outer, weight_inner)
        g_loss = g_loss + g_l1
        g_l1_val = g_l1.item()

    g_loss.backward()
    opt_G.step()

    return {
        "d_loss": d_loss.item(),
        "g_adv_loss": g_adv_loss.item(),
        "g_l1_loss": g_l1_val,
    }


def _render_quincunx_33(
    outer: torch.Tensor,
    inner: torch.Tensor,
) -> torch.Tensor:
    """Render the 17×17 outer + 16×16 inner into a 33×33 quincunx.

    The quincunx interleaves outer at (even, even) and inner at (odd, odd).
    """
    b = outer.shape[0]
    quincunx = torch.zeros(b, 33, 33, device=outer.device, dtype=outer.dtype)
    quincunx[:, ::2, ::2] = outer   # (B, 17, 17) → even rows/cols
    quincunx[:, 1::2, 1::2] = inner  # (B, 16, 16) → odd rows/cols
    return quincunx


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)