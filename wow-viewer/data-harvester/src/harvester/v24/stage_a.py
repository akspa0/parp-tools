"""FR-011: Stage A — minimap -> WDL prior correlation model.

A small U-Net (<= 1M params) on 64x64 inputs with two interpolation heads that
emit the C# reader's grid shapes (17x17 outer + 16x16 inner) via the 33x33
quincunx lattice (spec amendments A6/A7).

Input channels (13):
  0-2  cleaned minimap RGB (256 -> 64 mean pool)
  3-6  alpha_256 (256 -> 64)
  7-9  normal_xyz (257 -> 256 crop -> 64)
  10   mcnr_mask_257 (257 -> 256 crop -> 64)
  11   synthetic-WDL "cheat" channel (quincunx 33 -> 64, heights / HEIGHT_SCALE)
  12   synth presence flag (1 when channel 11 is populated)

During training the synth channel is dropped with probability 0.5 so the model
also learns the minimap-only deployment regime (User Story 3, scenario 5).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from harvester.v24 import lattice
from harvester.v24.tiles import HEIGHT_SCALE, TileRecord, downsample_mean

# Standard Stage A: 13-channel input (minimap + alpha + normal + mcnr + synth cheat)
IN_CHANNELS = 13
IN_CHANNELS_MINIMAP_ONLY = 3  # just cleaned minimap RGB
GRID = 64


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(4, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(4, cout),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StageAModel(nn.Module):
    """Small U-Net predicting a residual over the synth quincunx (RULE 7).

    When the synthetic prior is present (cheat regime) the model starts at the
    trivial baseline thanks to the zero-initialized head; when it is dropped
    (minimap-only deployment regime) the model predicts the full field.
    """

    def __init__(self, base: int = 28):
        super().__init__()
        self.enc1 = _ConvBlock(IN_CHANNELS, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.Conv2d(base * 2, base, 1)
        self.dec1 = _ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, x: torch.Tensor, synth_quincunx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (outer (B,17,17), inner (B,16,16)) in normalized height space.

        ``synth_quincunx`` is the (B, 33, 33) normalized synthetic prior with
        dropped samples zeroed; it is added to the predicted residual field.
        """
        e1 = self.enc1(x)  # (B, base, 64, 64)
        e2 = self.enc2(self.pool(e1))  # (B, 2b, 32, 32)
        e3 = self.enc3(self.pool(e2))  # (B, 4b, 16, 16)

        d2 = F.interpolate(e3, scale_factor=2, mode="nearest")
        d2 = self.dec2(torch.cat([self.up2(d2), e2], dim=1))  # (B, 2b, 32, 32)
        d1 = F.interpolate(d2, scale_factor=2, mode="nearest")
        d1 = self.dec1(torch.cat([self.up1(d1), e1], dim=1))  # (B, b, 64, 64)

        field = self.head(d1)  # (B, 1, 64, 64)
        quincunx = F.interpolate(
            field, size=(33, 33), mode="bilinear", align_corners=True
        ).squeeze(1)
        if synth_quincunx is not None:
            quincunx = quincunx + synth_quincunx
        outer = quincunx[:, ::2, ::2]
        inner = quincunx[:, 1::2, 1::2]
        return outer, inner


class StageAMinimapOnly(nn.Module):
    """Small U-Net predicting WDL prior from minimap RGB alone (3 channels).

    No synth quincunx cheat, no alpha/normal/mcnr inputs. Predicts the FULL
    WDL prior directly (no residual). Designed for standalone deployment where
    only a minimap image is available.
    """

    def __init__(self, base: int = 28):
        super().__init__()
        self.enc1 = _ConvBlock(IN_CHANNELS_MINIMAP_ONLY, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.Conv2d(base * 2, base, 1)
        self.dec1 = _ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (outer (B,17,17), inner (B,16,16)) in normalized height space.

        ``x`` is (B, 3, 64, 64) — cleaned minimap RGB, mean-pooled from 256².
        No synth quincunx — the model predicts the full prior directly.
        """
        e1 = self.enc1(x)   # (B, base, 64, 64)
        e2 = self.enc2(self.pool(e1))  # (B, 2b, 32, 32)
        e3 = self.enc3(self.pool(e2))  # (B, 4b, 16, 16)

        d2 = F.interpolate(e3, scale_factor=2, mode="nearest")
        d2 = self.dec2(torch.cat([self.up2(d2), e2], dim=1))
        d1 = F.interpolate(d2, scale_factor=2, mode="nearest")
        d1 = self.dec1(torch.cat([self.up1(d1), e1], dim=1))

        field = self.head(d1)  # (B, 1, 64, 64)
        quincunx = F.interpolate(
            field, size=(33, 33), mode="bilinear", align_corners=True
        ).squeeze(1)
        outer = quincunx[:, ::2, ::2]
        inner = quincunx[:, 1::2, 1::2]
        return outer, inner


def build_guided_input(
    cleaned_minimap: np.ndarray,
    normal: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the (9, 64, 64) input tensor from minimap + normal (+ Sobel).

    Channel layout:
      0-2: minimap RGB (mean-pooled to 64x64)
      3-5: normal XYZ (cropped to 256x256, mean-pooled to 64x64, normalized to [-1, 1])
      6-8: normal Sobel derivative (Sobel of normal Z component, mean-pooled)

    If ``normal`` is None, channels 3-8 are zeros (graceful fallback to
    the unguided model). This is a real degradation — the model won't
    have normal context — but the input shape stays the same so the
    same checkpoint can be loaded.
    """
    from scipy.ndimage import sobel  # type: ignore
    down_minimap = downsample_mean(cleaned_minimap, 4)  # (64, 64, 3)
    channels = [
        down_minimap[..., 0], down_minimap[..., 1], down_minimap[..., 2],
    ]
    if normal is not None:
        n = normal[:256, :256, :]  # crop to 64-aligned
        # If normal is not 256, resize.
        if n.shape[:2] != (256, 256):
            from PIL import Image
            n_pil = Image.fromarray(((n + 1.0) * 127.5).clip(0, 255).astype("uint8"))
            n_pil = n_pil.resize((256, 256), Image.Resampling.BILINEAR)
            n = (np.asarray(n_pil, dtype=np.float32) / 127.5) - 1.0
        # If single-channel, expand to 3-channel by replicating Z.
        if n.ndim == 2:
            n = np.stack([n, n, n], axis=-1)
        elif n.shape[-1] == 1:
            n = np.concatenate([n, n, n], axis=-1)
        # Mean-pool to 64.
        n64 = n.reshape(64, 4, 64, 4, n.shape[-1]).mean(axis=(1, 3))
        channels.extend([n64[..., 0], n64[..., 1], n64[..., 2]])
        # Sobel of the Z component (height-direction) as a curvature
        # signal. 64x64 Sobel on the 64x64 normal.
        sob = sobel(n64[..., 2], axis=0) + sobel(n64[..., 2], axis=1)
        sob = np.clip(sob / (np.max(np.abs(sob)) + 1e-6), -1.0, 1.0)
        channels.extend([sob, sob, sob])  # 3-channel Sobel, replicated
    else:
        # No normal: zero out channels 3-8 (model degrades gracefully).
        for _ in range(6):
            channels.append(np.zeros((64, 64), dtype=np.float32))
    return np.stack(channels).astype(np.float32)


class StageAMinimapOnlyGuided(nn.Module):
    """Small U-Net predicting WDL prior from minimap + normal + Sobel.

    9 input channels (vs 3 for the unguided model). Same output shape
    (outer 17x17 + inner 16x16). Slightly larger (~450K params vs 335K
    for the unguided model) but the same training shape. Adding normal
    information gives the model a much easier mapping from input to
    WDL prior: the model doesn't have to learn the normal-to-height
    relationship from scratch, it can lean on the input normals.
    """

    def __init__(self, base: int = 28):
        super().__init__()
        self.enc1 = _ConvBlock(9, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = _ConvBlock(base * 4, base * 2)
        self.up1 = nn.Conv2d(base * 2, base, 1)
        self.dec1 = _ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (outer (B,17,17), inner (B,16,16)) in normalized height space.

        ``x`` is (B, 9, 64, 64) — 3 minimap + 3 normal + 3 normal-Sobel.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = F.interpolate(e3, scale_factor=2, mode="nearest")
        d2 = self.dec2(torch.cat([self.up2(d2), e2], dim=1))
        d1 = F.interpolate(d2, scale_factor=2, mode="nearest")
        d1 = self.dec1(torch.cat([self.up1(d1), e1], dim=1))
        field = self.head(d1)
        quincunx = F.interpolate(
            field, size=(33, 33), mode="bilinear", align_corners=True
        ).squeeze(1)
        outer = quincunx[:, ::2, ::2]
        inner = quincunx[:, 1::2, 1::2]
        return outer, inner


def tta_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_aug: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Test-time augmentation: predict on n_aug flipped / rotated versions.

    Each prediction is un-flipped / un-rotated back to the original
    orientation, then averaged. The output is more robust than any
    single forward pass.

    The 5 augmentations are: identity, flip-LR, flip-UD, rot90, rot270.
    Only ``n_aug`` of these are used (1 = identity only, 5 = full TTA).
    """
    if n_aug <= 1:
        with torch.no_grad():
            return model(x)

    augs = [
        (x, lambda y: y),
        (torch.flip(x, dims=[-1]), lambda y: torch.flip(y, dims=[-1])),
        (torch.flip(x, dims=[-2]), lambda y: torch.flip(y, dims=[-2])),
        (torch.rot90(x, k=1, dims=[-2, -1]), lambda y: torch.rot90(y, k=-1, dims=[-2, -1])),
        (torch.rot90(x, k=-1, dims=[-2, -1]), lambda y: torch.rot90(y, k=1, dims=[-2, -1])),
    ][:n_aug]

    preds_o, preds_i = [], []
    with torch.no_grad():
        for x_aug, undo in augs:
            o, i = model(x_aug)
            o, i = undo(o), undo(i)
            preds_o.append(o)
            preds_i.append(i)
    outer = torch.stack(preds_o, dim=0).mean(dim=0)
    inner = torch.stack(preds_i, dim=0).mean(dim=0)
    return outer, inner


def build_minimap_only_input(cleaned_minimap: np.ndarray) -> np.ndarray:
    """Assemble the (3, 64, 64) input tensor from cleaned minimap only.

    ``cleaned_minimap`` is (256, 256, 3) float32 in [0, 1].
    Returns (3, 64, 64) float32.
    """
    down = downsample_mean(cleaned_minimap, 4)  # (64, 64, 3)
    return np.stack([down[..., 0], down[..., 1], down[..., 2]]).astype(np.float32)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_input(record: TileRecord, include_synth: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the (13, 64, 64) input tensor + the (33, 33) synth quincunx.

    The quincunx (normalized heights; zeros when ``include_synth`` is False) is
    the residual anchor the model adds its prediction to.
    """
    minimap = downsample_mean(record.cleaned_minimap, 4)  # (64, 64, 3)
    alpha = downsample_mean(record.alpha, 4)  # (64, 64, 4)
    normal = downsample_mean(record.normal[:256, :256], 4)  # (64, 64, 3)
    mcnr = downsample_mean(record.mcnr_mask[:256, :256].astype(np.float32), 4)  # (64, 64)

    channels = [
        minimap[..., 0], minimap[..., 1], minimap[..., 2],
        alpha[..., 0], alpha[..., 1], alpha[..., 2], alpha[..., 3],
        normal[..., 0], normal[..., 1], normal[..., 2],
        mcnr,
    ]

    if include_synth:
        quincunx = lattice.quincunx_33(record.synth_outer, record.synth_inner) / HEIGHT_SCALE
        synth64 = _resize_bilinear(quincunx, GRID)
        channels.append(synth64)
        channels.append(np.ones((GRID, GRID), dtype=np.float32))
    else:
        quincunx = np.zeros((33, 33), dtype=np.float32)
        channels.append(np.zeros((GRID, GRID), dtype=np.float32))
        channels.append(np.zeros((GRID, GRID), dtype=np.float32))

    return np.stack(channels).astype(np.float32), quincunx.astype(np.float32)


def object_gate_at_lattice(object_mask_257: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Object presence at the WDL lattice points (US2 scenario 4 loss gate).

    Samples the 257x257 ``object_precise_mask`` with the A1 rule — outer at
    (16r, 16c), inner at (16r+8, 16c+8) — the same slicing
    :func:`lattice.sample_lattice_from_height` uses for heights. Stage A's loss
    can then skip lattice cells that fall on object roofs, matching the minimap
    input cleaning. Returns ``(outer_gate (17,17) bool, inner_gate (16,16) bool)``.
    """
    mask = np.asarray(object_mask_257, dtype=np.float32)
    if mask.shape != (257, 257):
        raise ValueError(f"object_mask must be (257, 257); got {mask.shape}")
    outer_obj = mask[::16, ::16] > 0.5
    inner_obj = mask[8::16, 8::16] > 0.5
    return outer_obj, inner_obj


def build_target(record: TileRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Targets + per-cell loss weights.

    Weights combine three gates (per FR-012 + US2 scenario 4):
      * ``wdl_prior_confidence`` (per-cell sample weight),
      * learned-fill exclusion (``source != 2``),
      * object-pixel gate — lattice cells that fall on object roofs are
        excluded from the loss, matching the minimap input cleaning.
    """
    outer = record.prior_outer / HEIGHT_SCALE
    inner = record.prior_inner / HEIGHT_SCALE
    weight_outer = record.confidence_outer * (record.source_outer != 2)
    weight_inner = record.confidence_inner * (record.source_inner != 2)
    obj_outer, obj_inner = object_gate_at_lattice(record.object_mask)
    weight_outer = weight_outer * (~obj_outer).astype(np.float32)
    weight_inner = weight_inner * (~obj_inner).astype(np.float32)
    return (
        outer.astype(np.float32),
        inner.astype(np.float32),
        weight_outer.astype(np.float32),
        weight_inner.astype(np.float32),
    )


def weighted_l1(
    pred_outer: torch.Tensor,
    pred_inner: torch.Tensor,
    target_outer: torch.Tensor,
    target_inner: torch.Tensor,
    weight_outer: torch.Tensor,
    weight_inner: torch.Tensor,
) -> torch.Tensor:
    num = (weight_outer * (pred_outer - target_outer).abs()).sum() + (
        weight_inner * (pred_inner - target_inner).abs()
    ).sum()
    den = weight_outer.sum() + weight_inner.sum()
    return num / den.clamp_min(1e-6)


def _resize_bilinear(array: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]
    out = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=True)
    return out[0, 0].numpy()


# ---------------------------------------------------------------------------
# V24.1 — DA-V2-Small pretrained encoder + DPT head (Spec 101)
# ---------------------------------------------------------------------------

class _DAV2RefineBlock(nn.Module):
    """Small residual refine block for the DPT head."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class StageADAV2(nn.Module):
    """DA-V2-Small encoder + LoRA + DPT head for WDL prior prediction (Spec 101).

    Reuses the V23 ``DepthAnythingV2SmallEncoder`` (frozen backbone + LoRA
    adapters) and adds a lightweight DPT-style head that outputs the 33×33
    quincunx → 17×17 outer + 16×16 inner WDL prior.

    The backbone is pretrained on 62M images (DINOv2). Only LoRA adapters
    (rank 16), the patch projection, and the DPT head are trained.

    Total params: ~25M (24.8M backbone + ~200K head).
    Trainable params: ~1-2M (LoRA + patch proj + head).
    """

    def __init__(
        self,
        in_channels: int = 3,
        *,
        load_pretrained: bool = False,
        local_files_only: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        fusion_channels: int = 64,
    ):
        super().__init__()
        # Lazy import to avoid hard dependency on transformers/peft for
        # the rest of the V24 module (which only needs numpy + torch).
        from harvester.v23.encoder import DepthAnythingV2SmallEncoder

        self.in_channels = int(in_channels)
        self.encoder = DepthAnythingV2SmallEncoder(
            in_channels=self.in_channels,
            load_pretrained=load_pretrained,
            local_files_only=local_files_only,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        # Infer the neck feature schema with a dummy forward pass.
        from harvester.v23.model import infer_encoder_feature_schema

        schema_input_size = int(self.encoder.config.backbone_config.image_size)
        self._feature_schema = infer_encoder_feature_schema(
            self.encoder, input_size=schema_input_size,
        )
        neck_shapes = self._feature_schema["neck_features"]
        self._neck_channels = [int(shape[1]) for shape in neck_shapes]
        if len(self._neck_channels) != 4:
            raise ValueError(
                f"DA-V2 neck should have 4 levels; got {len(self._neck_channels)}"
            )

        # DPT head: project each neck level to fusion_channels, then
        # progressively upsample and refine to the finest resolution.
        self.projections = nn.ModuleList(
            nn.Conv2d(c, fusion_channels, kernel_size=1, bias=False)
            for c in self._neck_channels
        )
        self.refine_blocks = nn.ModuleList(
            _DAV2RefineBlock(fusion_channels) for _ in self._neck_channels
        )
        # Final head: fusion_channels → 1 channel → 33×33 quincunx.
        self.head = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (outer (B,17,17), inner (B,16,16)) in normalized height space.

        ``x`` is (B, in_channels, H, W) — minimap RGB (3ch) or guided (9ch).
        The DA-V2 encoder handles arbitrary input sizes (it patches the input);
        the head upsamples to 33×33 quincunx.
        """
        features = self.encoder(x)
        pyramid = list(features.neck_features)

        projected = [
            proj(level) for proj, level in zip(self.projections, pyramid, strict=True)
        ]
        # Progressive fusion: coarsest → finest, upsample + refine.
        fused = self.refine_blocks[0](projected[0])
        for idx in range(1, len(projected)):
            fused = F.interpolate(
                fused,
                size=projected[idx].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = self.refine_blocks[idx](fused + projected[idx])

        field = self.head(fused)  # (B, 1, H_finest, W_finest)
        # Interpolate to 33×33 quincunx.
        quincunx = F.interpolate(
            field, size=(33, 33), mode="bilinear", align_corners=True
        ).squeeze(1)  # (B, 33, 33)
        outer = quincunx[:, ::2, ::2]   # (B, 17, 17)
        inner = quincunx[:, 1::2, 1::2]  # (B, 16, 16)
        return outer, inner

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the trainable parameters (LoRA + patch proj + head)."""
        return [p for p in self.parameters() if p.requires_grad]


def build_dav2_input(
    cleaned_minimap: np.ndarray,
    normal: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the input tensor for the DA-V2 Stage A model.

    For 3-channel (minimap-only): returns (3, 256, 256) float32 in [0, 1].
    For 9-channel (guided): returns (9, 256, 256) float32 — 3 minimap + 3 normal + 3 Sobel.

    The DA-V2 encoder handles arbitrary input sizes; 256×256 is the minimap
    native resolution. The encoder's patch projection is replaced to accept
    the correct number of input channels.

    Unlike ``build_guided_input`` (which downsamples to 64×64), this function
    keeps the full 256×256 resolution because the DA-V2 encoder is designed
    for larger inputs.
    """
    minimap = cleaned_minimap  # (256, 256, 3) float32 [0, 1]
    channels = [minimap[..., 0], minimap[..., 1], minimap[..., 2]]
    if normal is not None:
        from scipy.ndimage import sobel  # type: ignore
        n = normal[:256, :256, :]  # crop to 256×256
        if n.shape[:2] != (256, 256):
            from PIL import Image
            n_pil = Image.fromarray(((n + 1.0) * 127.5).clip(0, 255).astype("uint8"))
            n_pil = n_pil.resize((256, 256), Image.Resampling.BILINEAR)
            n = (np.asarray(n_pil, dtype=np.float32) / 127.5) - 1.0
        if n.ndim == 2:
            n = np.stack([n, n, n], axis=-1)
        elif n.shape[-1] == 1:
            n = np.concatenate([n, n, n], axis=-1)
        channels.extend([n[..., 0], n[..., 1], n[..., 2]])
        # Sobel of the Z component at full 256×256 resolution.
        sob = sobel(n[..., 2], axis=0) + sobel(n[..., 2], axis=1)
        sob = np.clip(sob / (np.max(np.abs(sob)) + 1e-6), -1.0, 1.0)
        channels.extend([sob, sob, sob])
    # When normal is None, return 3 channels (minimap-only model).
    # The 9-channel guided model is only used when normal is provided.
    return np.stack(channels).astype(np.float32)


class SiLogLoss(nn.Module):
    """Scale-invariant log loss (Spec 101 Slice 2).

    The standard loss for metric depth estimation (Eigen et al. 2014, used
    by DA-V2 metric depth). Handles negative heights by shifting both pred
    and target by a constant before taking log.

    The shift breaks perfect scale invariance but is necessary for terrain
    heights which can be negative (world units range from ~-800 to +500).
    In normalized space (heights / HEIGHT_SCALE = heights / 100), the range
    is ~-8 to +5; a shift of 10.0 makes everything positive (2 to 15).

    Args:
        lambd: The λ parameter controlling the mean term (default 0.5).
        shift: Constant added to pred and target before log (default 10.0
            in normalized space = 1000 world units).
        epsilon: Small constant for numerical stability.
    """

    def __init__(self, lambd: float = 0.5, shift: float = 10.0, epsilon: float = 1e-8):
        super().__init__()
        self.lambd = float(lambd)
        self.shift = float(shift)
        self.epsilon = float(epsilon)

    def forward(
        self,
        pred_outer: torch.Tensor,
        pred_inner: torch.Tensor,
        target_outer: torch.Tensor,
        target_inner: torch.Tensor,
        weight_outer: torch.Tensor,
        weight_inner: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SiLogLoss on the weighted outer + inner cells.

        All inputs are in normalized height space (world units / HEIGHT_SCALE).
        Weights are per-cell (0 for excluded cells).
        """
        # Flatten and apply weights.
        pred = torch.cat([pred_outer.flatten(), pred_inner.flatten()])
        target = torch.cat([target_outer.flatten(), target_inner.flatten()])
        weight = torch.cat([weight_outer.flatten(), weight_inner.flatten()])

        # Only compute on weighted (non-zero) cells.
        mask = weight > 0
        if mask.sum() < 2:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_w = (pred[mask] + self.shift).clamp_min(self.epsilon)
        target_w = (target[mask] + self.shift).clamp_min(self.epsilon)

        diff_log = torch.log(target_w) - torch.log(pred_w)
        # Clamp the radicand to be non-negative BEFORE sqrt.
        # In fp16, (diff_log**2).mean() - lambd*(diff_log.mean()**2) can go
        # slightly negative due to rounding, and sqrt(negative) = NaN.
        # NaN.clamp_min(0.0) is still NaN, so the clamp must be inside.
        radicand = (diff_log ** 2).mean() - self.lambd * (diff_log.mean() ** 2)
        loss = torch.sqrt(radicand.clamp_min(self.epsilon))
        return loss


def hybrid_loss(
    pred_outer: torch.Tensor,
    pred_inner: torch.Tensor,
    target_outer: torch.Tensor,
    target_inner: torch.Tensor,
    weight_outer: torch.Tensor,
    weight_inner: torch.Tensor,
    silog_weight: float = 0.7,
    l1_weight: float = 0.3,
    silog_shift: float = 10.0,
) -> torch.Tensor:
    """Hybrid L1 + SiLogLoss (Spec 101 Slice 2).

    Combines the scale-invariant structural quality of SiLogLoss with the
    absolute-scale accuracy of L1. Default weights: 0.7 SiLogLoss + 0.3 L1.
    """
    l1 = weighted_l1(pred_outer, pred_inner, target_outer, target_inner,
                     weight_outer, weight_inner)
    silog = SiLogLoss(shift=silog_shift)(
        pred_outer, pred_inner, target_outer, target_inner,
        weight_outer, weight_inner,
    )
    return silog_weight * silog + l1_weight * l1
