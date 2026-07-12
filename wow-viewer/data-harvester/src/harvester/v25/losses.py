import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_texture_density(minimap: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """Compute per-pixel texture density from minimap gradient magnitude.

    Inspired by TexADiff (CVPR 2026) Relative Texture Density Map (RTDM).
    Returns a weight map in [0.5, 1.5] so flat regions still get some loss
    weight while texture-rich regions (terrain boundaries, forests, cliffs)
    are emphasized.

    Args:
        minimap: Input minimap tensor, shape (B, C, H, W).
        kernel_size: Local averaging window size for smoothing.

    Returns:
        density: Per-pixel weight map, shape (B, 1, H, W), values in [0.5, 1.5].
    """
    gray = minimap.mean(dim=1, keepdim=True)  # B, 1, H, W
    # Sobel-style finite differences
    dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    # Pad back to original spatial size
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    grad_mag = (dx ** 2 + dy ** 2).sqrt()
    # Local average pooling for spatial smoothness
    density = F.avg_pool2d(
        grad_mag, kernel_size, stride=1, padding=kernel_size // 2
    )
    # Normalize to [0.5, 1.5] — flat areas still contribute, but boundaries get 3× more weight
    max_val = density.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    density = 0.5 + density / max_val
    return density


def frequency_split_loss(
    pred: torch.Tensor, target: torch.Tensor, cutoff: float = 0.1
) -> tuple:
    """Split height loss into low-frequency (structure) and high-frequency (detail) components.

    Inspired by FRAMER (CVPR 2026) FFT frequency-decomposed training.
    LF captures overall terrain shape (WDL-prior scale), HF captures cliff
    edges and micro-terrain detail.

    Args:
        pred: Predicted heightmap, shape (B, H, W).
        target: Ground-truth heightmap, shape (B, H, W).
        cutoff: Frequency cutoff as fraction of spatial extent (0.1 = keep lowest 10% frequencies).

    Returns:
        Tuple of (lf_loss, hf_loss) as scalar tensors.
    """
    # Force full precision to bypass cuFFT power-of-two size constraints in half precision (float16/bfloat16)
    pred_f32 = pred.float()
    target_f32 = target.float()
    # ``norm="ortho"`` prevents this auxiliary loss from scaling with the
    # 257x257 grid size.  The previous unnormalised FFT made it thousands of
    # times larger than every other head and silently starved them of signal.
    pred_fft = torch.fft.rfft2(pred_f32, norm="ortho")
    target_fft = torch.fft.rfft2(target_f32, norm="ortho")

    H, W_half = pred_fft.shape[-2], pred_fft.shape[-1]
    # Build radial LF mask centred at DC
    y = torch.arange(H, device=pred.device, dtype=torch.float32)
    x = torch.arange(W_half, device=pred.device, dtype=torch.float32)
    # DC is at (0, 0) for rfft2 — wrap y around H/2
    y = torch.where(y > H / 2, y - H, y)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = cutoff * min(H, W_half * 2)
    lf_mask = ((yy ** 2 + xx ** 2) <= radius ** 2).float()

    # L1 loss in frequency domain, separated by band
    diff_fft = (pred_fft - target_fft).abs()
    lf_loss = (diff_fft * lf_mask).mean()
    hf_loss = (diff_fft * (1.0 - lf_mask)).mean()
    return lf_loss, hf_loss


class V25UnifiedLoss(nn.Module):
    """Unified loss function combining segmentation, mesh heights, object placements, and texturing.

    Enhanced with CVPR 2026 techniques:
    - TexADiff texture-density weighting for alpha loss (when minimap is provided)
    - FRAMER frequency-split height loss (LF structure + HF detail)
    """
    def __init__(self, use_freq_split: bool = True, freq_cutoff: float = 0.1,
                 lf_weight: float = 3.0, hf_weight: float = 2.0,
                 frequency_aux_weight: float = 0.25):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()
        self.use_freq_split = use_freq_split
        self.freq_cutoff = freq_cutoff
        self.lf_weight = lf_weight
        self.hf_weight = hf_weight
        self.frequency_aux_weight = frequency_aux_weight

    def forward(self, pred_outputs, target_outputs, minimap=None):
        """Calculate weighted multi-task losses.

        Args:
            pred_outputs: Dict of predicted outputs containing:
                - "mask_logits": (B, 1, 256, 256)
                - "placements": dict of class_logits, coords, rotations, exist_logits
                - "h_257": (B, 257, 257)
                - "mtex_logits": (B, vocab_size)
                - "mcly_logits": (B, 4, 16, 16, 4)
                - "alpha_256": (B, 4, 256, 256)
                - "clean_rgb": (B, 3, 256, 256) — optional TerrainInpaintHead output
                - "h_33": (B, 33, 33) — optional Stage A WDL prior prediction
            target_outputs: Dict of ground-truth targets containing:
                - "mask": (B, 1, 256, 256)
                - "placements": dict of class_ids, coords, rotations, exist
                - "h_257": (B, 257, 257)
                - "mtex_labels": (B, vocab_size)
                - "mcly_labels": (B, 16, 16, 4)
                - "alpha_256": (B, 4, 256, 256)
                - "clean_rgb": (B, 3, 256, 256) — optional clean minimap target
                - "h_33": (B, 33, 33) — optional WDL prior target
            minimap: Optional input minimap (B, 3, H, W) for texture-density
                weighting. When provided, alpha loss is modulated by per-pixel
                texture density (TexADiff).

        Returns:
            losses: Dict of specific loss components and the final "loss" tensor.
                When frequency split is enabled, also includes "height_lf" and "height_hf".
                When minimap is provided, also includes "texture_density_mean".
        """
        # 1. Footprint mask segmentation loss
        loss_mask = self.bce(pred_outputs["mask_logits"], target_outputs["mask"])

        # 1b. Optional clean terrain-shadow inpaint loss (TerrainInpaintHead)
        loss_clean = None
        if "clean_rgb" in pred_outputs and "clean_rgb" in target_outputs:
            loss_clean = self.l1(pred_outputs["clean_rgb"], target_outputs["clean_rgb"])

        # 1c. Optional Stage A WDL prior loss (V25StageAPredictor, 33x33),
        # masked by "h_33_mask" when provided (strided from the height mask).
        loss_h33 = None
        if "h_33" in pred_outputs and "h_33" in target_outputs:
            mask_33 = target_outputs.get("h_33_mask")
            if mask_33 is not None:
                denom = mask_33.sum().clamp(min=1.0)
                loss_h33 = (
                    (pred_outputs["h_33"] - target_outputs["h_33"]).abs() * mask_33
                ).sum() / denom
            else:
                loss_h33 = self.l1(pred_outputs["h_33"], target_outputs["h_33"])

        # 2. Height loss — optionally frequency-decomposed (FRAMER), optionally
        # masked by a per-vertex validity map ("height_mask", 1 = supervise).
        # Liquid areas carry the water surface, not terrain, so the trainer
        # masks them out of height supervision.
        pred_h = pred_outputs["h_257"]
        tgt_h = target_outputs["h_257"]
        mask_h = target_outputs.get("height_mask")
        if self.use_freq_split:
            if mask_h is not None:
                # Zero the residual in masked areas before the FFT: the
                # frequency loss then only sees errors on valid terrain.
                pred_h_eff = pred_h * mask_h + tgt_h * (1.0 - mask_h)
            else:
                pred_h_eff = pred_h
            lf_loss, hf_loss = frequency_split_loss(pred_h_eff, tgt_h, self.freq_cutoff)
            frequency_height = self.lf_weight * lf_loss + self.hf_weight * hf_loss
        else:
            lf_loss = hf_loss = frequency_height = None

        # The spatial L1 is the physically interpretable height term.  Keep it
        # primary; normalized FRAMER frequency bands are auxiliary structure and
        # detail guidance rather than a grid-size-dependent replacement.
        if mask_h is not None:
            denom = mask_h.sum().clamp(min=1.0)
            spatial_height = ((pred_h - tgt_h).abs() * mask_h).sum() / denom
        else:
            spatial_height = self.l1(pred_h, tgt_h)
        loss_height = spatial_height
        if frequency_height is not None:
            loss_height = loss_height + self.frequency_aux_weight * frequency_height

        # 3. Object placement coordinates, rotations, classifications, and existences
        pred_p = pred_outputs["placements"]
        tgt_p = target_outputs["placements"]

        # Mask coordinates and rotations by ground-truth existence
        exist_mask = tgt_p["exist"].view(-1, tgt_p["exist"].shape[1], 1)  # (B, max_objects, 1)
        loss_coords = self.mse(pred_p["coords"] * exist_mask, tgt_p["coords"] * exist_mask)
        loss_rotations = self.mse(pred_p["rotations"] * exist_mask, tgt_p["rotations"] * exist_mask)

        # Existence prediction loss
        loss_exist = self.bce(pred_p["exist_logits"], tgt_p["exist"].unsqueeze(-1).float())

        # Classification prediction loss
        pred_class = pred_p["class_logits"].view(-1, pred_p["class_logits"].shape[-1])
        tgt_class = tgt_p["class_ids"].view(-1).long()
        loss_class = self.ce(pred_class, tgt_class)

        # 4. Texturing losses: MTEX paths, MCLY indices, and MCAL alpha canvas
        loss_mtex = self.bce(pred_outputs["mtex_logits"], target_outputs["mtex_labels"].float())
        loss_mcly = self.ce(pred_outputs["mcly_logits"], target_outputs["mcly_labels"].long())

        # Alpha loss — optionally texture-density modulated (TexADiff)
        pred_alpha = pred_outputs["alpha_256"]
        tgt_alpha = target_outputs["alpha_256"]
        if minimap is not None:
            # Compute texture density weight map at alpha resolution
            tex_density = compute_texture_density(minimap)
            # Resize density to alpha resolution if needed
            if tex_density.shape[-2:] != pred_alpha.shape[-2:]:
                tex_density = F.interpolate(
                    tex_density, size=pred_alpha.shape[-2:], mode="bilinear", align_corners=False
                )
            # Weighted L1: higher weight at texture-rich boundaries
            loss_alpha = (torch.abs(pred_alpha - tgt_alpha) * tex_density).mean()
        else:
            tex_density = None
            loss_alpha = self.l1(pred_alpha, tgt_alpha)

        components = {
            "mask": (loss_mask, 1.0),
            "height": (loss_height, 1.0),
            "coords": (loss_coords, 2.0),
            "rotations": (loss_rotations, 1.0),
            "exist": (loss_exist, 1.0),
            "class": (loss_class, 1.0),
            "mtex": (loss_mtex, 1.0),
            "mcly": (loss_mcly, 1.0),
            "alpha": (loss_alpha, 3.0),
        }
        if loss_clean is not None:
            components["clean_rgb"] = (loss_clean, 2.0)
        if loss_h33 is not None:
            components["h_33"] = (loss_h33, 1.0)
        weighted = {name: value * weight for name, (value, weight) in components.items()}
        total_loss = sum(weighted.values())

        result = {
            "loss": total_loss,
            "mask": loss_mask,
            "height": loss_height,
            "coords": loss_coords,
            "rotations": loss_rotations,
            "exist": loss_exist,
            "class": loss_class,
            "mtex": loss_mtex,
            "mcly": loss_mcly,
            "alpha": loss_alpha,
        }
        result.update({f"weighted_{name}": value for name, value in weighted.items()})
        if loss_clean is not None:
            result["clean_rgb"] = loss_clean
        if loss_h33 is not None:
            result["h_33"] = loss_h33
        result["height_spatial"] = spatial_height
        if frequency_height is not None:
            result["height_frequency"] = frequency_height
        if lf_loss is not None:
            result["height_lf"] = lf_loss
            result["height_hf"] = hf_loss
        if tex_density is not None:
            result["texture_density_mean"] = tex_density.mean()
        return result
