"""Minimap Albedo De-Mixing & Dynamic MCAL Layer Decipherer (Spec 268 Phase 2).

De-mixes composite minimap RGB colors into discrete authentic tileset textures (MTEX)
and multi-layer alpha splats (MCLY/MCAL), implementing a dynamic layer assignment
algorithm to optimize layer stacks (<= 4 layers per MCNK chunk) across continuous quilt canvases (AC-002).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TilesetTexture:
    texture_id: int
    name: str
    color_rgb: Tuple[float, float, float]  # Normalized RGB in [0.0, 1.0]

    @property
    def rgb_array(self) -> np.ndarray:
        return np.array(self.color_rgb, dtype=np.float32)


# Built-in archetypal development and temperate world tileset palette
DEFAULT_TILESET_PALETTE: List[TilesetTexture] = [
    TilesetTexture(0, "Tileset/Base/GreenGrass.blp", (0.34, 0.44, 0.22)),
    TilesetTexture(1, "Tileset/Base/DarkDirt.blp", (0.38, 0.30, 0.22)),
    TilesetTexture(2, "Tileset/Base/MountainRock.blp", (0.48, 0.48, 0.48)),
    TilesetTexture(3, "Tileset/Base/CobblestoneRoad.blp", (0.55, 0.52, 0.46)),
    TilesetTexture(4, "Tileset/Base/DryGrass.blp", (0.50, 0.48, 0.28)),
    TilesetTexture(5, "Tileset/Base/ForestMulch.blp", (0.28, 0.24, 0.18)),
    TilesetTexture(6, "Tileset/Base/HighlandSnow.blp", (0.85, 0.88, 0.90)),
    TilesetTexture(7, "Tileset/Base/SandyGround.blp", (0.68, 0.60, 0.45)),
]


@dataclass
class DecipheredChunkLayers:
    chunk_x: int
    chunk_y: int
    active_texture_ids: List[int]  # Length <= 4
    mcal_alpha_layers: List[np.ndarray]  # Up to 3 alpha layers (for L1..L3), each (64, 64) uint8
    coverage_fractions: List[float]


class McalLayerDecipherer:
    """De-mixes minimap imagery into authentic tileset texture layers and implements

    dynamic layer stacking constraints across ADT chunks.
    """

    def __init__(self, palette: Optional[Sequence[TilesetTexture]] = None) -> None:
        self.palette = list(palette or DEFAULT_TILESET_PALETTE)
        self._palette_matrix = np.stack([tex.rgb_array for tex in self.palette], axis=1)  # (3, K)

    def demix_pixel_albedo(
        self,
        minimap_rgb_256: np.ndarray,
        top_k: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose minimap RGB into per-pixel texture weights and bare illumination.

        Args:
            minimap_rgb_256: (256, 256, 3) float32 in [0, 1] or uint8 in [0, 255].
            top_k: Maximum number of active textures considered globally.

        Returns:
            texture_weights: (256, 256, K) float32 array with sum_k weights = 1.0.
            reconstructed_albedo: (256, 256, 3) float32 composite 2D texture albedo.
            bare_illumination: (256, 256) float32 estimated surface shading scalar.
        """
        img = np.asarray(minimap_rgb_256, dtype=np.float32)
        if img.max() > 1.5:
            img = img / 255.0

        h, w = img.shape[:2]
        k = len(self.palette)

        # 1. Estimate luminance / bare illumination
        # Illumination is roughly the maximum channel intensity or weighted luminance
        lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        bare_illum = np.maximum(lum, 1e-4)

        # Normalized chromaticity: P / bare_illum
        chroma = img / bare_illum[..., None]
        chroma_flat = chroma.reshape(-1, 3)  # (N, 3)

        # 2. Compute similarity against each palette entry in both chromaticity and intensity
        # For each palette texture k:
        pal_mats = self._palette_matrix  # (3, K)
        pal_norms = np.linalg.norm(pal_mats, axis=0)  # (K,)
        pal_unit = pal_mats / np.maximum(pal_norms, 1e-6)  # (3, K)

        img_flat = img.reshape(-1, 3)  # (N, 3)
        img_norms = np.linalg.norm(img_flat, axis=-1, keepdims=True)  # (N, 1)
        img_unit = img_flat / np.maximum(img_norms, 1e-6)  # (N, 3)

        # Directional cosine similarity in RGB color space
        cos_sim = np.dot(img_unit, pal_unit)  # (N, K)

        # Intensity/reflectance similarity: compare norm ratios
        # Best-fit scale factor s = dot(img, T_k) / norm(T_k)^2
        dot_prods = np.dot(img_flat, pal_mats)  # (N, K)
        scales = dot_prods / np.maximum(pal_norms ** 2, 1e-6)  # (N, K)

        # Reconstruction error: ||img - s * T_k||^2
        recon_errors = np.zeros_like(cos_sim)
        for k_idx in range(k):
            pred_k = scales[:, k_idx : k_idx + 1] * pal_mats[:, k_idx]
            recon_errors[:, k_idx] = np.sum((img_flat - pred_k) ** 2, axis=-1)

        # Softmax / exponential inverse-distance weighting with sharpness temperature
        temperature = 0.005
        weights_raw = np.exp(-recon_errors / temperature)
        weights_raw *= np.maximum(cos_sim, 0.0) ** 4

        sums = np.sum(weights_raw, axis=-1, keepdims=True)
        weights_norm = weights_raw / np.maximum(sums, 1e-6)

        weights_reshaped = weights_norm.reshape(h, w, k)

        # Global texture usage ranking
        mean_usage = np.mean(weights_reshaped, axis=(0, 1))
        top_indices = np.argsort(mean_usage)[::-1][:top_k]

        mask = np.zeros(k, dtype=bool)
        mask[top_indices] = True
        weights_reshaped[..., ~mask] = 0.0
        final_sums = np.sum(weights_reshaped, axis=-1, keepdims=True)
        weights_final = weights_reshaped / np.maximum(final_sums, 1e-6)

        # Reconstructed albedo: sum_k w_k * T_k
        recon_albedo = np.dot(weights_final, self._palette_matrix.T)  # (H, W, 3)

        # True illumination scalar: ||img|| / ||recon_albedo||
        albedo_mag = np.linalg.norm(recon_albedo, axis=-1)
        img_mag = np.linalg.norm(img, axis=-1)
        refined_illum = np.clip(img_mag / np.maximum(albedo_mag, 1e-4), 0.0, 2.5)

        return weights_final, recon_albedo, refined_illum

    def build_chunk_layers(
        self,
        texture_weights_256: np.ndarray,
    ) -> List[DecipheredChunkLayers]:
        """Convert continuous 256x256 texture weights into 16x16 ADT chunks conforming to editor rules:

        - Maximum 4 active layers per chunk (L0 base + L1..L3 alpha splats).
        - Dynamic layer stacking optimizes adjacent chunk layer consistency.
        - Generates 64x64 uint8 alpha maps for L1, L2, L3.
        """
        h, w, k = texture_weights_256.shape
        chunk_pixels = 16  # 256 / 16 = 16 pixels per chunk in minimap

        # Global base layer preference: texture with highest overall footprint is preferred L0
        global_usage = np.sum(texture_weights_256, axis=(0, 1))
        global_base_tex = int(np.argmax(global_usage))

        chunk_results: List[DecipheredChunkLayers] = []

        # Previous chunk layer assignments for spatial continuity across row/col scans
        layer_history: Dict[Tuple[int, int], List[int]] = {}

        for cy in range(16):
            for cx in range(16):
                u0, u1 = cx * chunk_pixels, (cx + 1) * chunk_pixels
                v0, v1 = cy * chunk_pixels, (cy + 1) * chunk_pixels

                chunk_crop = texture_weights_256[v0:v1, u0:u1, :]  # (16, 16, K)
                chunk_means = np.mean(chunk_crop, axis=(0, 1))  # (K,)

                # Pick candidates present in chunk (threshold > 1.5% coverage)
                present_texs = [i for i in range(k) if chunk_means[i] > 0.015]
                if not present_texs:
                    present_texs = [global_base_tex]

                # Sort by chunk presence
                present_texs.sort(key=lambda idx: chunk_means[idx], reverse=True)

                # Clamp to at most 4 layers
                selected_texs = present_texs[:4]

                # Dynamic Layer Stacking Optimization:
                # If western or northern neighbor chunk has an established layer order,
                # align shared texture slots to reduce layer flipping.
                neighbors = []
                if cx > 0 and (cx - 1, cy) in layer_history:
                    neighbors.append(layer_history[(cx - 1, cy)])
                if cy > 0 and (cx, cy - 1) in layer_history:
                    neighbors.append(layer_history[(cx, cy - 1)])

                if neighbors:
                    # Find reference layer ordering from primary neighbor
                    ref_order = neighbors[0]
                    aligned_order: List[int] = []
                    # Keep shared textures in their existing slot if possible
                    for t in ref_order:
                        if t in selected_texs and t not in aligned_order:
                            aligned_order.append(t)
                    # Append remaining new textures
                    for t in selected_texs:
                        if t not in aligned_order:
                            aligned_order.append(t)
                    selected_texs = aligned_order[:4]

                layer_history[(cx, cy)] = selected_texs

                # Extract alpha splats: upsample 16x16 chunk crop to 64x64 MCAL resolution
                # L0 is base (implied 1.0 where upper layers are 0).
                # L1..L3 store explicit alpha in [0, 255].
                active_weights = chunk_crop[..., selected_texs]  # (16, 16, num_layers)
                w_sums = np.sum(active_weights, axis=-1, keepdims=True)
                active_norm = active_weights / np.maximum(w_sums, 1e-6)

                # Bilinear zoom to 64x64
                from scipy import ndimage
                alpha_layers_64: List[np.ndarray] = []

                # Layers 1 to 3
                for l_idx in range(1, len(selected_texs)):
                    raw_layer_16 = active_norm[..., l_idx]
                    zoomed = ndimage.zoom(raw_layer_16, 4.0, order=1)[:64, :64]
                    alpha_uint8 = np.clip(zoomed * 255.0, 0, 255).astype(np.uint8)
                    alpha_layers_64.append(alpha_uint8)

                fractions = [float(chunk_means[t]) for t in selected_texs]

                chunk_results.append(
                    DecipheredChunkLayers(
                        chunk_x=cx,
                        chunk_y=cy,
                        active_texture_ids=selected_texs,
                        mcal_alpha_layers=alpha_layers_64,
                        coverage_fractions=fractions,
                    )
                )

        return chunk_results

    def generate_mtex_manifest(
        self,
        chunk_layers: List[DecipheredChunkLayers],
    ) -> List[str]:
        """Generate the unique MTEX texture path manifest for an ADT tile."""
        unique_ids = set()
        for cl in chunk_layers:
            unique_ids.update(cl.active_texture_ids)

        sorted_ids = sorted(unique_ids)
        return [self.palette[tid].name for tid in sorted_ids if tid < len(self.palette)]

    def match_color_to_palette(self, color_rgb: np.ndarray) -> Tuple[int, str, float]:
        """Find the closest palette texture ID and name for a given RGB vector in [0, 1]."""
        c = np.asarray(color_rgb, dtype=np.float32).flatten()[:3]
        if c.max() > 1.5:
            c = c / 255.0
        c_norm = np.linalg.norm(c)
        c_unit = c / max(c_norm, 1e-6)

        best_id = 0
        best_score = -1e9
        best_dist = 1e9

        for tex in self.palette:
            t = tex.rgb_array
            t_norm = np.linalg.norm(t)
            t_unit = t / max(t_norm, 1e-6)
            cos_sim = float(np.dot(c_unit, t_unit))
            dist = float(np.linalg.norm(c - t))
            score = cos_sim - 0.5 * dist
            if score > best_score:
                best_score = score
                best_id = tex.texture_id
                best_dist = dist

        return best_id, self.palette[best_id].name, best_dist

    def predict_d1_neural_layers(
        self,
        minimap_rgb_256: np.ndarray,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Decompose minimap into 2 tileset layers and 2 MCAL alpha masks using trained Model D1 (D1UNet).

        Recovers:
          - tileset_1 (base layer color and texture ID)
          - tileset_2 (overlay layer color and texture ID)
          - alpha_1 and alpha_2 (continuous MCAL alpha splats)

        Falls back seamlessly to analytic demix_pixel_albedo if checkpoint is not found.

        Returns:
            texture_weights: (256, 256, K) float32 weights over palette.
            reconstructed_albedo: (256, 256, 3) composite 2D texture albedo.
            bare_illumination: (256, 256) isolated surface shading scalar.
            metadata: Diagnostic dictionary with predicted texture IDs, names, and coverage stats.
        """
        import os
        from pathlib import Path
        import torch

        ckpt = checkpoint_path
        if ckpt is None:
            # Default checkpoint locations
            default_paths = [
                Path(__file__).resolve().parents[3] / "checkpoints" / "d1_best.pt",
                Path(__file__).resolve().parents[2] / "checkpoints" / "d1_best.pt",
                Path("checkpoints/d1_best.pt"),
            ]
            for p in default_paths:
                if p.is_file():
                    ckpt = p
                    break

        if ckpt is None or not Path(ckpt).is_file():
            logger.info("D1 checkpoint not found, falling back to analytic albedo de-mixing.")
            weights, albedo, illum = self.demix_pixel_albedo(minimap_rgb_256)
            return weights, albedo, illum, {"model": "analytic_fallback"}

        try:
            from harvester.d1_model import D1UNet

            dev = torch.device(
                device
                if device is not None
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )

            model = D1UNet()
            state = torch.load(str(ckpt), map_location=dev, weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            model.to(dev)
            model.eval()

            img = np.asarray(minimap_rgb_256, dtype=np.float32)
            if img.max() > 1.5:
                img = img / 255.0

            inp = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)
            with torch.no_grad():
                pred_t1, pred_t2, pred_a1, pred_a2 = model(inp)

            t1 = pred_t1.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            t2 = pred_t2.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            a1 = pred_a1.squeeze(0).squeeze(0).cpu().numpy().clip(0.0, 1.0)
            a2 = pred_a2.squeeze(0).squeeze(0).cpu().numpy().clip(0.0, 1.0)

            # Match t1 and t2 to palette textures
            mean_c1 = np.mean(t1, axis=(0, 1))
            mean_c2 = np.mean(t2, axis=(0, 1))
            id_1, name_1, dist_1 = self.match_color_to_palette(mean_c1)
            id_2, name_2, dist_2 = self.match_color_to_palette(mean_c2)

            k = len(self.palette)
            weights = np.zeros((256, 256, k), dtype=np.float32)

            # Layer 0 (base) and Layer 1 (overlay)
            w2 = a2
            w1 = np.maximum(1.0 - w2, 0.0)
            weights[..., id_1] += w1
            weights[..., id_2] += w2

            # Normalize weights
            w_sum = np.sum(weights, axis=-1, keepdims=True)
            weights = weights / np.maximum(w_sum, 1e-6)

            # Composite albedo: sum_k w_k * T_k
            recon_albedo = np.dot(weights, self._palette_matrix.T)

            # Bare illumination: ||minimap|| / ||albedo||
            albedo_mag = np.linalg.norm(recon_albedo, axis=-1)
            img_mag = np.linalg.norm(img, axis=-1)
            bare_illum = np.clip(img_mag / np.maximum(albedo_mag, 1e-4), 0.0, 2.5)

            meta = {
                "model": "D1UNet",
                "checkpoint": str(ckpt),
                "layer_1_texture_id": id_1,
                "layer_1_texture_name": name_1,
                "layer_2_texture_id": id_2,
                "layer_2_texture_name": name_2,
                "layer_2_mean_alpha": float(np.mean(a2)),
            }

            return weights, recon_albedo, bare_illum, meta

        except Exception as e:
            logger.warning(f"Failed to run D1UNet inference ({e}), falling back to analytic de-mixing.")
            weights, albedo, illum = self.demix_pixel_albedo(minimap_rgb_256)
            return weights, albedo, illum, {"model": "analytic_fallback", "error": str(e)}
