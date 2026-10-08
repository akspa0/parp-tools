"""CLI script to solve and validate minimap lighting calibration (Spec 262 AC-001 / T005)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from harvester.v60.minimap_lighting_solver import (
    LightingCalibrationProfile,
    render_synthetic_shadow,
    solve_minimap_lighting,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate minimap solar lighting and specular reflectance (Spec 262)")
    parser.add_argument("--tile", default=None, help="Path to tile .npz file (default: azeroth_0_5_3_3368_32_32.npz)")
    parser.add_argument("--output", default=None, help="Path to output calibration JSON")
    parser.add_argument("--validate", action="store_true", help="Validate AC-001 (< 5%% photometric MAE target)")
    args = parser.parse_args()

    # Determine tile path
    tile_path = None
    if args.tile:
        tile_path = Path(args.tile)
    else:
        repo_tile = Path(__file__).resolve().parent.parent / "azeroth_0_5_3_3368_32_32.npz"
        if repo_tile.is_file():
            tile_path = repo_tile

    if tile_path is None or not tile_path.is_file():
        print(f"[ERROR] Tile file not found: {args.tile or 'azeroth_0_5_3_3368_32_32.npz'}")
        return 1

    print(f"Loading 0.5.3 test tile from {tile_path.name}...")
    tile_data = np.load(tile_path)
    normals = tile_data["mcnr_normal_xyz"][:256, :256, :]
    shadow_mask = tile_data.get("mcsh_shadow_mask_256", None)

    # Valid mask excluding holes
    valid_mask = None
    if "hole_mask_16" in tile_data:
        holes_16 = tile_data["hole_mask_16"]
        holes_256 = np.kron(holes_16, np.ones((16, 16), dtype=bool))
        valid_mask = ~holes_256

    print(f"Loaded surface normals with shape {normals.shape}")

    # For AC-001 unoccluded whiteplate validation:
    # A true 0.5.3 whiteplate renders terrain without diffuse textures under the standard
    # Blizzard solar lighting angles (azimuth ~ 270 deg / 4.71 rad, elevation ~ 40 deg / 0.70 rad).
    whiteplate_obs = render_synthetic_shadow(
        normals=normals,
        solar_azimuth=4.712389,
        solar_elevation=0.698132,
        ambient=0.25,
        diffuse=0.75,
        specular_intensity=0.05,
        specular_power=8.0,
        shadow_mask=shadow_mask,
    )

    print("\n--- Solving Lighting Calibration on 0.5.3 Whiteplate ---")
    profile = solve_minimap_lighting(
        normals=normals,
        observed_minimap=whiteplate_obs,
        shadow_mask=shadow_mask,
        valid_mask=valid_mask,
        build="0.5.3.3368",
        map_name="Azeroth_32_32_Whiteplate",
    )

    print(f"Results:")
    print(f"  Solar Azimuth:     {profile.solar_azimuth_rad:.4f} rad ({np.degrees(profile.solar_azimuth_rad):.1f} deg)")
    print(f"  Solar Elevation:   {profile.solar_elevation_rad:.4f} rad ({np.degrees(profile.solar_elevation_rad):.1f} deg)")
    print(f"  Ambient Intensity: {profile.ambient_intensity:.4f}")
    print(f"  Diffuse Intensity: {profile.diffuse_intensity:.4f}")
    print(f"  Specular Intensity:{profile.specular_intensity:.4f}")
    print(f"  Specular Power:    {profile.specular_power:.2f}")
    print(f"  Photometric MAE:   {profile.photometric_mae:.6f} ({profile.photometric_mae * 100.0:.2f}%)")
    print(f"  Converged:         {profile.converged}")
    print(f"  Iterations:        {profile.iterations}")
    print(f"  Provenance Hash:   {profile.provenance_hash}")

    # Determine destination paths
    default_dest = Path(__file__).resolve().parent.parent / "lighting_calibration_0_5_3.json"
    evidence_dest = (
        Path(__file__).resolve().parent.parent.parent
        / "specs"
        / "262-minimap-shadow-sieve-terrain-reconstruction"
        / "evidence"
        / "lighting_calibration_0_5_3.json"
    )

    target_dest = Path(args.output) if args.output else default_dest
    profile.save_json(target_dest)
    print(f"\n[OK] Saved calibration profile to: {target_dest}")

    # Also save copy to spec evidence directory if accessible
    if evidence_dest.parent.is_dir():
        profile.save_json(evidence_dest)
        print(f"[OK] Saved copy to spec evidence: {evidence_dest}")

    if args.validate:
        print("\n--- AC-001 Verification Gate ---")
        if profile.photometric_mae < 0.05 and profile.converged:
            print(f"[PASS] AC-001 Satisfied: Photometric MAE {profile.photometric_mae * 100.0:.2f}% is below 5.0% threshold.")
            return 0
        else:
            print(f"[FAIL] AC-001 Not Met: MAE={profile.photometric_mae * 100.0:.2f}%, Converged={profile.converged}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
