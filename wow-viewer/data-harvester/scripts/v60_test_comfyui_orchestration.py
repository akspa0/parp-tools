"""Interactive / CLI test for ComfyUI SAM 3.1 & Mothership VLM Orchestration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from harvester.v60.comfyui_orchestrator import ComfyUIConnectionError, ComfyUIOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Test ComfyUI Orchestrator for SAM 3.1 & Mothership VLM")
    parser.add_argument("--url", default="http://127.0.0.1:8199", help="ComfyUI server URL")
    parser.add_argument("--image", default="v18_minimap_row04851.png", help="Path to test image")
    parser.add_argument("--test-vlm", action="store_true", help="Run live Mothership VLM test")
    parser.add_argument("--test-sam", action="store_true", help="Run live SAM 3.1 segmentation test")
    args = parser.parse_args()

    orchestrator = ComfyUIOrchestrator(base_url=args.url)

    print(f"Connecting to ComfyUI at {args.url}...")
    try:
        health = orchestrator.check_health()
        print(f"[OK] Health check passed!")
        devices = health.get("devices", [])
        for dev in devices:
            print(f"  Device: {dev.get('name')} (VRAM: {dev.get('vram_free', 0) / (1024**3):.2f} GB free)")
    except ComfyUIConnectionError as e:
        print(f"[ERROR] Failed to connect: {e}")
        return 1

    # Verify custom nodes presence
    print("\nVerifying custom node availability...")
    info = orchestrator.get_object_info()
    has_sam3 = "SAM3_Detect" in info
    has_mothership = "MothershipGemmaVLM" in info
    print(f"  SAM3_Detect: {'PRESENT' if has_sam3 else 'MISSING'}")
    print(f"  MothershipGemmaVLM: {'PRESENT' if has_mothership else 'MISSING'}")

    img_path = Path(args.image)
    if not img_path.is_file():
        # Fallback to local files in data-harvester
        for candidate in ["v18_minimap_row04851.png", "azeroth_32_32_synth.png"]:
            cand_p = Path(__file__).resolve().parent.parent / candidate
            if cand_p.is_file():
                img_path = cand_p
                break

    if not img_path.is_file():
        print(f"[WARN] No test image found at {args.image}, skipping inference tests.")
        return 0

    print(f"\nUsing test image: {img_path.name}")

    if args.test_vlm:
        print("\n--- Testing Mothership VLM Inference ---")
        prompt = "Describe the prominent terrain, structures, or roads visible in this overhead minimap crop."
        try:
            print(f"Prompt: {prompt}")
            res = orchestrator.analyze_vlm(img_path, prompt=prompt, slot=1, temperature=0.2, timeout_seconds=45.0)
            print(f"VLM Response:\n{res}")
        except Exception as ex:
            print(f"VLM Test Error: {ex}")

    if args.test_sam:
        print("\n--- Testing SAM 3.1 Object Segmentation ---")
        try:
            mask = orchestrator.segment_objects(img_path, threshold=0.5, refine_iterations=2, timeout_seconds=60.0)
            out_mask_path = Path("out") / f"test_mask_{img_path.stem}.png"
            out_mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(out_mask_path)
            print(f"Segmentation successful! Mask shape: {mask.shape}, non-zero pixels: {int((mask > 0).sum())}")
            print(f"Saved mask to: {out_mask_path}")
        except Exception as ex:
            print(f"SAM Test Error: {ex}")

    print("\nOrchestrator smoke test completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
