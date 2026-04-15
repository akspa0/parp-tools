from __future__ import annotations

import argparse
import torch
from pathlib import Path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from v76_prediction_utils import (
    DEFAULT_MODEL_PATH,
    ensure_unique_sample_id,
    finalize_prediction_dataset,
    load_source_image,
    load_v76_model,
    predict_height_and_albedo,
    prepare_prediction_layout,
    write_prediction_sample,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run V7.6 inference and emit a structured predicted dataset.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Checkpoint path.")
    parser.add_argument("--input-dir", default="inference_input", help="Directory containing source PNG images.")
    parser.add_argument("--output-dir", default="inference_output", help="Prediction dataset root.")
    parser.add_argument("--no-mesh", action="store_true", help="Skip OBJ/MTL export.")
    parser.add_argument("--source-kind", default="arbitrary_image", choices=["arbitrary_image", "harvested_tile"], help="Source provenance label for all inputs in this run.")
    return parser.parse_args()


def run_inference(args):
    model_path = Path(args.model_path)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    layout = prepare_prediction_layout(output_dir)

    print(f"Loading model from {model_path}...")
    model = load_v76_model(model_path, DEVICE)

    input_files = sorted(input_dir.glob("*.png"))
    print(f"Found {len(input_files)} images in {input_dir}")

    if not input_files:
        print("No images found. Copy PNG inputs into the input directory first.")
        return

    records = []
    used_ids: set[str] = set()
    for img_path in input_files:
        try:
            source_image, input_tensor, source_meta = load_source_image(img_path)
            height_image, albedo_image, height_map = predict_height_and_albedo(model, input_tensor, DEVICE)
            sample_id = ensure_unique_sample_id(img_path.stem, used_ids)
            record = write_prediction_sample(
                layout=layout,
                sample_id=sample_id,
                source_path=img_path,
                source_kind=args.source_kind,
                source_meta=source_meta,
                model_path=model_path,
                height_image=height_image,
                albedo_image=albedo_image,
                height_map=height_map,
                write_mesh=not args.no_mesh,
            )
            records.append(record)
            print(f"Processed: {img_path.name} -> predictions/{sample_id}.json")
            _ = source_image
        except Exception as exc:
            print(f"Error processing {img_path.name}: {exc}")

    if not records:
        print("No predictions were written.")
        return

    manifest_path = finalize_prediction_dataset(layout=layout, model_path=model_path, records=records)
    print(f"Prediction dataset complete: samples={len(records)}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    run_inference(parse_args())
