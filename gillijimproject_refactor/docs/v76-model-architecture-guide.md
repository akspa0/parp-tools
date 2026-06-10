# WoW Paired Reconstruction Model V7.6

Updated Apr 14, 2026.

V7.6 is a separate reconstruction branch from the active V7.5.1 multichannel terrain regressor.

V7.5.1 is the current harvested-corpus terrain model line. V7.6 is the paired-output image-to-terrain-material line that tries to take an input image, infer terrain geometry, and also infer a cleaner terrain albedo surface from the same observation.

That distinction matters because V7.6 answers a different question:

- not only "what height surface could have produced this image?"
- but also "what cleaner terrain material surface should sit under that image once baked lighting, object clutter, or renderer-specific appearance is stripped away?"

## Snapshot

- Input channels: `3`
- Input resolution: `512x512`
- Outputs: `1` predicted height channel plus `3` predicted albedo channels
- Core architecture: shared ResNet34 encoder with dual U-Net-style decoder heads
- Training cache: `cached_v7_6/`
- Training script: `src/WoWMapConverter/scripts/train_v7_6.py`
- Cache builder: `src/WoWMapConverter/scripts/cache_v7_6_data.py`
- Inference script: `src/WoWMapConverter/scripts/inference_v7_6.py`
- Map quilt script: `src/WoWMapConverter/scripts/stitch_full_map.py`
- Current status: documented branch with a real code path, but not yet integrated into the structured harvested-corpus workflow used by V7.5.1

## Intended Job

V7.6 is meant to learn three things from paired training data.

### 1. Terrain Shape Recovery

The model sees an input image and tries to predict a `512x512` terrain height surface.

This is the geometry side of the task:

- roads, ridges, coastlines, cliffs, and basin-like color transitions in the source image should correlate with height structure
- the predicted height output is the geometry surface used for later mesh generation or dataset packaging

### 2. Terrain Appearance Recovery

The model also predicts a clean albedo image.

The purpose of that output is not to reproduce the raw source image exactly. It is to learn a cleaner terrain-material interpretation of the scene:

- less baked lighting
- less minimap-specific shading bias
- less renderer-specific tinting
- a flatter terrain-color surface that is closer to "terrain material" than to "final rendered screenshot"

### 3. Reconstructable Output Packaging

The branch is also meant to be usable on arbitrary image inputs.

That means the output should not stop at a loose PNG dump. It should become a structured predicted dataset with:

- the source image preserved
- the model provenance recorded
- the predicted height and albedo stored deterministically
- optional mesh exports and quilt outputs linked back to the prediction record

That packaging contract is specified separately in `docs/v76-output-dataset-spec.md`.

## What The Data Teaches

The checked-in V7.6 pipeline is a paired-supervision setup.

The three important surfaces are:

1. input image
2. target height
3. target albedo

### Input Image

The current cache builder reads a rendered tile image and uses that as the only model input.

In the checked-in script this is stored as:

- `input_<x>_<y>.pt`

Current code shape:

- RGB only
- resized or enforced to `512x512`
- no explicit WDL, object-mask, liquid, or brush channels in this branch

This is what makes V7.6 different from V7.5.1. V7.6 is trying to solve more from the image alone.

### Target Height

The cache builder reads the global heightmap target and stores it as:

- `target_height_<x>_<y>.pt`

This is the geometry teacher.

It teaches the height decoder how image cues map to terrain elevation.

### Target Albedo

The cache builder synthesizes a terrain albedo target and stores it as:

- `target_albedo_<x>_<y>.pt`

This target is built from harvested terrain-layer information, not from a GAN.

The current cache path synthesizes albedo by:

- reading `terrain_data.chunk_layers`
- resolving terrain texture images from `tilesets/`
- compositing them with available alpha masks

That teaches the albedo decoder a different problem from the height decoder:

- not "what was rendered"
- but "what terrain material stack likely sits underneath the rendered image"

## Why Joint Training Helps

The two targets reinforce each other.

The shared encoder is forced to learn features that matter to both outputs:

- large terrain landforms help height prediction directly
- texture and terrain-pattern boundaries help albedo prediction directly
- roads, shorelines, cliffs, terraces, and paint transitions often help both tasks

In practice that means the encoder is learning a terrain representation that is useful for:

- geometry
- material interpretation
- later packaging into a reusable predicted dataset

## Training Data Path

The checked-in V7.6 branch currently uses a cache-preparation step instead of reading the structured harvested dataset root directly during training.

### Cache Builder

`cache_v7_6_data.py` currently does the following:

1. reads a harvested dataset root
2. loads the tile minimap image as model input
3. loads the tile global heightmap as the geometry target
4. synthesizes a terrain albedo target from chunk layers, alpha masks, and tileset textures
5. writes `.pt` tensors into `cached_v7_6/`

Current cached tensor families:

- `input_<x>_<y>.pt`
- `target_height_<x>_<y>.pt`
- `target_albedo_<x>_<y>.pt`

Important boundary:

- the checked-in script currently hardcodes a specific root under `test_data/vlm-datasets/053_Azeroth_v30`
- that makes it a real branch with real code, but not yet a generalized dataset-root consumer like the V7.5.1 training path

## Model Shape

The checked-in `train_v7_6.py` defines a shared encoder with two decoder heads.

### Encoder

- backbone: pretrained `ResNet34`
- shared across both tasks
- used as the terrain-image feature extractor

### Bottleneck

- `1024` channels at the deepest layer

### Height Head

- decodes toward a single-channel height prediction
- output activation: sigmoid
- training target normalized to `0..1`

### Albedo Head

- decodes toward a 3-channel RGB prediction
- output activation: sigmoid
- intended to approximate a clean terrain albedo surface

## Losses

The checked-in V7.6 branch uses a simple paired loss stack.

### Height Loss

- `L1` loss against the normalized target height map

This teaches the model the geometry surface directly.

### Albedo Loss

- `L1` reconstruction loss against the synthesized albedo target
- `0.1 * VGG perceptual loss`

This teaches the model both:

- pixel-level terrain-color recovery
- higher-level appearance consistency beyond straight per-pixel matching

### No GAN In The Checked-In V7.6 Branch

Unlike the active V7.5.1 terrain line, the checked-in V7.6 script does not currently include adversarial training.

That is important for public framing:

- the branch still depends on paired supervision from real harvested data
- the albedo target is synthesized deterministically from harvested terrain layers and textures
- there is no checked-in GAN data-generation path here

## Inference Contract

`inference_v7_6.py` currently accepts arbitrary `.png` inputs from `inference_input/`.

Current behavior:

- loads any input image as RGB
- resizes it to `512x512` if needed
- predicts height and albedo
- saves loose files in `inference_output/`
- optionally writes OBJ and MTL outputs per image

Current loose outputs:

- `<name>_height_pred.png`
- `<name>_albedo_pred.png`
- `<name>.obj`
- `<name>.mtl`

This is workable as a prototype but not strong enough as a dataset surface.

## Structured Output Dataset Goal

The intended V7.6 inference result should be a predicted dataset, not just a folder of unnamed artifacts.

That is the reason for the separate spec in `docs/v76-output-dataset-spec.md`.

The goal is:

- preserve provenance of the source image
- preserve model provenance of the checkpoint used
- store predicted height, albedo, and mesh outputs under a stable layout
- emit per-sample JSON and run-level manifest files the same way harvested datasets already do

## Full-Map Quilt Path

`stitch_full_map.py` is the large-scale map reconstruction companion to `inference_v7_6.py`.

Current behavior:

- scans `MapName_X_Y.png` tiles
- runs inference tile by tile
- writes full-map albedo and height quilts
- writes per-tile OBJ/MTL/texture bundles under an `objs/` folder

Current quilt outputs are still loose and should eventually be folded into the structured predicted-dataset contract as optional stitched artifacts.

## What V7.6 Should Be Used For

The cleanest description of the intended V7.6 use is:

1. start from a source image
2. infer a terrain geometry interpretation
3. infer a cleaner terrain-material interpretation
4. package the result as a structured predicted dataset for later inspection, stitching, or downstream conversion

That makes it useful for:

- arbitrary image-driven terrain reconstruction experiments
- dataset expansion experiments where the source is an image but the output should still look like terrain data
- later translator-style work where image input should yield reusable terrain assets instead of a one-off mesh dump

## Boundary

V7.6 is a real branch, but it is not the same thing as the active V7.5.1 harvested-corpus terrain line.

Current limitations:

- current cache builder is hardcoded to a legacy example root instead of the generalized `datasets/` workflow
- current inference output is loose-file oriented instead of spec-driven
- current height-to-world conversion uses heuristic assumptions such as `MAX_HEIGHT = 1200.0` in inference and quilt export
- current branch is not yet the canonical grounded terrain-training path; V7.5.1 still owns that role

So the right way to present it today is:

- V7.6 is the documented paired-output reconstruction branch
- it is meant to learn geometry and material interpretation from image input
- it should emit a structured predicted dataset
- but its code path still needs to be aligned with the structured harvested-dataset workflow before it can replace the current V7.5.1 path