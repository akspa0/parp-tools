# Quickstart: Terrain Paste and Fractal Motif Archaeology

The scripts below are planned operator commands; they are not executable until their corresponding implementation tasks are complete. No harvest or training is part of the documentation-only kickoff.

From `wow-viewer/data-harvester`:

```powershell
uv run --no-cache python scripts/v60_build_terrain_motif_corpus.py --control-corpus "../output/datasets/v60/control-v1" --real-corpus "../output/datasets/v60/real-transfer-v1" --output "../output/datasets/v60/terrain-motif-v1" --window-size 96 --stride 48 --seed 14001
uv run --no-cache python scripts/v60_validate_terrain_motif_corpus.py --corpus "../output/datasets/v60/terrain-motif-v1" --write-report
uv run --no-cache python scripts/v60_visualize_terrain_motif_corpus.py --corpus "../output/datasets/v60/terrain-motif-v1" --output-dir "../output/datasets/v60/terrain-motif-v1/visual-review" --rows-per-family 4
uv run --no-cache python scripts/v60_retrieve_terrain_motifs.py --corpus "../output/datasets/v60/terrain-motif-v1" --write-report --write-atlas
uv run --no-cache python scripts/v60_analyze_terrain_paint_order.py --corpus "../output/datasets/v60/terrain-motif-v1" --write-report --write-atlas
```

The first user-owned execution gate is the atlas and retrieval report. Training commands will be added only after G0 and G1 pass.
