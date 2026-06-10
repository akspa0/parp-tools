# Experiments

Run on real tiles.

Baselines:
- A) No snapping: default flags

Snapping runs:
- B) --snap-to-plane --height-scale 1.0
- C) --snap-to-plane --height-scale 0.02777778   # 1/36
- D) --snap-to-plane --height-scale 0.0625       # 1/16

Optional for diagnosis: add `--group-by surface`.

Outputs: capture visuals and residual CSVs if enabled; note crack closure vs explosion.
