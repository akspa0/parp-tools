# Relative-Height Target Contract

## Encode/decode contract

1. The model's supervised target MUST be a per-tile normalized height field, never a value on a
   global/absolute elevation scale. `normalized = (h - tile_min) / max(tile_max - tile_min,
   RANGE_FLOOR)`, clipped to `[0, 1]`.
2. `tile_min`, `tile_max`, and `RANGE_FLOOR` (a fixed world-unit constant, not per-tile) MUST be
   recorded alongside every normalized target so `decode(normalized, tile_min, tile_max) =
   normalized * max(tile_max - tile_min, RANGE_FLOOR) + tile_min` exactly inverts the encoding,
   including for a genuinely flat tile (`tile_max - tile_min` below `RANGE_FLOOR`).
3. Adding any constant offset to every height value in a tile MUST NOT change that tile's
   normalized target. This MUST hold as a property test, not only an implementation intention —
   see `tests/v50/test_height_relative_model.py`.
4. `contract_version` MUST be recorded in every checkpoint's run identity and training summary. A
   checkpoint whose target math changes incompatibly (e.g. a different `RANGE_FLOOR`, a different
   clip range) MUST bump the version; inference tooling MUST refuse a checkpoint/store target-
   contract mismatch rather than silently reinterpreting values.

## Model input/output contract

1. The model's input MUST be derived only from `minimap_rgb` (or `minimap_rgb_1024`, once its
   coverage gap is closed) — no ground-truth lighting, time-of-day, absolute world coordinates, or
   map identity may reach the input tensor (constitution-adjacent: matches the existing Spec 103/106
   rule that ground-truth lighting/time is never a deployed-model input, extended here to map
   identity and absolute position specifically because they are the mechanism of the rejected
   model's failure).
2. The model MUST predict exactly one signal: the normalized relative height field. No auxiliary
   head, no multi-task loss term, no shared weights with any other model (constitution IV, Residual
   Model Chain). Growth to additional targets (normals, texture layers, liquid) requires a new spec;
   it MUST NOT be added to this model's output as a "while we're at it" extension.
3. Evaluation tooling MUST restrict validation/holdout metrics to Kalimdor and Azeroth tiles. A
   request naming any other map (including PVPZone02 or Kalidar) MUST fail closed with an explicit
   message, never silently substitute or ignore the request.

## Training/evaluation execution contract

1. No command in this feature may launch a GPU training run without a separate, explicit user
   go-ahead given at the point of execution. Preparing the curriculum, the model code, or a
   ready-to-run command does not itself authorize execution — the assistant prints the exact
   command and the user runs it (standing rule, hardened 2026-07-18 after a violation this session).
2. Every executed training run MUST write a training summary (see data-model.md) recording
   curriculum identity, split mode, target contract version, per-epoch metrics, and a
   tile-mean-baseline comparison computed in the same run — a checkpoint without this record is not
   eligible for any downstream promotion decision.
3. A run whose best validation epoch is epoch 1 MUST be treated as a structural failure signal (the
   exact symptom the absolute-elevation lane produced), not reported as a successful training run,
   even if the loss value itself looks plausible in isolation.
