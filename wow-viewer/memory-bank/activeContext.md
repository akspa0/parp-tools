# Active Context — wow-viewer

Last updated: 2026-07-04
Keep current contract only. Older notes live in `memory-bank/archive/2026-07-04-pre-2026-06-27.md`.

## Main target

- Spec 089 `089-dav2-height-predictor`.
- Local first. Proof owner = real 12 GB CUDA run, not RunPod.
- Source work through Phase 7 is local-complete.
- Current local proof: `uv run python -m pytest tests/v23 -m v23 -q` -> `28 passed, 14 warnings`.

## Current gate

- Run T035 on real local 12 GB card.
- Save `peak_vram.json`, smoke metrics, and checkpoint output.
- Do not reopen remote proof until local 12 GB envelope is real.

## V23 contract

- One signal only: height.
- Input = Spec 088 V22 paths-only store.
- Model = DA-V2-Small encoder + LoRA + compact height head + affine anchor.
- Trainer handles memory profiles, grad accumulation, and OOM backoff.
- Inference + CAI stitch path exist and are deterministic in local tests.

## V22 contract

- Spec 088 is active V22 design.
- Canonical stores exist for `0_5_3_3368` and `3_3_5_12340`.
- Store is `paths_only`; no embedded M2/WMO/BLP payload blobs.
- Remaining bounded gate: rerun same proof for `4_0_0_11927`, then close 088.

## UI compatibility lane

- Spec 080 Phase A source slice is landed in `gillijimproject_refactor/src/MdxViewer`.
- Bottom display bar owns terrain/world toggles.
- Top toolbar is launcher strip.
- PM4 Object Match and PM4/WMO Correlation now render from `DrawUI()`.
- Proof level = source-only. Legacy `MdxViewer.sln` build still fails on pre-existing missing refs outside this slice.

## Recent background still live

- 2026-06-30: Spec 088 replaced broken V22 payload plans with `V22Enrich` + paths-only store.
- 2026-06-29: Spec 077 loss-gate fix moved teacher-prior weighting to `object_precise_mask` first.
- Spec 076 and Spec 077 remain paused/background unless user reopens them.

## Boundaries

- Do not move new work back into `gillijimproject_refactor`.
- Do not claim remote proof from Pod creation alone.
- Do not claim UI compile validation from legacy-solution failures outside touched slice.
- Staged clients only under `output/tmp/wowarchive-clients/`.
