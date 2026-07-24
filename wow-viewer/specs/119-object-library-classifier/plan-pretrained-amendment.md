# Amendment: Pretrained Backbone for Spec 119 Classifier

**Date**: 2026-07-23 | **Spec**: [spec.md](spec.md) | **Original plan**: [plan.md](plan.md)

## Change

Replace the from-scratch conv encoder in `ObjectClassifier` with a **DINOv2 ViT-S/14** backbone (21M params, 384-d embedding). The linear head stays: 384 → 4 classes. The embedding is the ViT's [CLS] token output (384-d), which is reused for the quality lens and retrieval.

## Rationale

The from-scratch 98K-param encoder hits a ceiling on wmo recall (43%) because it lacks the representation capacity to separate the genuinely confusable classes. A pretrained ViT brings 21M params of general visual knowledge — texture, shape, material cues — that are directly applicable to the clean 128px library captures. The 384-d embedding is much richer than the current 128-d, improving near-duplicate detection and any future retrieval.

## Changes

- **FR-003 relaxed**: "MUST be small and trained from scratch" → "MUST be small and may use a pretrained backbone (DINOv2 ViT-S/14 or equivalent single-digit-millions-to-21M param model)."
- **SC-005 relaxed**: "single-digit-millions" → ≤21M params (DINOv2 ViT-S/14).
- **New dependency**: `torch.hub` (DINOv2 is available via `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`). No additional PyPI package needed — DINOv2 is served through PyTorch Hub.
- **Architecture**: `ObjectClassifier` gains a `backbone` parameter: `"dinov2_vits14"` (default) or `"scratch"` (original, for comparison). The backbone is frozen at first with only the linear head trained (linear probing), then optionally fine-tuned end-to-end.
- **Input size**: DINOv2 ViT-S/14 expects 224×224. The library captures are 128×128. Two options: (a) resize captures to 224 (lossy, but DINOv2's patch-based processing handles it), or (b) change the capture pipeline to 224. Option (a) is simpler — resize at training time (PIL.BILINEAR), no re-capture needed.
- **New dependency check**: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')` — verify it downloads and runs on the target machine.

## Delivery

1. Add `dinov2_vits14` backbone option to `ObjectClassifier` in `classifier_model.py`.
2. Update `classifier_train.py` to accept `--backbone dinov2_vits14` (default `scratch` for backward compatibility).
3. Retrain the classifier on the existing split with `--backbone dinov2_vits14 --epochs 60 --lr 1e-3 --confirm-run`.
4. Compare wmo recall against the scratch baseline.
5. If the DINOv2 classifier significantly improves wmo recall, promote it as the default and update the quality lens to use the 384-d embeddings.

## Comparison

| Metric | Scratch (base 16) | DINOv2 ViT-S/14 |
|--------|-------------------|-----------------|
| Params | 98K | 21M |
| Embedding dim | 128 | 384 |
| WMO recall (held-out) | 43% | Target: ≥60% |
| Epochs | 60 | 60 (linear 30 + full 30) |
| Training speed | fast | ~2–4× slower (21M params) |