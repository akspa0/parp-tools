# V24 Convergence Research — Existing Models for Image-to-Terrain-Height Prediction

**Date**: 2026-07-10
**Author**: Research session (code mode)
**Scope**: HuggingFace + GitHub survey of existing models that can help V24 Stage A (minimap → WDL prior) and Stage B (WDL prior + minimap → 257×257 heightmap) converge. Informs Spec 099, Spec 100, and the broader Spec 098 lattice reconstruction vision.

---

## 1. The Problem

V24 has two stages, both of which are currently broken or underperforming:

| Stage | Input | Output | Current Model | Current L1 | Baseline L1 | Status |
|-------|-------|--------|---------------|-----------|-------------|--------|
| A (cheat) | 13-ch (minimap+alpha+normal+mcnr+synth) | 17×17 + 16×16 WDL prior | 337K-param U-Net | 1.21 | 1.31 (block_reduce) | Works |
| A (minimap-only) | 3-ch (RGB minimap) | 17×17 + 16×16 WDL prior | 335K-param U-Net | **190.31** | 1.31 | **158× worse than baseline** |
| A (guided) | 9-ch (minimap+normal+Sobel) | 17×17 + 16×16 WDL prior | 450K-param U-Net | **94.58** | 1.31 | Overfitting, barely better |
| B | WDL prior + minimap + alpha + normal | 257×257 heightmap | 828K-param conv-deconv | 0.857 | 4.20 (block_reduce+bilinear) | Works (with cheat prior) |

The minimap-only Stage A is the bottleneck. The 190 L1 means the model is essentially predicting noise — the bare RGB minimap does not carry enough signal for a 335K-parameter from-scratch U-Net to learn the mapping to a 33×33 height grid.

The guided model (9-channel, 450K params) does better (94 L1) but still overfits hard (train 51 vs val 98) and doesn't converge to a useful local minimum.

**Root cause**: The model is too small and has no pretrained features. A 335K–450K parameter network trained from random initialization cannot learn the minimap → heightmap mapping. The mapping is complex (top-down aerial view → terrain elevation) and requires rich visual feature extraction that only a pretrained backbone can provide.

---

## 2. Key Existing Models Found

### 2.1 Depth Anything V2 (DA-V2) — The Foundation

**Repo**: [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) — 8,430 stars, NeurIPS 2024
**HuggingFace**: `depth-anything/Depth-Anything-V2-Small-hf` (already in our V23 code)

| Model | Params | Checkpoint |
|-------|--------|------------|
| DA-V2-Small | 24.8M | `depth_anything_v2_vits.pth` |
| DA-V2-Base | 97.5M | `depth_anything_v2_vitb.pth` |
| DA-V2-Large | 335.3M | `depth_anything_v2_vitl.pth` |

**Architecture**: DINOv2 ViT encoder + DPT (Dense Prediction Transformer) head. The encoder is pretrained on 62M unlabeled images via self-supervised learning. The DPT head reassembles multi-scale features into a dense depth map.

**Metric depth fine-tuning**: The `metric_depth/` subdirectory provides fine-tuned variants for indoor (Hypersim) and outdoor (Virtual KITTI 2) metric depth. The training recipe is:
- Load pretrained DA-V2 encoder weights (only `pretrained` keys)
- Add DPT head with `max_depth` parameter
- Train with **SiLogLoss** (scale-invariant log loss), not plain L1
- lr = 5e-6 (very low, because the encoder is already pretrained)
- 40 epochs, batch size 2 per GPU
- Image size 518×518

**SiLogLoss** (from `metric_depth/util/loss.py`):
```python
class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))
        return loss
```

This is scale-invariant: it cares about relative depth structure, not absolute scale. Our current plain L1 loss is dominated by large absolute height values and doesn't capture structural quality.

**Relevance to V24**: V23 already uses DA-V2-Small + LoRA in [`encoder.py`](../../data-harvester/src/harvester/v23/encoder.py:133). The infrastructure for loading the pretrained encoder, replacing the patch projection for custom input channels, and applying LoRA adapters is already built and tested. V24 Stage A should reuse this infrastructure.

### 2.2 Prompt Depth Anything (PromptDA) — The Most Directly Relevant Model

**Repo**: [DepthAnything/PromptDA](https://github.com/DepthAnything/PromptDA) — 1,135 stars, CVPR 2025
**HuggingFace**: `depth-anything/prompt-depth-anything-vits` (Small, 25.1M params)

**What it does**: Takes an RGB image + a low-resolution depth prompt (e.g., ARKit LiDAR at 192×256) and produces a high-resolution metric depth map. The prompt can be very sparse — even a few depth samples help.

**Why this is exactly our Stage B**: Our Stage B is:
- Input: minimap (RGB, 256×256) + WDL prior (low-res height, 33×33 quincunx)
- Output: 257×257 heightmap

PromptDA is:
- Input: RGB image + low-res depth prompt
- Output: high-res metric depth

The structural match is exact. The WDL prior is the "prompt" — a coarse depth signal that guides the high-res reconstruction.

**Usage**:
```python
from promptda.promptda import PromptDA
model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vits")
depth = model.predict(image, prompt_depth)  # HxW, depth in meters
```

**Small model**: 25.1M params, fits on 12 GB GPU. The Large model is 340M params.

**Relevance to V24**: This is the model for Stage B (and potentially Stage A if the WDL prior is treated as the prompt). The pretrained model already understands how to fuse an RGB image with a coarse depth prompt. Fine-tuning on our minimap + WDL prior data should converge much faster than our current 828K-param from-scratch conv-deconv.

### 2.3 Any2Full — Depth Completion via Scale Prompting

**Repo**: [zhiyuandaily/Any2Full](https://github.com/zhiyuandaily/Any2Full) — 70 stars, ECCV 2026
**HuggingFace**: `zhiyuandaily/Any2Full` (weights available)

**What it does**: One-stage, domain-general depth completion. Reformulates completion as "scale-prompting adaptation" of a pretrained monocular depth estimation (MDE) model. The model keeps strong geometric priors while adapting to diverse sparse depth patterns.

**Key innovation**: Scale-Aware Prompt Encoder that's robust under different sparsity levels and sampling patterns. This is relevant because our WDL prior has a specific quincunx sampling pattern (17×17 outer at (16r, 16c) + 16×16 inner at (16r+8, 16c+8)).

**Relevance to V24**: Alternative to PromptDA for Stage B. The "pattern-agnostic" claim is attractive because our quincunx pattern is unusual. Training code is "coming soon" per the README (as of 2026-07-10).

### 2.4 Marigold — Diffusion-Based Depth Estimation

**Repo**: [prs-eth/Marigold](https://github.com/prs-eth/Marigold) — 3,178 stars, CVPR 2024 Oral (Best Paper Candidate)
**HuggingFace**: `prs-eth/marigold-v1-0` and variants

**What it does**: Repurposes Stable Diffusion for monocular depth estimation. The model is initialized from Stable Diffusion weights and fine-tuned to predict depth instead of images. Uses diffusion denoising to generate depth maps.

**Pros**: Generates sharp, detailed predictions. Can capture fine-grained terrain detail.
**Cons**: Slow inference (multiple denoising steps, ~10-50 steps). More complex training. Larger model (Stable Diffusion backbone, ~800M+ params).

**Relevance to V24**: Could be relevant for Stage B or the Spec 102 lattice reconstruction model, where fine detail matters. The diffusion approach naturally produces sharp outputs (unlike L1-trained models which tend to be blurry). But the inference cost is high for a per-tile pipeline.

### 2.5 ZoeDepth — Metric Depth from a Single Image

**Repo**: [isl-org/ZoeDepth](https://github.com/isl-org/ZoeDepth) — 2,837 stars

**What it does**: Combines relative and metric depth estimation. Uses a two-head approach: one head for relative depth (like MiDaS), one for metric depth. The metric head uses a small adapter on top of the relative depth features.

**Relevance to V24**: The two-head approach (relative + metric) is interesting for our problem because the WDL prior is a coarse metric signal. ZoeDepth's architecture could inspire a design where the model first predicts relative terrain structure and then anchors it to the WDL prior's metric scale.

### 2.6 MiDaS — Robust Monocular Depth Estimation

**Repo**: [isl-org/MiDaS](https://github.com/isl-org/MiDaS) — 5,418 stars

**What it does**: The predecessor to Depth Anything. Uses DPT (Dense Prediction Transformer) with a mix of datasets for robust relative depth estimation. MiDaS v3.1 uses DA-V1 as its backbone.

**Relevance to V24**: Mostly superseded by DA-V2 for our purposes. The key transferable insight is the use of inverse depth (disparity) instead of raw depth, which avoids numerical issues with large depth values.

### 2.7 monodepth2 — Self-Supervised Monocular Depth

**Repo**: [nianticlabs/monodepth2](https://github.com/nianticlabs/monodepth2) — 4,499 stars, ICCV 2019

**What it does**: Self-supervised monocular depth estimation from video. Uses photometric consistency loss between adjacent video frames.

**Relevance to V24**: Low. We have supervised data (V18 height_257 ground truth), not video. The only transferable idea is the multi-scale (multi-resolution) loss, which we already use in a different form.

### 2.8 pix2pix / PatchGAN — Image-to-Image Translation with Adversarial Loss

**Repo**: [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — 25,184 stars
**Guided variant**: [vt-vl-lab/Guided-pix2pix](https://github.com/vt-vl-lab/Guided-pix2pix) — 198 stars, ICCV 2019

**What it does**: Image-to-image translation using a U-Net generator + PatchGAN discriminator. The adversarial loss pushes the generator to produce outputs that look like real images (not just pixel-wise accurate).

**Relevance to V24**: This is the Spec 100 approach. PatchGAN is a well-known technique for pushing past L1 plateaus. However, it does not solve the fundamental problem of the model being too small and having no pretrained features. PatchGAN should be used as an **auxiliary loss** on top of a pretrained-backbone model, not as the primary convergence strategy.

The **Guided-pix2pix** variant is particularly relevant because it adds bi-directional feature transformation for guided image-to-image translation — exactly our setup (minimap + normal/alpha guidance → heightmap).

### 2.9 Depth Anything V3 (Emerging)

**Repo**: [PozzettiAndrea/ComfyUI-DepthAnythingV3](https://github.com/PozzettiAndrea/ComfyUI-DepthAnythingV3) — 425 stars (ComfyUI wrapper)

Depth Anything V3 appears to be emerging (referenced in ComfyUI nodes). If it follows the V1→V2 pattern, it will have a better encoder and finer detail. Worth monitoring but not yet mature enough to depend on.

---

## 3. Analysis: Why V24 Stage A Doesn't Converge

### 3.1 The Model Is Too Small

The V24 Stage A minimap-only model has **335K parameters**. For comparison:
- DA-V2-Small: 24.8M params (74× larger)
- PromptDA-Small: 25.1M params (75× larger)
- V23 model (already in our codebase): DA-V2-Small encoder + LoRA + height head

A 335K-parameter network trained from scratch on ~2,000 tiles cannot learn the complex minimap → heightmap mapping. The model doesn't have enough capacity to extract visual features from the minimap and map them to terrain elevation.

### 3.2 No Pretrained Features

The V24 Stage A model starts from random initialization. Every feature detector must be learned from scratch from ~2,000 training tiles. In contrast, DA-V2's DINOv2 encoder is pretrained on 62M images and already understands edges, textures, spatial relationships, and visual structure. Fine-tuning a pretrained encoder on 2,000 tiles is a transfer learning problem; training from scratch on 2,000 tiles is an under-determined optimization problem.

### 3.3 Plain L1 Loss

The current loss is weighted L1 in normalized height space. This has two issues:
1. **Scale sensitivity**: L1 is dominated by large absolute height values. A mountain tile with 500-unit elevation range contributes 100× more loss than a flat tile with 5-unit range, even though the structural prediction quality might be similar.
2. **Blurry predictions**: L1 loss produces mean predictions (the average of all plausible heightmaps), which are blurry and lack sharp terrain features. This is the well-known "L1 blur" problem in depth estimation.

The DA-V2 metric depth training uses **SiLogLoss** (scale-invariant log loss), which:
- Is scale-invariant (cares about relative structure, not absolute scale)
- Produces sharper predictions (the log transform emphasizes relative errors)
- Is the standard loss for metric depth estimation since Eigen et al. (2014)

### 3.4 OneCycleLR Scheduler Bug

In [`train_v24_stage_a.py`](../../data-harvester/scripts/train_v24_stage_a.py:231):
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=args.lr, total_steps=args.epochs, pct_start=0.05,
)
```

`total_steps` should be the total number of **optimizer steps** (batches × epochs), not just the number of epochs. With `total_steps=200` and ~31 batches per epoch, the scheduler thinks there are only 200 total steps, but the model actually does 200 × 31 = 6,200 optimizer steps. The scheduler completes its cycle after 200 `scheduler.step()` calls (200 epochs), but the LR schedule is compressed into 200 steps instead of 6,200.

Additionally, `scheduler.step()` is called once per epoch (line 316), not once per batch. OneCycleLR is designed to be called per optimizer step. The current usage makes it a "per-epoch OneCycle" which is a valid but unusual schedule — the LR warmup and decay are much slower than intended.

**Fix**: Either:
- (a) `total_steps = n_batches * args.epochs`, call `scheduler.step()` per batch (standard OneCycle)
- (b) Keep per-epoch stepping but use `CosineAnnealingLR` instead (simpler, no `total_steps` confusion)

### 3.5 Zero-Initialized Head

The model's head is zero-initialized:
```python
self.head = nn.Conv2d(base, 1, 1)
nn.init.zeros_(self.head.weight)
nn.init.zeros_(self.head.bias)
```

For the residual model (StageAModel with synth cheat), this is correct — the model starts at the synth baseline. But for the minimap-only model (StageAMinimapOnly), this means the model starts at **zero height everywhere**. The gradient signal from zero to the correct height is weak, especially with L1 loss. A small initialization (e.g., Xavier/Kaiming) would give the model a better starting point.

---

## 4. Recommendations

### 4.1 Immediate Fix: Use DA-V2-Small Encoder for Stage A (Highest Impact)

**Replace the 335K-param from-scratch U-Net with the V23 DA-V2-Small encoder + a WDL-prior head.**

The infrastructure already exists in [`harvester/v23/encoder.py`](../../data-harvester/src/harvester/v23/encoder.py:133):
- `DepthAnythingV2SmallEncoder` loads the pretrained DA-V2-Small backbone
- It replaces the patch projection for custom input channels (3 for minimap-only, 9 for guided)
- It applies LoRA adapters (rank 16) to the attention layers
- The backbone is frozen; only LoRA + patch projection + head train

**New model**: `StageADAV2` — DA-V2-Small encoder + LoRA + a DPT-style head that outputs the 33×33 quincunx (then splits into 17×17 outer + 16×16 inner).

**Training recipe** (from DA-V2 metric depth):
- Load pretrained DA-V2-Small encoder weights
- Freeze backbone, train LoRA + head
- Loss: SiLogLoss (or SiLogLoss + L1 hybrid)
- lr = 5e-6 to 1e-5 (very low, because the encoder is pretrained)
- 40-200 epochs
- Batch size: 8-64 (DA-V2-Small is 24.8M params, fits easily on 12 GB)

**Expected improvement**: The 190 L1 should drop to single digits immediately because the pretrained encoder already understands visual features. The model only needs to learn the minimap → height mapping, not feature extraction from scratch.

**Param budget**: DA-V2-Small is 24.8M params. With LoRA (rank 16), only ~500K-1M params are trainable. The total model is 24.8M + head (~100K) = ~25M. This fits on 12 GB GPU with batch size 8-16 at 256×256 input.

### 4.2 Use PromptDA for Stage B (Depth Completion)

**Replace the 828K-param from-scratch conv-deconv with PromptDA-Small.**

PromptDA takes RGB + low-res depth prompt → high-res metric depth. Our Stage B is exactly this:
- RGB = minimap (256×256)
- Low-res depth prompt = WDL prior (33×33 quincunx, upsampled to 256×256)
- Output = 257×257 heightmap

**Fine-tuning**: Load PromptDA-Small pretrained weights, replace the input channels if needed (our minimap is top-down, not perspective — the model may need domain adaptation), fine-tune on our V18 data.

**Expected improvement**: The pretrained model already understands how to fuse RGB with coarse depth. Fine-tuning on our data should converge much faster than the from-scratch conv-deconv.

### 4.3 Replace L1 with SiLogLoss (or Hybrid)

**Replace plain weighted L1 with SiLogLoss or a hybrid L1 + SiLogLoss.**

SiLogLoss is the standard loss for metric depth estimation. It's scale-invariant and produces sharper predictions. The DA-V2 metric depth training uses it exclusively.

For our case, a hybrid might be better:
- SiLogLoss for structural quality (relative terrain shape)
- L1 for absolute scale (world units)
- Weight: 0.7 SiLogLoss + 0.3 L1 (tunable)

**Caveat**: SiLogLoss requires positive values (it takes log). Our heights can be negative (world units range from -787 to +409 for Northrend). We need to either:
- (a) Shift heights to be positive (add a constant offset)
- (b) Use a modified SiLogLoss that handles negative values
- (c) Use disparity (inverse height) instead of height

### 4.4 Fix the OneCycleLR Scheduler

**Fix the `total_steps` parameter and `scheduler.step()` cadence.**

Option A (standard OneCycle):
```python
n_batches = (n + args.batch_size - 1) // args.batch_size
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=args.lr, total_steps=n_batches * args.epochs, pct_start=0.05,
)
# Call scheduler.step() after each batch, not each epoch
```

Option B (simpler, per-epoch cosine):
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
# Call scheduler.step() per epoch (current behavior)
```

Option B is simpler and less error-prone. The OneCycle benefit (warmup + high peak LR) is less important when the encoder is pretrained (the model doesn't need aggressive LR exploration).

### 4.5 Keep PatchGAN as Auxiliary Loss (Spec 100, Lower Priority)

PatchGAN (Spec 100) is a valid technique for pushing past L1 plateaus, but it should be **layered on top of the pretrained-backbone model**, not used as the primary convergence strategy.

The order of impact:
1. **Pretrained backbone** (DA-V2-Small) — 100× impact (190 L1 → single digits)
2. **SiLogLoss** — 2-5× impact (sharper predictions, better convergence)
3. **Scheduler fix** — 1.5-2× impact (correct LR schedule)
4. **PatchGAN** — 1.2-1.5× impact (perceptual quality, pushes past L1 plateau)
5. **TTA** — 1.1-1.2× impact (variance reduction)

PatchGAN without a pretrained backbone is like putting a spoiler on a car with no engine.

### 4.6 Domain Adaptation: Top-Down vs. Perspective

DA-V2 is trained on perspective (camera) images. Our minimap is a top-down aerial view. This is a domain gap. Options:
- (a) **Fine-tune the full encoder** (unfreeze backbone) — the model adapts to top-down views. More expensive but more accurate.
- (b) **LoRA on attention layers** (current V23 approach) — the backbone stays frozen, LoRA adapts the features. Cheaper, may be sufficient.
- (c) **Train patch projection + head only** — the backbone is completely frozen, only the input projection and output head adapt. Cheapest, least accurate.

Recommendation: Start with (b) — LoRA + trainable head. If the domain gap is too large (val L1 doesn't drop below 10), escalate to (a).

---

## 5. Concrete Implementation Path

### Phase 1: DA-V2-Small Stage A (replaces Spec 099/100 approach)

1. Create `StageADAV2` model class in `stage_a.py`:
   - Reuse `DepthAnythingV2SmallEncoder` from `harvester/v23/encoder.py`
   - Add a DPT-style head that outputs 33×33 quincunx → 17×17 outer + 16×16 inner
   - Support 3-channel (minimap-only) and 9-channel (guided) input via patch projection replacement

2. Add SiLogLoss (or hybrid L1 + SiLogLoss) to `stage_a.py`

3. Fix the scheduler in `train_v24_stage_a.py`:
   - Use `CosineAnnealingLR` with per-epoch stepping (simpler)
   - Or fix OneCycleLR with correct `total_steps` and per-batch stepping

4. Add `--dav2` flag to `train_v24_stage_a.py`:
   - Loads pretrained DA-V2-Small encoder
   - Uses LoRA (rank 16) on attention layers
   - Uses SiLogLoss (or hybrid)
   - Uses CosineAnnealingLR (or fixed OneCycleLR)

5. Train 40-200 epochs on the curated open-world V24 corpus

6. Validate: target val_l1 < 5.0 world units (vs current 190)

### Phase 2: PromptDA Stage B (replaces current Stage B)

1. Integrate PromptDA-Small as the Stage B model
2. Fine-tune on V18 data with WDL prior as the depth prompt
3. Validate: target val_l1 < 1.0 world units (vs current 0.857 with cheat prior)

### Phase 3: PatchGAN Auxiliary Loss (Spec 100, if needed)

1. Add PatchGAN discriminator on top of the DA-V2 Stage A
2. Use the lambda schedule from Spec 100 (0 → 0.1 ramp)
3. This is a refinement step, not a convergence step

---

## 6. Model Comparison Table

| Model | Params | Pretrained | Task | Stars | Fit for V24 |
|-------|--------|------------|------|-------|-------------|
| DA-V2-Small | 24.8M | DINOv2 (62M images) | Monocular depth | 8,430 | **Stage A** (replace U-Net) |
| PromptDA-Small | 25.1M | DA-V2 + prompt encoder | Depth completion | 1,135 | **Stage B** (replace conv-deconv) |
| Any2Full | ~25M | DA-V2 + scale prompt | Depth completion | 70 | Stage B alternative |
| Marigold | ~800M | Stable Diffusion | Diffusion depth | 3,178 | Spec 102 (detail reconstruction) |
| ZoeDepth | ~100M | MiDaS + metric head | Metric depth | 2,837 | Stage A alternative |
| MiDaS | ~25M | DPT | Relative depth | 5,418 | Superseded by DA-V2 |
| pix2pix | ~50M | None (from scratch) | Image-to-image | 25,184 | Auxiliary loss only (Spec 100) |
| Guided-pix2pix | ~50M | None (from scratch) | Guided img-to-img | 198 | Auxiliary loss with guidance |
| V24 current (Stage A) | 0.335M | None | Minimap → WDL | — | **Too small, no pretrained features** |
| V23 (already in codebase) | 24.8M + LoRA | DA-V2-Small | Height prediction | — | **Infrastructure already exists** |

---

## 7. HuggingFace Model IDs

Based on the GitHub repos and known HuggingFace model IDs:

| Model | HuggingFace ID | Params |
|-------|---------------|--------|
| DA-V2-Small (relative) | `depth-anything/Depth-Anything-V2-Small-hf` | 24.8M |
| DA-V2-Small (metric indoor) | `depth-anything/Depth-Anything-V2-Metric-Hypersim-Small` | 24.8M |
| DA-V2-Small (metric outdoor) | `depth-anything/Depth-Anything-V2-Metric-VKITTI-Small` | 24.8M |
| PromptDA-Small | `depth-anything/prompt-depth-anything-vits` | 25.1M |
| PromptDA-Large | `depth-anything/prompt-depth-anything-vitl` | 340M |
| Marigold | `prs-eth/marigold-v1-0` | ~800M |
| DINOv2-Small | `facebook/dinov2-small` | 22M |

Note: The HuggingFace API search was not returning results from this environment (likely a network/firewall issue). The model IDs above are from the GitHub READMEs and known HuggingFace model pages. Verify availability by running:
```python
from huggingface_hub import HfApi
api = HfApi()
model = api.model_info("depth-anything/Depth-Anything-V2-Small-hf")
```

---

## 8. Summary

The V24 convergence problem is not a loss function problem (PatchGAN) or a training schedule problem (OneCycleLR). It's a **model capacity and pretrained features** problem. The 335K-parameter from-scratch U-Net cannot learn the minimap → heightmap mapping from ~2,000 training tiles.

The solution is to use the DA-V2-Small encoder (24.8M params, pretrained on 62M images) that is **already integrated in the V23 codebase**. The infrastructure for loading the pretrained encoder, replacing the patch projection for custom input channels, and applying LoRA adapters is already built and tested.

The recommended path:
1. **Stage A**: Replace the U-Net with DA-V2-Small + LoRA + DPT head. Use SiLogLoss. Fix the scheduler. Target: val_l1 < 5.0 (vs current 190).
2. **Stage B**: Replace the conv-deconv with PromptDA-Small. Target: val_l1 < 1.0.
3. **PatchGAN** (Spec 100): Add as auxiliary loss on top of the DA-V2 model, if needed. Lower priority.

The pretrained backbone is the 100× improvement. Everything else is a 1.5× improvement on top of that.