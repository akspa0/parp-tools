# Phase 0 Research: Minimap Super-Resolution (Real-ESRGAN)

## Decision 1 — the detail render is a new sampling mode inside the existing compositor, not a new renderer

**Finding**: `TerrainMinimapCompositor.BlendLayers` composes MCAL layers per pixel and calls
`TerrainTextureSampler.TrySample(textureId, out color)`, which today returns each texture's
*full-texture average color* (`CalculateAverageColor`, cached per texture id) — deliberately, to
avoid diffuse-repeat moire at 256px (comment at `TerrainMinimapCompositor.cs:205-208`). The average
is position-independent, so the render carries no real texel detail. The per-pixel loop already
knows `sourceX/sourceY` (the 256²/1024² pixel), `chunkX/chunkY`, and the MCAL weights — everything
needed to compute a real texel UV.

**Decision**: Add a detail sampling path — `TrySampleTexel(textureId, u, v, out color)` — that reads
the actual BLP texel at the terrain UV, and a compositor render mode that uses it for the 1024 pass
while the 256 minimap keeps material-average. The UV is the terrain texture coordinate: WoW terrain
diffuse textures tile at a fixed frequency per chunk, so `u,v = frac(tilePosition * repeatsPerTile)`
indexed into the decoded BLP (bilinear). No new renderer, no duplicate reader (constitution II) —
the texels are the same decoded BLPs already passed to the sampler.

**Why moire won't recur (FR-002)**: moire came from sampling one low mip while downsampling hard to
256px. At 1024 the effective sample density is 4× higher and the downsample is gentle; bilinear (or
mip-correct) texel sampling at 1024 is standard terrain rendering, not the pathological case. This
is validated, not assumed — US1 SC-001 measures high-frequency energy and eyeballs a sample for
moire.

**Alternatives considered**: keep material-average and let the SR model hallucinate detail
(rejected — defeats the entire "unique case," the user's explicit reason for the spec); render via
the live GL terrain renderer instead of the compositor (rejected — pulls a GPU/windowing dependency
into the harvester and duplicates the material path the compositor already owns).

## Decision 2 — authored↔detail alignment is measured, with a fixed corrective transform searched before any pairing

**Finding**: The authored minimap comes from `TryLoadMinimapFromMpq` (md5translate → BLP decode),
the detail render from our compositor. Both are per-`(tileX,tileY)` tile images covering the same
world bounds, but pixel-level orientation is unverified and this codebase has real orientation
history: the north/south `TerrainSolarDirection` reversals (Spec 110) and the still-open GLB Y-axis
texture-mirror bug. So a flip/transpose/rotation between authored and render is a live possibility.

**Decision**: US1 ships an alignment analyzer that, for a sample of tiles with both images,
downsamples the detail HR to the authored resolution and computes registration (normalized
cross-correlation / phase correlation) under each of the 8 dihedral transforms (identity, 3
rotations, 4 flips/transposes) plus a small translational search. It reports the best transform and
its residual error per tile and in aggregate. The **gate**: either identity wins within tolerance,
or one single transform wins consistently across all sampled tiles (a fixed correction we then apply
to the render or the pairing). A per-tile-varying "best" transform means the images are not a
consistent SR pair and the spec halts (spec Edge Case + SC-002).

**Alternatives considered**: assume identity alignment (rejected — the whole spec's validity rests
on this and the codebase has burned us on orientation before); learnable/optical-flow registration
(rejected — an SR pair must be a fixed geometric correspondence, not a per-tile warp, or the model
learns to undo a warp instead of to add detail).

## Decision 3a (supersedes 3's arch choice) — ComfyUI-native architecture: RealPLKSR primary, DAT-2 quality ceiling, RRDBNet compatibility floor

**New user constraint (2026-07-18)**: the trained upscaler must load in ComfyUI out of the box.
ComfyUI's `Load Upscale Model` node uses **spandrel**, which auto-detects and loads 30+ supervised
SR architectures from a standard checkpoint — including ESRGAN/RRDBNet, SwinIR, HAT, DAT, ATD,
SPAN, PLKSR/RealPLKSR, DRCT, MoSR. So ComfyUI support is not a framework question, it's an
architecture question: train any spandrel-recognized arch and save its standard state dict, and it
drops into ComfyUI natively with zero custom nodes.

**Decision**: primary architecture becomes **RealPLKSR** — community training consensus (neosr
ecosystem) is that it approaches DAT-2 quality (the heavyweight transformer) at roughly 10× the
speed with far lower VRAM, which fits both our 16 GB local training budget and fast ComfyUI
inference. **DAT-2** is the quality-ceiling option if RealPLKSR's fidelity proves insufficient on
the SC-004/SC-005 gates; **RRDBNet ×4** remains the compatibility floor (largest pretrained zoo, in
case pretrained-init proves valuable). All three are spandrel/ComfyUI-native. The vendoring
approach, PSNR-first-then-optional-GAN staging, real-pair training (Decision 4), and evaluation
contract are unchanged — only the generator arch swaps. The trainer saves checkpoints as standard
spandrel-recognizable state dicts so the deliverable is directly a ComfyUI upscale model.

**Generative upscalers considered and rejected for training** (user asked about SeedVR2 and NVIDIA
RTX VSR): both are prior-driven/generative — they hallucinate plausible detail from natural-image/
video priors, which is the opposite bias from this spec's premise (recovering the REAL lost detail
we can render from source assets; SC-005 explicitly fails fabricated structure). RTX VSR is
additionally a closed driver-level model with no training surface at all — kept as an optional
inference *baseline* in the SC-004 comparison, nothing more. SeedVR2 is a heavy video
diffusion-transformer; LoRA on it is technically conceivable but wrong-biased, wrong-modality
(video), and VRAM-hostile for a 16 GB card.

## Decision 3 (arch choice superseded by 3a) — Real-ESRGAN: compact RRDBNet generator, PSNR-first then optional GAN, dependency only if it stays clean

**Finding**: No SR/ESRGAN code exists in the repo; `torch>=2.5` (CUDA 13.0 wheels) is already a
dependency; `basicsr`/`realesrgan`/`lpips` are not. The official Real-ESRGAN (RRDBNet generator +
U-Net-SN discriminator, L1 + perceptual(VGG) + GAN losses, ×4) is the reference architecture. The
`basicsr`/`realesrgan` packages are usable but historically pin old torch/opencv and carry training
scaffolding we don't need.

**Decision**: Vendor a compact RRDBNet generator (the ESRGAN residual-in-residual dense block, ×4)
as `sr_esrgan_model.py` rather than depending on `basicsr`'s trainer, to keep torch/opencv versions
under our control (constitution I: dependency hygiene). Train in two honest stages, smallest-signal
first (time-to-signal): **(1)** a PSNR/L1-only generator (RealESRNet-style) — proves the data and
pairing produce a working upscaler before any GAN instability; **(2)** an *optional* GAN fine-tune
(add a U-Net-SN discriminator + VGG perceptual + adversarial loss) only if stage 1's outputs are
too smooth, decided after reviewing stage 1. Optionally initialize the generator from public
RealESRGAN ×4 weights if license/shape permit (a contained speed-up, not required). `lpips` is added
as an eval-only metric dependency.

**Alternatives considered**: full `realesrgan` package end-to-end (rejected — heavy, version-pinned,
and it bakes in a synthetic degradation pipeline we are deliberately NOT using since our LR is real
authored data, not degraded HR); a from-scratch non-RRDB CNN (rejected — RRDBNet is the proven SR
generator and the user named Real-ESRGAN specifically).

## Decision 4 — the LR is real authored data, so the degradation model is NOT synthetic

**Finding**: Standard Real-ESRGAN *synthesizes* LR by degrading HR (blur/noise/JPEG/downsample)
because it lacks real LR/HR pairs. We have real LR (authored client minimaps) genuinely paired with
renderable HR — the exact thing that pipeline fakes. Using Real-ESRGAN's synthetic degradation here
would be wrong: it would train the model to invert a degradation our real LR doesn't have.

**Decision**: Train on the real (authored LR, detail HR) pairs directly — no synthetic degradation
pipeline. The "degradation" the model learns to invert is the real authored→detail gap (lower
resolution + the client's own rendering). This is the substantive way our case differs from stock
Real-ESRGAN and is recorded so an implementer doesn't reflexively wire in the synthetic degrader.
(The pure-synthetic fallback in the spec's Edge Cases — degrade detail HR to make LR — is only for
the contingency that authored↔detail alignment fails US1; it changes the deployment story and needs
a separate decision.)

**Alternatives considered**: synthetic degradation of the detail HR as the primary LR (rejected as
above — it discards the real authored input that is the whole point); mixing real and synthetic LR
(deferred — a possible robustness augmentation once the real-pair model works).

## Decision 5 — evaluation: bicubic + material-average baselines, reference metrics where the HR is trustworthy, plus the user visual gate

**Finding**: The HR "ground truth" is our own detail render, not a photographic original, so
reference metrics (PSNR/SSIM/LPIPS of model-output vs detail-HR) measure "did the model reproduce
our render," while the *point* is upscaling the authored LR. Both questions matter and neither
metric alone is sufficient.

**Decision**: Report, on held-out tiles: (1) reference metrics (PSNR/SSIM/LPIPS) of the model's
output against the detail HR — did it learn the target; (2) a high-frequency-content comparison of
model-output vs a bicubic upscale of the authored LR and vs the material-average 1024 — did it add
real detail over naive baselines (SC-004); and (3) the SC-005 user side-by-side gate — is the added
detail genuine terrain structure vs fabricated/restyled. SC-005 is decisive where metrics and the
"add real detail without hallucinating" goal can diverge.

**Alternatives considered**: a single scalar metric (rejected — no single SR metric captures both
"matches target" and "adds real, non-hallucinated detail"); no-reference perceptual metrics only
(kept as secondary; the reference detail HR is available so use it).
