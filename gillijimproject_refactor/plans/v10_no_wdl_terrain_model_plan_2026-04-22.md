# V10 No-WDL Terrain Model Plan

## Recommendation

- treat the next real branch as `v10`, not `v9.5`
- reserve `v9.5` only for one short ablation lane that measures how much of the current failure is caused by WDL dependence versus late-stage schedule drag
- make `generate without WDL` a first-class training requirement instead of an inference-time fallback

## Why V10 Instead Of V9.5

- the strongest current problem is not only schedule saturation; it is a broken training contract
- `v9` is trained as a residual-correction model whose base prior is assumed to be valid
- when `wdl_17` is missing, the trainer currently substitutes `height_17` as the base prior, which means the model never truly learns the no-WDL case during supervised training
- that makes the new direct-PNG no-WDL or optional-WDL inference path fundamentally out-of-distribution for the trained model
- small loss-weight changes or another 100 epochs will not fix that mismatch cleanly

## Evidence From The Current V9 Stack

- the active input contract explicitly includes `wdl_17_or_height_17_base_prior` as one of the named signals
- `V9NativeDatasetOptimized._precompute(...)` uses `wdl_17` when present and otherwise falls back to `height_17.clone()` as `base_17`
- prediction reconstruction in `build_predictions(...)` always adds learned deltas on top of a provided base height at `17`, `65`, and `257`
- dev-eval currently measures performance on WDL-backed entries and can select checkpoints using `dev_wdl_mae_improvement`, which keeps model selection coupled to WDL-relative behavior
- the saved `v9_at_epoch_691` summary does not show a magical hidden continuation: it reports `final_epoch = 415`, `best_epoch = 411`, `best_val_loss = 0.003267933180856351`, `stop_reason = completed_requested_epochs`, and `final_learning_rate = 3.75e-05`

## Current Failure Mode

### What V9 Is Good At

- taking a coarse base surface and learning bounded corrections on top of it
- using minimap and mask signals to refine local detail where the base prior is already structurally close
- producing stable outputs under the current residual-heavy supervision contract

### What V9 Is Bad At

- synthesizing a globally correct coarse terrain shape when no trustworthy WDL prior exists
- separating what the minimap uniquely implies from what the WDL prior quietly injected
- learning a graceful fallback path for tiles that have minimap evidence but no low-resolution terrain prior

### Practical Consequence

- when no WDL is available, `v9` is being asked to do a task it was not really trained for
- that is why “WDL as a helpful prior” turned into “WDL is effectively required for the model to stay on-distribution”

## V10 Core Requirements

- no-WDL generation must be a normal training case, not an exceptional one
- WDL must become an optional conditioning signal whose absence is expected and exercised every epoch
- the model must learn coarse terrain structure from visual and semantic signals alone before any optional WDL fusion is applied
- model selection must include a no-WDL benchmark and must not be anchored only to WDL-relative improvement

## V10 Architecture Direction

### High-Level Shape

- keep one deployable model graph
- split the model into bounded expert branches and optional heads inside that graph instead of many fully separate models
- prefer one training lane and one checkpoint family, with curriculum and loss scheduling deciding when different capabilities matter
- use three explicit stages:
  - visual coarse terrain branch
  - optional prior fusion branch
  - high-resolution refinement branch

### Stage 1: Visual Coarse Terrain Branch

- inputs:
  - minimap RGB
  - normal RGB if available
  - luma and gradient maps
  - static context masks such as liquid, object, PM4, brush, holes, and height-range hints if they remain predictive
- output:
  - an absolute coarse terrain estimate at `17x17`
  - optionally a `65x65` mid surface or a learned latent from which `65x65` is decoded
- key point:
  - this branch must not depend on WDL presence to define the global terrain shape

### Stage 2: Optional Prior Fusion Branch

- inputs:
  - coarse prediction from stage 1
  - optional WDL prior when present
  - a presence flag and optional corruption score channel
- outputs:
  - refined coarse or mid surface
  - prior confidence gate or fusion weight map
- behavior:
  - if WDL is absent, this branch should reduce to identity or low-impact correction
  - if WDL is present and trustworthy, it may improve the coarse terrain estimate
  - if WDL is present but bad, the gate should learn to suppress it

### Stage 3: Detail Refinement Branch

- inputs:
  - fused coarse or mid terrain
  - high-resolution visual signals
  - object and liquid masks
  - optional local semantic priors
- output:
  - final `257x257` height field or residual over the fused mid surface
- focus:
  - recover chunk-scale and sub-chunk terrain detail without taking responsibility for the whole global shape

### Stage 4: Optional Detail Refiner Lane

- treat this as an explicit later capability inside the same model, not as a separate model that sits on top of the base output
- inputs:
  - stage-3 terrain prediction
  - minimap detail crops
  - local masks for liquid, object, PM4, brush, and holes
  - optional learned latent from the base model
- outputs:
  - local residual correction at `257x257`
  - optional confidence or "needs refinement" map
- focus:
  - ridge sharpening
  - cliff cleanup
  - local continuity repair
  - high-frequency residual cleanup where the base model gets broad structure right but misses local shape
- implementation preference:
  - add this as an internal patch-aware head or auxiliary branch that is trained jointly or activated later in the same training program
  - do not require a second long-running top-level training job unless the integrated path clearly fails

### Stage 5: Optional Local Realism Feedback

- this is where a bounded PatchGAN-style discriminator belongs if we add it at all
- the discriminator should critique local terrain realism over rendered or derived local patches
- it should never become the first teacher for global terrain structure
- this still fits the "single big machine" approach if the discriminator is treated as an auxiliary training critic rather than a second deployable terrain model

## Training Contract Changes

### Remove The Fake No-WDL Path

- stop using `height_17` as the substituted base prior for no-WDL training examples
- keep `height_17` only as supervision, never as the hidden replacement conditioning signal when WDL is absent

### Add Prior-Dropout As A Default Mechanism

- every training epoch should include all three modes:
  - `no_wdl`: no prior at all
  - `real_wdl`: true WDL input
  - `corrupted_wdl`: noisy, shifted, flattened, or partially blanked WDL input
- first recommended mix:
  - `50%` no WDL
  - `30%` real WDL
  - `20%` corrupted WDL
- add an explicit `wdl_present` or `prior_mode` channel so the model is not forced to infer absence indirectly from zeros alone

### Reframe The Targets

- stage 1 should predict absolute coarse terrain, not only residuals against a base prior
- stage 2 may predict a correction relative to stage 1 and optional prior fusion
- stage 3 may remain residual-based, but only after the coarse shape is already produced by the model itself

### Loss Structure

- keep multi-scale supervision, but change what each stage is responsible for:
  - coarse absolute loss at `17x17`
  - mid absolute loss at `65x65`
  - full-resolution `257x257` L1 or Charbonnier loss
  - gradient loss for slope and ridge continuity
  - optional edge or normal-consistency loss if it improves detail without flattening
- reduce the current emphasis on “residual over supplied base” as the only geometry story
- add a fusion regularizer so the optional prior branch does not dominate when `wdl_present = 0`

### Add Explicit Small-Win Objectives

- do not ask the full model to win only through one monolithic best-val-loss curve
- track and reward smaller structural wins that can improve even when the top-line metric moves slowly:
  - coarse `17x17` MAE improvement
  - mid `65x65` MAE improvement
  - no-WDL gradient error improvement
  - local-detail residual error improvement on high-detail tiles
  - ridge or cliff tile subset improvement
  - liquid-edge subset improvement
  - masked object-footprint subset improvement
- use these as monitored diagnostics first and as secondary checkpoint tie-breakers later if needed

### Add Explicit Training Buckets

- stop treating the curated pool as one undifferentiated stream once the base contract is stable
- assign each sample to one or more measurable buckets that match real failure modes:
  - flat or low-detail terrain
  - steep or cliff-heavy terrain
  - ridge or ravine terrain
  - liquid-edge terrain
  - high-object-footprint terrain
  - minimap-ambiguous terrain
  - WDL-present terrain
  - WDL-missing terrain
  - corrupted-WDL training mode
  - low-resolution-source-derived terrain from historical world images
- use these buckets for:
  - subset validation
  - sampler weighting
  - best-triggered focus phases
  - deciding when a new head or auxiliary loss should activate

### Add Alternate Input Views

- the model should not only see the exact final inference presentation during training
- add controlled alternate views that preserve terrain semantics while broadening robustness:
  - lower-resolution minimap variants upsampled back to model input size
  - blurred or softened minimap variants to simulate older stitched or cut images
  - contrast-shifted or luma-compressed variants for weak archival inputs
  - bounded crop-context variants when a patch-aware head is active
- do not add arbitrary image augmentation that destroys geography cues
- the guiding rule is: teach robustness to expected input families, not generic vision noise

## Schedule Changes

- replace the purely reactive late-stage plateau mindset with an explicit curriculum

### Phase A: No-WDL Bootstrap

- train only the visual coarse branch and the basic detail branch
- disable real WDL conditioning for the first warmup window
- goal:
  - force the model to learn a usable terrain prior from minimap and masks alone

### Phase B: Mixed Optional Prior Training

- enable real, absent, and corrupted WDL modes together
- train the fusion branch and its gating behavior
- measure whether WDL improves outputs without becoming required

### Phase C: Hard-Case Detail Focus

- continue hard replay and detail-focus sampling
- bias toward tiles with:
  - steep height ranges
  - strong object footprints
  - liquid boundaries
  - minimap ambiguity
  - weak or misleading WDL

### Phase C1: Best-Triggered Focus Shift

- when the run hits a real new best on the main no-WDL metric, do not immediately change the whole architecture
- instead, switch one bounded lever for the next window of epochs:
  - increase weight on hard-detail buckets
  - add low-resolution-source-derived buckets if they are underrepresented
  - enable alternate-view variants for a bounded fraction of samples
  - increase detail-head loss weights only on the targeted subsets
- treat this as a scheduled exploitation phase after a proven baseline gain, not as random trainer mutation

### Phase D: Local Refiner Warm Start

- once the base v10 lane is structurally sane, activate the integrated local-refinement head on patches cut from its own predictions
- prefer partial freezing or reduced learning-rate updates for early layers before considering a separate refiner program
- goal:
  - get reliable local improvement wins without destabilizing the base coarse solver

### Phase E: Bounded Adversarial Feedback

- only after deterministic detail refinement is stable, add a lightweight PatchGAN-style discriminator over local terrain patches
- adversarial pressure should target local realism and shape plausibility, not replace supervised geometry losses
- recommended first PatchGAN target surfaces:
  - local height residual patches
  - derived slope or normal patches
  - optionally rendered terrain-only local minimap patches if the forward rendering bridge is stable
- keep GAN off the critical path for early convergence

### Optimizer And Scheduler

- keep `AdamW`
- prefer an explicit warmup plus cosine decay or cosine restarts for the main v10 run instead of relying only on `ReduceLROnPlateau`
- keep plateau reduction only if it still helps after the curriculum is stable
- do not use checkpoint selection rules that prefer WDL-relative gains over no-WDL reliability

### Hurdle Response Policy

- when training stalls, do not immediately spend another `100` epochs on the same contract
- react by switching to the smallest next lever that can create a measurable local win:
  1. if coarse shape is wrong, work on stage 1 and no-WDL curriculum rather than detail loss weights
  2. if coarse shape is right but local detail is weak, increase detail-focus sampling or activate the refiner lane
  3. if deterministic detail metrics improve but outputs still look too smooth locally, test bounded PatchGAN feedback
  4. if WDL presence still changes behavior too sharply, increase no-WDL and corrupted-WDL exposure before touching the scheduler
- each hurdle response should answer one narrow question instead of changing the whole training stack at once

### Suggested Trigger Rules For Small Interventions

- if no-WDL coarse `17x17` MAE stalls for `N` epochs:
  - increase no-WDL sampling share or extend bootstrap phase
- if no-WDL full `257x257` MAE stalls but coarse improves:
  - activate or up-weight detail refiner training
- if local detail metrics improve while realism still lags:
  - enable a low-weight PatchGAN burst for a bounded window
- if GAN improves realism but worsens metadata or MAE:
  - disable GAN and treat it as a failed branch, not a mandatory permanent component

### Suggested Trigger Rules For Best-Triggered Phase Shifts

- if the model sets a new best no-WDL validation score and coarse metrics improved more than detail metrics:
  - increase sampling from steep, ridge, liquid-edge, and object-heavy buckets for the next phase window
- if the model sets a new best but low-resolution-source-derived buckets remain weak:
  - enable low-resolution alternate-view augmentation for a bounded fraction of the training stream
- if the model sets a new best and detail subsets also improved:
  - keep the sampler stable and only raise detail-head emphasis modestly
- if a best is driven only by real-WDL gains while no-WDL buckets stay flat:
  - do not treat that as a curriculum trigger; instead increase no-WDL and corrupted-WDL exposure

## PatchGAN Recommendation

- yes, PatchGAN can be useful here, but only as a local feedback loop after the supervised model has learned sane geometry
- the old `v7` lane already proved a lightweight patch discriminator path is implementable in this repo
- the correct role for PatchGAN in `v10` is:
  - late
  - local
  - low-weight
  - explicitly monitored
  - easy to disable
- the incorrect role is:
  - teaching global terrain structure from the start
  - masking a broken no-WDL training contract
  - becoming the main reason the model appears to improve
- if we use it, it should attach to the same primary `v10` training run as an auxiliary critic, not create a second deployable terrain generator stack

### PatchGAN Guardrails

- do not start adversarial training in phase A or early phase B
- keep reconstruction, gradient, and absolute geometry losses dominant
- use scheduled GAN bursts or cooldown-style windows instead of always-on adversarial pressure at first
- monitor at least these values separately:
  - no-WDL MAE
  - no-WDL gradient error
  - detail subset MAE
  - GAN discriminator loss
  - GAN generator loss
  - whether the GAN changed checkpoint ranking or only visuals
- if GAN does not create a measurable local-detail win, remove it quickly

## Small Wins Strategy

- the goal is not to force improvement every epoch; it is to avoid waiting huge blocks of time before learning whether the model changed in a useful way
- define a ladder of expected wins:
  1. base no-WDL coarse terrain becomes stable
  2. optional WDL fusion adds lift without changing the base behavior too much
  3. integrated detail-head metrics improve on hard subsets
  4. patch-aware refinement inside the same model improves local quality on already-good tiles
  5. PatchGAN, if used, improves local realism without hurting geometry
- this gives multiple places to make progress without pretending the whole stack should improve uniformly at once

## Pilot Scale Recommendation

- start with a small but still diverse audited subset around `800` tiles
- the point of the `800`-tile pilot is not to maximize final quality; it is to answer whether the integrated architecture and loss contract move in the right direction quickly
- recommended pilot goals:
  - prove the no-WDL branch learns usable coarse terrain
  - prove optional WDL improves rather than dominates
  - prove integrated detail learning improves local subsets without needing a second model
- once those are true, scale the same model and same training contract upward rather than redesigning the stack again

### Pilot Composition Guidance

- the first `~800` tile run should be deliberately mixed rather than purely random
- minimum intent for the pilot split:
  - a stable no-WDL-heavy backbone set
  - a useful slice of WDL-present examples for optional-prior learning
  - explicit hard-detail buckets
  - a small but non-trivial low-resolution-source-derived bucket for future continent-image handling
- the pilot does not need perfect balance, but it should not accidentally exclude the hard cases we care about later

## Validation Requirements

- define three standing evaluation tracks for every serious run:
  - no-WDL dev eval
  - real-WDL dev eval
  - corrupted-WDL dev eval
- require the main selection metric to be a blended score rather than only `val_loss` on WDL-heavy tiles
- recommended first blended metric:
  - primary: no-WDL global MAE
  - secondary: no-WDL gradient error
  - tertiary: real-WDL improvement over stage-1-only output
- do not allow a model to win solely because it beats WDL on tiles that already have WDL

### Add Subset Validation Buckets

- keep a few fixed validation subsets that answer specific questions fast:
  - steep terrain subset
  - cliff or ridge subset
  - liquid-edge subset
  - high-object-footprint subset
  - low-WDL-confidence subset
- use those to decide when to escalate from base training to refiner work or GAN work

## What To Keep From V9

- the native tensor-cache contract is still the right base direction
- multi-scale supervision at `17`, `65`, and `257` is still correct
- hard replay and detail-focus sampling are worth keeping
- mask and semantic side channels remain useful, but they need to serve a model that can stand without WDL
- the earlier bounded GAN experiment path is still useful as an implementation reference, not as the default training mode
- the optimized-trainer operational surface is the correct baseline for `v10`, not the older bare trainer path

## What To Drop Or Downgrade From V9

- drop the hidden fallback `base_17 = height_17` training path for no-WDL samples
- downgrade WDL from mandatory geometry anchor to optional conditioning input
- stop using WDL-centered dev selection as the main proof of progress
- stop interpreting continued late-stage training as evidence that the architecture is still the main bottleneck-free path

## V9.5 Bounded Ablation Lane

- only do this if a cheap proof is needed before `v10` implementation starts
- bounded goals:
  - add prior-dropout to `v9`
  - remove `height_17` fallback as the fake no-WDL base during training
  - add no-WDL dev eval and compare against current best
- if this gives a meaningful jump, keep the result as baseline evidence for `v10`
- do not treat this as the final architecture lane

## V10 Build Order

1. add explicit prior-mode support to the cache or trainer contract so samples can be trained with real, absent, and corrupted WDL
2. create a new `train_v10.py` branch rather than continuing to accrete special cases inside `train_v9_optimized.py`
3. implement a coarse absolute terrain branch that does not use WDL as its base
4. add an optional WDL fusion branch with a learned gate
5. add explicit subset metrics so small structural wins are visible before top-line convergence
6. run an `~800` tile pilot on the current development holdout or audited sane pool before scaling to full data
7. reuse the existing detail branch ideas only after the coarse branch is proven on no-WDL data
8. activate integrated patch-aware refinement for local cleanup once deterministic detail learning is stable
9. add blended dev eval that reports separate metrics for no-WDL, real-WDL, and corrupted-WDL modes plus hard-subset buckets
10. only then test a bounded PatchGAN local-feedback lane inside the same training job

## Optimized Trainer Parity Checklist

- `v10` should inherit the operational floor of `train_v9_optimized.py`, not only its broad modeling ideas
- the priority carryovers are:
  - explicit CUDA backend setup for TF32 and cuDNN benchmark on supported hardware
  - `persistent_workers` support instead of forcing `False`
  - per-epoch timing breakdown for load, host-to-device, forward, backward, validation, and dev-eval time
  - resume and last-checkpoint behavior that is already present in the optimized trainer family
  - hard-replay and detail-focus sampler continuity
  - clear run-summary capture of compile state, timing, and phase changes
  - optional pause or review hooks when a stall threshold is crossed
- `v10` does not need to copy every old option blindly, but it should match the throughput, observability, and recovery features that made `train_v9_optimized.py` the real baseline

## Success Criteria

- no-WDL outputs are structurally plausible and no longer collapse into the “missing prior” failure mode
- real WDL still improves results when available, but the model does not depend on it to stay on-distribution
- checkpoint selection prefers the model that best solves the actual recovery task, not the one that best exploits WDL presence
- the first trustworthy statement after `v10` should be:
  - the model can generate useful terrain from minimap-only inputs
  - WDL, when present, acts as an optional quality boost rather than a hidden requirement