# Contract: Asset Load Pipeline

**Date**: 2026-09-01

In-process contract between `AssetLoadPipeline`, the worker pool, and `WorldAssetManager`. Stated as
behaviour so `AssetLoadPipelineTests` can pin it headless.

## C1 — GL belongs to one thread

```
Any GL call from a thread other than the render thread is a defect.
```

Enforced by an assertion active in debug builds (SC-005), not by convention. The whole design rests
on this line; a reviewer must be able to check it mechanically rather than by reading every call
site.

## C2 — A decoded payload holds no GPU handle

```
DecodedAssetPayload contains CPU arrays only — no buffer id, texture id, VAO, or program.
```

This is what makes the payload safe to build on a worker and safe to discard without touching GL
when its request is cancelled (C5). A payload that carried a handle would have to be destroyed on
the render thread, which reintroduces the coupling this spec exists to remove.

## C3 — The three phases are separately timed and separately reported

```
fetch    → worker : bytes for every file the asset needs
decode   → worker : parse, adapt, image decode, CPU mesh build
upload   → render : GPU resource creation only
```

Reported as three numbers, never summed into one (FR-012). The single `DeferredAssetLoads` figure is
exactly what made this defect take a code read to diagnose; collapsing them again would restore
that.

## C4 — The budget governs upload, and only upload

```
DeferredLoadBudget admits uploads.  Fetch and decode are not frame-budgeted.
```

Once upload is the only render-thread cost, the unconditional oversized admission (research R2) must
be **removed**, not merely made rarer. `OversizedAdmissionCount` staying at zero under load is the
proof that the residual it measures has actually been paid off (SC-002).

## C5 — Cancellation never leaks and never half-applies

```
A cancelled request either: never uploads, or uploads completely.
```

No partially-uploaded asset may become visible. A request cancelled after decode but before upload
discards its payload with no GL work; a request cancelled mid-upload completes the upload and is
then evicted normally, because tearing down a half-built GPU resource is more dangerous than
briefly holding a complete one.

## C6 — An asset is invisible until it is fully uploaded

```
resident(asset)  ⟺  upload completed successfully
```

Nothing may submit an asset for rendering on the strength of its decode having finished (FR-005).
This is the invariant that keeps FR-008 true: what is drawn does not change, only when it starts
being drawn.

## C7 — One request per key in flight

```
Two requests for the same key coalesce into one decode.
```

The second requester waits on the first rather than duplicating the work, and a key that failed is
recorded once with the existing suppression semantics (FR-006) rather than retried per frame.

## C8 — Deterministic drain remains available

```
DrainToCompletion() ⟹ every queued request is fetched, decoded and uploaded before returning.
```

Capture and harvest paths need completion, not best-effort streaming (FR-011). This is the same
pipeline run to quiescence, not a second synchronous implementation — C2's split is what makes one
implementation serve both.

## C9 — Off means the old path, exactly

```
Off-thread decode disabled ⟹ fetch, decode and upload run inline on the render thread,
                             in request order, with output identical to the pre-change loader.
```

The toggle (FR-009) bypasses the worker stage; it does not select a different code path. That is
what makes an in-session A/B trustworthy when a visual regression is suspected.
