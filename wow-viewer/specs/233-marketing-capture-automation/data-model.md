# Data Model: Renderer Marketing Capture Automation

## FeatureTourRecipe

| Field | Meaning | Validation |
|---|---|---|
| `schemaVersion` | Recipe document contract version | Exact supported version. |
| `id` | Stable machine identifier | Non-empty lowercase slug; unique per catalog. |
| `displayName` | Operator-facing recipe name | Non-empty. |
| `version` | Recipe revision | Non-empty. |
| `requiresWarmPath` | Whether playback must wait for preload | `true` for the initial Flyby workflow. |
| `capture` | Requested rate/container and clean-scene mode | FPS 12–60; tour uses the full-frame capture tap. |
| `beats` | Ordered feature-presentation timeline | Strictly increasing non-negative timestamps; known presentation kind. |

The initial built-in recipe is derived from the loaded camera path at run time; it never stores a client-root path.

## FeatureTourBeat

| Field | Meaning | Validation |
|---|---|
| `atSeconds` | Playback timestamp | Within the camera-path duration; strictly ordered. |
| `kind` | `callout` now; registered-control presentation later | Supported kind only. |
| `featureId` | Stable target identity | Non-empty slug; must resolve for control presentations. |
| `title` / `body` | Human-readable feature explanation | Non-empty title; bounded text. |
| `durationSeconds` | Visible interval | Positive and may not overlap a conflicting beat. |

## FeatureTourAttempt

| Field | Meaning |
|---|---|
| `attemptId` | UTC sortable run identifier. |
| `recipe` | Immutable recipe identity/version. |
| `input` | Map/build and camera-path identity, not client filesystem path. |
| `startedUtc` / `endedUtc` | Actual lifecycle timestamps. |
| `beatResults` | Started, displayed, skipped, or failed result for every beat. |
| `terminalOutcome` | `completed`, `degraded`, `aborted`, `rejected`, or `failed`. |
| `terminalReason` | Typed reason supporting that outcome. |

## TourAttemptReceipt

Written as `<capture filename>.tour-receipt.json` alongside the capture inside the managed output root. It contains the immutable attempt fields plus renderer/application identity, output relative path and encoder terminal result, capture dimensions/FPS/UI mode, frame-history summary/hitches, and receipt schema version/timestamp.

## AuthoringHandoff

| Field | Meaning | Validation |
|---|---|---|
| `schemaVersion` | Handoff contract version | Exact supported version. |
| `attemptId` | Links to receipt | Matches receipt. |
| `captureRelativePath` | Video reference | Relative and within managed output root. |
| `receiptRelativePath` | Evidence reference | Relative and within managed output root. |
| `stillRelativePaths` | Candidate stills, when real captures exist | Every entry relative and in root. |
| `provenance` | Recipe/version/map/build/performance summary | Contains no client-root or machine-local secret. |

The handoff is an artifact descriptor, not a publication request and not a ComfyUI workflow.

