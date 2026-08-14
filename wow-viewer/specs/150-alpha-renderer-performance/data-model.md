# Data Model: Alpha 0.5.3 Renderer Performance Evidence

## NativeRenderEvidence

One behavior observation from the exact 0.5.3 client.

| Field | Type | Rules |
|---|---|---|
| `build` | string | Required; initially `0.5.3.3368` |
| `binaryPath` | string | Validation record only; never a source hardcode |
| `anchor` | string | Function name/address, symbol, xref, or data reference |
| `area` | enum | `world`, `terrain`, `object`, `resource`, `state`, `lod`, `unknown` |
| `observation` | string | Behavior summary, not copied implementation code |
| `confidence` | enum | `proven`, `inferred`, `unknown` |
| `viewerImplication` | string | Candidate constraint or experiment |
| `sourceNote` | string | Workstream/spec note containing the evidence |

Validation: a proposed optimization cannot cite an unanchored `proven` observation.

## RenderControlScene

The fixed comparison identity.

| Field | Type | Rules |
|---|---|---|
| `clientRoot` | runtime path | Configured externally; not persisted in source |
| `build` | string | Must match the client root |
| `map` | string | Required |
| `mapInput` | string | WDT or equivalent configured input |
| `tileOrRoute` | string | Exact tile, camera route, or bounded profile target |
| `camera` | object | Position, target/forward, projection, resolution |
| `residencyPolicy` | object | Detailed/retained/capture-preload policy |
| `warmupFrames` | integer | Non-negative and recorded |
| `measuredFrames` | integer | Positive and recorded |
| `sourceRevision` | string | Git revision or dirty-tree identity |

Two samples are comparable only when their control-scene identity matches or the difference is
explicitly explained.

## RenderPerformanceSample

One frame or aggregate from the production viewer path.

| Field | Type | Rules |
|---|---|---|
| `controlScene` | `RenderControlScene` | Required |
| `frameIndex` | integer | Required for per-frame samples |
| `cpuTotalMs` | number | CPU render-loop time; never labeled GPU |
| `gpuTimeMs` | number/null | Backend timer result, or null with unavailable reason |
| `gpuTimingState` | enum | `available`, `unavailable`, `not-requested`, `failed` |
| `stageDurations` | map | Stable stage names from `WorldRenderFrameStats` |
| `workloadCounts` | map | Visible, culled, submitted, fallback, pending counts |
| `drawStatePressure` | object | Draw calls, uniforms, texture/state binds when available |
| `residency` | object | Loaded, selected, retained, pending terrain/assets |
| `findings` | list | Dominant owner, missing coverage, or unsettled-scene notes |

## OptimizationExperiment

One A/B change and its decision.

| Field | Type | Rules |
|---|---|---|
| `experimentId` | string | Stable human-readable identifier |
| `selectedOwner` | enum | One measured stage or explicit evidence gap |
| `nativeEvidence` | list | Zero or more linked ledger rows; unknown is allowed if labeled |
| `baseline` | sample reference | Same control scene required |
| `candidate` | sample reference | Same control scene required |
| `toggle` | string | Runtime/validation switch or explicit fallback boundary |
| `fallbackReasonCounts` | map | Required when fallback exists |
| `visualGate` | enum | `not-run`, `pass`, `fail`, `user-owned` |
| `decision` | enum | `accepted`, `rejected`, `deferred`, `user-proof-pending` |

Acceptance requires the selected owner to meet the spec threshold, no total p95 regression, and no
unexplained correctness difference.
