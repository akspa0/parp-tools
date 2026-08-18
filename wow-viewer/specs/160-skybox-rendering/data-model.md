# Phase 1 Data Model: Skybox Rendering

**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Date**: 2026-08-18

Entities below are runtime resolution results, not persisted records. Nothing here is written to
disk; all of it is derived per map load or per frame from client data already parsed by existing
readers.

The central shape is that **gradient and model resolve independently** (research R1). There is no
combined "sky profile" entity, deliberately — see the rationale under `SkySourceSelection`.

---

## SkyProvenance

Attached to every resolved sky value. Satisfies FR-003 and SC-004.

| Field | Type | Notes |
|---|---|---|
| `SourceKind` | enum | `MapScopedLit`, `LightDbcChain`, `WmoDeclaration`, `DiscoveryFallback`, `HardcodedFallback` |
| `SourceIdentity` | string | The file or table the value came from, e.g. the `.lit` path or `LightSkybox` |
| `RecordIdentity` | string? | The row, track, or chunk within that source; null when the source has no sub-record |
| `IsAuthored` | bool | `true` for client-authored data, `false` for any fallback |
| `BuildIdentity` | string | The client build the value resolved against |

**Rules**

- Every rendered sky value carries exactly one `SkyProvenance`. There is no unprovenanced path.
- `IsAuthored == false` is a first-class reportable state, not an error. FR-005 requires the fallback
  to say it is a fallback.
- `SourceKind.HardcodedFallback` is the only kind permitted to originate inside the viewer. All
  others name client data.

---

## SkyBand

One authored colour at a defined height in the vertical gradient.

| Field | Type | Notes |
|---|---|---|
| `Order` | int | Position in the horizon→zenith sequence; 0 is nearest the horizon |
| `Color` | Vector3 | Linear RGB |
| `HeightFactor` | float | Where this band sits on the dome, in the same 0→1 space as the dome's `vHeight` |

**Rules**

- Bands are held in a strictly ordered set; `Order` is dense and ascending.
- `HeightFactor` maps against the **hemisphere** convention — the dome has no geometry below the
  horizon (research R2). The mapping is fixed in [contracts/sky-gradient.md](./contracts/sky-gradient.md).
- A source authoring fewer bands yields a shorter set. It is **never** zero-filled to a fixed length
  (FR-008); the shortfall is reported instead.

---

## SkyGradientSource

The resolved gradient for the current map and time. One of the two independent resolution results.

| Field | Type | Notes |
|---|---|---|
| `Bands` | ordered `SkyBand[]` | May be shorter than the full set; never padded |
| `FogColor` | Vector3 | Already shared with terrain fog today; preserved, not redefined |
| `Provenance` | `SkyProvenance` | Per FR-003 |
| `AuthoredBandCount` | int | What the source actually authored |
| `ExpectedBandCount` | int? | What the source *kind* would normally author; null when unknown |

**Rules**

- `AuthoredBandCount < ExpectedBandCount` is the reportable shortfall state of FR-008.
- Bands are evaluated against the world time-of-day clock using the selected source's timed samples
  (FR-006).
- All bands in one `SkyGradientSource` come from **one** source. Never assembled across sources
  (FR-002).

---

## SkyModelReference

A named client asset to be drawn as sky, with the reason it was chosen. The second independent
resolution result.

| Field | Type | Notes |
|---|---|---|
| `AssetPath` | string | Resolved path within the configured data source |
| `Provenance` | `SkyProvenance` | Which declaration named it |
| `SelectionReason` | enum | `OutdoorProfile`, `WmoInterior`, `DiscoveryFallback` |
| `LoadState` | enum | `Resolved`, `Loading`, `Unresolvable` |

**Rules**

- On LIT-era builds there is **no** model declaration (research R1). `DiscoveryFallback` is the
  expected normal state there, not a degraded one.
- `Loading` and `Unresolvable` both render gradient-only (FR-014). Neither blocks the render thread
  (FR-024).
- `Unresolvable` is reported **once**, not per frame (FR-014).
- When candidates tie, selection is deterministic so it cannot oscillate between frames (FR-013).

---

## SkySourceSelection

The single resolved sky in effect for the current frame — the join point, and the entity that
enforces FR-002.

| Field | Type | Notes |
|---|---|---|
| `Gradient` | `SkyGradientSource` | Resolved independently |
| `Model` | `SkyModelReference?` | Resolved independently; null is valid and renders gradient-only |
| `IsManualOverride` | bool | True when the user's LIT override forced the gradient source (research R7) |

**Rules**

- **The invariant**: `Gradient.Provenance` and `Model.Provenance` may legitimately differ — that is
  the LIT-era case, an authored LIT gradient beside a discovered model. What is forbidden is a
  *single* result assembled from two sources. FR-002 constrains composition **within** each result,
  not agreement **between** them.
- This is precisely why there is no combined profile entity. One record with both a band set and a
  model field would be null-model on every alpha build, creating exactly the pressure to fill that
  null from the other source that FR-002 exists to prevent (research R1).
- `IsManualOverride` is recorded in provenance so an override is never mistaken for a resolution.

---

## State transitions

### Model load

```text
Unresolvable ◄── name resolves to no asset ──┐
                                             │
(declared name) ──► Loading ──► Resolved ────┘
                       │
                       └─► gradient-only render while loading (never blocks)
```

`Unresolvable` is terminal for that name and is reported once. Re-entering a map or resolving a
different name starts a new transition.

### Interior/exterior swap (US4)

```text
Outdoor ──camera enters WMO declaring a skybox──► Interior
Interior ──camera exits──► Outdoor
Interior ──declared name unresolvable──► Outdoor (reported, FR-018)
```

Crossing must be stable under repeated transitions and under nested or overlapping WMOs
(FR-017, SC-005). Hysteresis or an equivalent guard lives at the transition, not in the renderer.

---

## Relationships

```text
SkySourceSelection
├── Gradient : SkyGradientSource ──► Bands : SkyBand[]  (ordered, horizon→zenith)
│                                └─► Provenance : SkyProvenance
└── Model    : SkyModelReference? ──► Provenance : SkyProvenance
```

Every leaf reaching the renderer terminates in a `SkyProvenance`. That is the structural guarantee
behind SC-004's "100% of rendered sky values report their source".

---

## Ownership

Per Constitution II, all five entities are **library** types under
`src/core/WowViewer.Core.Runtime/World/Sky/`. The viewer consumes them for wiring, draw order, and
display; it does not construct or decide them. `WorldScene.cs` gains no resolution logic.
