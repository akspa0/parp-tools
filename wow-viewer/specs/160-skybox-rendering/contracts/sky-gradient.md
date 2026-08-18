# Contract: Sky Gradient Band Mapping

**Spec**: [../spec.md](../spec.md) | **Satisfies**: FR-007, FR-008, FR-009 | **Research**: R2

Fixes how an ordered band set maps onto the dome, so shader work in Phase 3 starts from a decided
convention rather than inventing one.

---

## G1: The dome is a hemisphere

`SkyDomeVertexBuilder` sweeps `phi` from `0` to `π/2` and writes a per-vertex `heightFactor` of
`ring / rings`. The vertex shader forwards it as `vHeight`.

```text
vHeight = 1.0   zenith        (phi = π/2)
vHeight = 0.0   horizon ring  (phi = 0)
```

There is **no geometry below the horizon**. A band the source authors as below-horizon has nowhere
to land, and must be handled by G4 rather than assumed to render.

---

## G2: Bands are ordered horizon→zenith

`SkyBand.Order` is dense and ascending from 0 at the horizon. The canonical five-band set, per the
LIT colour-track table (indices 2-6):

| Order | Semantic | Track index |
|---|---|---|
| 0 | Sky Horizon | 6 |
| 1 | Sky Lower | 5 |
| 2 | Sky Middle | 4 |
| 3 | Sky Upper | 3 |
| 4 | Sky Top (zenith) | 2 |

Note the **order is the reverse** of the track index. Track 2 is the zenith and track 6 is the
horizon, so a direct index-to-order copy inverts the sky. This is the most likely single mistake in
Phase 3; the unit test in Phase 3 step 6 exists specifically to catch it.

---

## G3: Interpolation

Colour at a fragment is interpolated between the two bands bracketing its `vHeight`:

```text
find i such that Bands[i].HeightFactor <= vHeight <= Bands[i+1].HeightFactor
t     = (vHeight - Bands[i].HeightFactor) / (Bands[i+1].HeightFactor - Bands[i].HeightFactor)
color = mix(Bands[i].Color, Bands[i+1].Color, t)
```

- Below the lowest band's `HeightFactor`: clamp to `Bands[0].Color`, then apply G5.
- Above the highest: clamp to `Bands[last].Color`.
- Interpolation is continuous at every band boundary — `t` reaching 1 at band `i` equals `t` at 0 for
  band `i+1`. This is what FR-009's no-seam requirement means concretely.

**Default `HeightFactor` placement**: bands are distributed evenly across `[0, 1]` unless the source
authors explicit heights. For the five-band set that is `0.0, 0.25, 0.5, 0.75, 1.0`. Even spacing is
a **stated default, not a client-verified fact** — if real client comparison shows different band
placement, this contract is what gets corrected, and the change is recorded here.

---

## G4: Below-horizon bands

Bands whose authored height falls below the horizon ring are **not** discarded silently. They are:

1. clamped to `HeightFactor = 0.0`, and
2. counted in the shortfall report of G6.

Rationale: the dome cannot show them, but the data authored them. Discarding without reporting would
make a real difference between source and screen invisible — the exact class of failure this spec
exists to fix.

---

## G5: The existing below-horizon fog blend is preserved

The current shader blends toward fog colour under `vHeight < 0.15`:

```text
fogBlend  = smoothstep(0.15, 0.0, vHeight)
skyColor  = mix(skyColor, uFogColor, fogBlend)
```

This behaviour is **retained** and applies *after* band interpolation. Band mapping must be defined
against it rather than replacing it — Band 0 is the sky colour at the horizon, not the final
rendered horizon colour, which is Band 0 blended toward fog.

**Constitution guard**: fog colour is already shared with terrain. Changing this blend is out of
scope, and Phase 6 re-checks terrain fog on both Alpha-era and LK 3.3.5 terrain.

---

## G6: Short band sets

A source authoring fewer bands than the full set uses the bands it has. It is **never** padded to a
fixed length (FR-008).

- `AuthoredBandCount` records what was authored.
- `AuthoredBandCount < ExpectedBandCount` is reported as a shortfall.
- Interpolation proceeds over the shorter ordered set by G3; a two-band set reduces exactly to the
  current two-colour behaviour, which is what makes this change safe to land incrementally.

**Assertion**: a two-band set renders identically to the pre-change gradient. This is the Phase 3
regression guard.

---

## G7: Geometry is unchanged

`SkyDomeVertexBuilder` is **not** modified (research R2). No new vertex attributes, no new buffers,
no ring-per-band rebuild. Band count is a uniform-side concern so it can vary per map and per source
without touching the mesh.
