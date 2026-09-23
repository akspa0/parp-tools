# Research: Native Day/Night Direction Recovery

## Decision

The 0.5.3.3368 outdoor world-light direction is a timed native-client computation, not a LIT/DBC field and not the fixed 45-degree shadow-projection constant.

## Exact evidence

The matching `Wowae.pdb` exposed these live functions in staged `WoWClient.exe` 0.5.3.3368:

| Symbol | Address | Finding |
|---|---:|---|
| `DayNightUpdateLighting` | `0x006bd6c0` | derives time state and calls color/direction paths separately |
| `SetColors` | `0x006bb5d0` | applies the timed color state |
| `SetDirection` | `0x006bca40` | interpolates phi/theta and writes the native light ray |

`SetDirection` writes:

```text
ray = (sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi))
```

The theta table is four samples of `3.926991` radians (225 degrees). The phi table alternates `2.216568` (127 degrees) and `1.919862` (110 degrees) across normalized quarter-day markers. At normalized time `0.6976439`, process memory held `(-0.6481626, -0.6481628, -0.3997127)`, matching interpolation and the formula.

The native vector is a downward light ray. A renderer storing a vector toward the source inverts it: constant source azimuth is 45 degrees, while source elevation varies from approximately 20 to 37 degrees. The `cos(pi/4)` constant in `FUN_006cbd50` belongs to unit/dynamic shadow projection only.

## Implications

1. LIT/DBC remain authoritative for their build-scoped timed color/fog/sky records; they do not provide a sun-direction record.
2. Exact direction must be modeled separately per client build.
3. A single native/viewer image comparison proves only the coordinate/sign transform between native and viewer terrain axes. It does not discover an unknown azimuth.
4. The current authored rotating direction is an explicit fallback and must not be reused by a profile labeled client-exact.

## Rejected approaches

| Rejected approach | Reason |
|---|---|
| Search LIT/DBC for a sun vector | direction is computed in `SetDirection`, not stored there |
| Use fixed 45 degrees as world sun elevation | that constant belongs to the distinct dynamic shadow projection path |
| Fit a direction independently for every capture time | violates the recovered single native model and hides coordinate-transform errors |
| Treat an image-only fit as direction evidence | it can validate viewer-axis mapping but cannot replace client function/table evidence |

## Remaining evidence gap

Only the native-vector to viewer-terrain coordinate/sign transform needs a controlled native/viewer comparison. MCSH attenuation and sky-band altitude placement are separate open measurements.
