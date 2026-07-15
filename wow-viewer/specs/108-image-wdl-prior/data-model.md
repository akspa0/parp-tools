# Data Model

## WdlPriorTarget

| Field | Shape | Source |
|---|---:|---|
| `outer_17` | 17×17 float32 | `height_257[::16, ::16]` |
| `inner_16` | 16×16 float32 | `height_257[8::16, 8::16]` |
| `values` | 545 float32 | normalized outer followed by normalized inner |

## PredictionArchive

| Field | Shape | Meaning |
|---|---:|---|
| `rows` | N int64 | source Zarr row addresses |
| `outer_17` | N×17×17 float32 | generated world-unit WDL outer grid |
| `inner_16` | N×16×16 float32 | generated world-unit WDL inner grid |
| `metadata_json` | scalar string | store/checkpoint/normalization contract |

Rows must be unique and are the only V8 handoff key.
