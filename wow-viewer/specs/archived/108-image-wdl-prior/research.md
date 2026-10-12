# Research: Image-Only WDL Prior

## Decisions

| Decision | Rationale |
|---|---|
| Target both WDL grids | The client WDL layout is paired: outer `height_257[::16,::16]` and inner `height_257[8::16,8::16]`. A 33×33 substitute is wrong. |
| RGB-only model input | A deployed minimap does not carry height, normals, liquid data, or object masks. |
| Compact representative corpus | Repeated terrain art and lighting contribute little new information. Use the existing curated pattern/prefab selection and hold out complete source groups. |
| Separate prior model | It satisfies the residual chain rule and lets V8 be fine-tuned independently against generated priors later. |
| Row-addressed NPZ archive | It binds a prediction to the source Zarr row and lets the V8 entry point reject missing/misaligned priors. |

## Rejected Alternatives

- Train on all V18 rows: unnecessary repetition and an expensive first experiment.
- Feed ground-truth WDL to V8 under a generated-prior label: invalid deployment proof.
- Predict a 33×33 raster: does not represent the verified WDL outer/inner layout.
