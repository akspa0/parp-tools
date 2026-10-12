# Internal Contract — Composition Boundary

Spec 232 introduces no HTTP, REST, GraphQL, or CLI API. The binding internal contract is:

| Producer | Contract | Consumer | Invariant |
|---|---|---|---|
| Layer UI / project JSON | `PhaseLayerSettings` | `PhaseCompositionPolicy` | The policy is the only target-to-donor transform authority. |
| `PhaseCompositionPolicy` | resolved donor tile/chunk + transform kinds | Alpha and Standard adapters | 64x64 confinement and transform order agree for streaming, minimap, live composition and export. |
| `AlphaTileData.RotateQuarterTurn` | full-tile derived data | `ToTileLoadResult` | A shared source lattice edge remains a shared target lattice edge. |
| Terrain manager | layer stack + base gates | future export service | Full output uses the same composition rules as live rendering. |

No producer may silently synthesize a missing source tile or ignore a locked layer/tile.

