# Spec 189 — requirements checklist

Every line is a claim about the data, so every line needs a measurement and a control. A tick means
"measured with a control that could have failed", never "looks right" or "has a name".

## Ledger integrity

- [ ] Every field of PM4 and PD4 appears exactly once with MEASURED / PARTIAL / UNKNOWN.
- [ ] UNKNOWN explicitly includes fields carrying a confident name nobody has tested.
- [ ] Every MEASURED status names the control that could have refuted it.
- [ ] The ledger is emitted from live data, not transcribed, so it cannot drift from the reader.

## MSLK — the chunk under most suspicion

- [x] `_0x04` grouping structure measured — group key with tiny groups (283,066 pairs, 250,192
      singletons); the "near-unique" reading is an artefact of small groups.
- [x] Half-edge / edge-id reading refuted — pairs point at each other 0.08% of the time.
- [x] Paired records share `TypeFlags` 99.82%.
- [x] Pair is NOT geometry-bound-to-anchor — one-of-each is 0.00% over 283,066 pairs.
- [ ] What a PAIR denotes, given both members are always the same kind.
- [x] `_0x00` TypeFlags — a bitfield; bit 0 means "carries no geometry" (0.0% vs 100.0%, no
      exceptions). Bits 1 and 2 never co-occur.
- [ ] What bits 1/2 select between, and what bits 3/4 modify.
- [x] `_0x01` Subtype — NOT a taxonomy. All 19 values statistically identical on geometry; counts
      decay like a counter.
- [ ] What `_0x01` counts. Sequence within a group and hierarchy depth are the candidates.
- [ ] The 34.4% of entries with no adjacency component link.
- [ ] The 1.24% of edges that do not reciprocate — cross-tile neighbours is the obvious candidate and
      is untested.

## MPRL

- [x] Component order measured (height `Y`, horizontal `Z` then `X`).
- [x] Points are terrain contacts.
- [x] Not an object anchor; does not carry `_0x1C`.
- [ ] `Unk04` — index-like per the sweep, yet `Pm4ObjectPositionDecoder` treats it as a heading. Test
      the heading reading; remove the conversion if it fails rather than leaving it running.
- [ ] `Unk00`, `Unk14`, `Unk16`.

## MSUR

- [x] `_0x00 == 0x10` marks underside faces, not a storey index.
- [ ] The remaining `_0x00` values characterised the same way.
- [ ] `_0x10` — plane-distance convention confirmed or replaced.

## Elsewhere

- [ ] `MPRR` under the 4n+3 constraint.
- [ ] `MSHD._0x00` / `._0x08` spans matched to something.
- [ ] `MVER` high byte `0x30` on PM4.
- [ ] `MCRC` on PD4.
- [ ] Whether ground-level `MSUR` floors are a further terrain-height source (would lift the 9.51%
      coverage `MPRL` alone gives).

## Anomalies not yet explained

Recorded so they cannot be quietly dropped.

- [ ] What sets the vertical extent of stretched `_0x1C == 0` surfaces. The over-water reading is
      refuted at cell granularity; a per-vertex version is untested.
- [ ] Whether `MSCN`'s rings are navmesh contours — test contour points against surface BOUNDARIES
      rather than interiors.
- [ ] `AdtPm4MaskBuilder`'s corner-relative coordinate space, never verified.
