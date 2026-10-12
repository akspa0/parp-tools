# Terrain Method Translation Contract

Schema identity: `v60-terrain-method-translation-v1`

## Required manifest sections

```json
{
  "schema": "v60-terrain-method-translation-v1",
  "method": {},
  "input_contract": {},
  "corpus": {},
  "conditions": [],
  "baselines": [],
  "forbidden_reads": [],
  "metrics": {},
  "decision": "reference"
}
```

## Modality values

- `rgb_only`: runtime-visible RGB plus explicitly predicted auxiliaries.
- `height_prior`: requires an input elevation surface not supplied by the current RGB minimap contract.
- `point_cloud`: requires XYZ/return data.
- `combined`: requires more than one primary modality and is never RGB-only.

## Decision rules

- `reference`: inspected for ideas; no local execution or dependency claim.
- `diagnostic`: executed only against a declared offline source.
- `candidate`: passes contract and baseline gates but is not yet promoted.
- `hold`: evidence is incomplete or a required metric regresses.
- `rejected`: violates modality/provenance or fails a required gate.
- `promoted`: all required evidence and independent metrics pass.

## Forbidden-read audit

The audit must enumerate every input array read by the run and compare it with the declared contract. Any deployment-bound read of `height_257`, `terrain_shadow_256`, raw MCSH/shadow targets, target-side object masks, WDL, or equivalent target-derived data changes the decision to `rejected`.
