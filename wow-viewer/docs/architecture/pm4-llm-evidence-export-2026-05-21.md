# PM4 LLM Evidence Export

## Scope

This note records the current PM4 evidence-export surface after the `MSHD.Field04`
region-id seam was promoted into `wow-viewer` and consumed by the legacy
`MdxViewer` compatibility workbench.

The goal is narrow: make the currently visible PM4 overlay easier to inspect and
export in a form that both humans and vision-capable LLMs can read without
starting from raw geometry dumps.

## Current Behavior

- `wow-viewer` remains the owner of the PM4 `MSHD.Field04` grouping contract.
- `MdxViewer` uses that seam as a compatibility consumer only.
- The PM4 workbench selection view now exposes a **Selected MSHD Region** panel
  built from the same visible overlay objects that are currently rendered.
- That panel shows:
  - visible object count in the selected region
  - visible tile count
  - same-CK24 / same-MSLK / same-MDOS counts relative to the selected object
  - visible type mix
  - peer rows with select, frame, and collect actions

## Export Bundle

`MdxViewer` now exports a PM4 LLM evidence bundle from the current visible
overlay set.

Output bundle contents:

- `pm4_llm_bundle.json`
  - compact machine-readable summary of the visible overlay, top regions,
    current legend mode, selected object, and selected-region peers
- `pm4_llm_bundle.md`
  - concise narrative summary intended for human review or direct LLM ingestion
- `pm4_visible_regions.svg`
  - bar-chart style infographic of the top visible MSHD regions
- `pm4_selected_region.svg`
  - selected-region peer sheet when a PM4 object is selected

## Boundaries

- This export is a **viewer evidence** surface, not a new PM4 decode layer.
- It does not change PM4 placement math, object splitting semantics, or match
  scoring.
- `MSHD.Field04` is still treated as a research-driven grouping aid, not closed
  proof of final PM4 object identity.
- The bundle is intentionally derived from the **currently visible overlay** so
  screenshots, peer counts, and exported summaries stay aligned with what the
  operator is looking at on screen.
