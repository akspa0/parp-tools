# WMO 0.9.0 Render Path — Part 05 (Parser↔Render Contract Cross-check)

## Scope
Cross-check parser-proven invariants against renderer-side WMO group and liquid functions.

## Parser-side proven invariants (existing converter/parser track)
- Strict required chunk ordering and optional-flag gating for group chunks.
- Optional dependency checks (paired/clustered optional chunks) remain structural requirements.
- Liquid chunk payload is treated as structured data with bounds checks; no blind pass-through.

## Renderer-side anchor points (to fill)
- `TBD_FUN_WMO_GROUP_RENDER_ENTRY`: group stream/material/light consumption.
- `TBD_FUN_WMO_LIQUID_BUILD`: liquid grid decode and vertex/index generation.

## Contract implications
1. Group material/color/light streams must remain index-domain coherent at render handoff.
2. Liquid grid dimensions, axis convention, and traversal order must match renderer assumptions.
3. Parser and renderer both enforce bounds assumptions; parser-side hard-fails are required for 0.9.0 profile.

## Updated confidence
- Parser contract: **High**.
- Renderer function localization: **Pending**.
- Color/light gate operand closure: **Pending**.
- Liquid layout gate operand closure: **Pending**.

## Practical tooling rule (0.9.0 profile)
- Treat group color/light coherence and liquid layout coherence as mandatory pre-render validation inputs.
- Do not downgrade parser hard-fails into warnings for this profile.
