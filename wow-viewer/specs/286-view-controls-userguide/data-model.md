# Data model — Spec 286

Proposed local contracts, not implemented APIs. Existing settings, selection and counters remain
authoritative; immutable presentation does not add competing stores.

| Entity | Fields / relationships | Validation / lifecycle |
|---|---|---|
| View preference | user adjustment, default, existing settings key | finite 0.85–2.20/default 1.0; normalize persisted outliers once; effective display scale manager-owned |
| Reset action | layout-only, scale-only, combined | touches exactly named owners; keeps client/map |
| Activation | monotonic generation, source identity, success | advances on committed activation only; refresh/fail/cancel do not advance |
| Maps request | generation, discovery outcome, consumed flag | at most once for current generation; stale completions ignored; close respected |
| Footer view | actual region, effective scale, essential actions/FPS, existing diagnostics snapshot | no scrolling or hover I/O; detailed payload disclosed on demand |
| Selection view | existing key, display/fallback identifier, kind, semantic category/evidence | clears/rebuilds on selection/client changes; no repicking or invented HP |
| Category style | category, color, readable label, evidence/fallback reason | same policy for header/hover/bounds; Unknown explicit; emphasis independent |
| Illustration | build, workflow, display/scale, relative image, caption, review state | missing → captured → reviewed; only genuine reviewed images count |

Transitions: activation pending → committed → current discovery → Maps request consumed; failed
loads never enter the success branch. Selection empty → existing key → presentation → replace/clear.
FPS idle → tooltip → optional manager-owned popup → closed; snapshot collection stays unchanged.

No database/schema migration follows from these transient views. Change layout version only if
actual persisted-layout incompatibility is demonstrated, not merely because controls move.
