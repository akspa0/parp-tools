# UI Release Convergence Research

## Decision: Treat Spec 080 as the sole completion owner

**Rationale**: Specs 049, 069, 070, and 071 contain incompatible layout
decisions. Spec 080 already owns the active viewer shell and can record the
disposition of each predecessor without resurrecting competing architectures.

**Alternatives considered**:

- Finish each historical plan independently: rejected because their ownership
  and layout assumptions conflict.
- Copy legacy panels wholesale: rejected because the legacy viewer is
  read-only reference evidence and would preserve duplicate/dead routes.

## Decision: Audit routes before redesigning frames

**Rationale**: Active source confirms a concrete tabbed-mode failure: menu
handlers set `_showSettingsWindow`, but the main draw loop renders Settings
only in the legacy branch. Similar window flags are intentionally suppressed
in tabbed mode, so source inventory must establish whether a valid sub-tab
replacement exists before controls are moved or removed.

**Alternatives considered**:

- Immediately replace sidebars with named frames: rejected because it can
  hide more currently reachable controls.
- Keep all duplicate controls: rejected because it prevents users finding a
  canonical owner and violates the completed de-duplication work.

## Decision: Converters are one Tools surface, not viewer logic

**Rationale**: 073b defines cards that invoke the existing converter
executable. Its entire task pack remains open, so the release plan owns its
user-facing route but does not duplicate conversion parsing or implementation.

## Decision: Preserve source/build selection and Model animations

**Rationale**: 057 owns client/version selection while 053 owns animation
data tooling. The release UI needs their existing user-facing controls, but
does not need their archive-backend or AnimFarm deliverables to converge.

## Decision: UI performance is measured with the existing renderer counters

**Rationale**: Spec 090 already exposes process/cache memory and Spec 093
already exposes WMO/MDX/terrain/liquid/overlay pressure, but neither has its
dense-map capture matrix completed. The release plan adds per-surface UI and
minimap attribution only where those counters cannot explain a cost.

**Alternatives considered**:

- Assume ImGui is cheap and optimize renderers first: rejected because minimap
  visibility changes world loading cadence and UI overlays can create real CPU
  submission/allocation pressure.
- Rewrite WMO/MDX batching immediately: rejected because 093 requires a
  measured top-cost decision first.
