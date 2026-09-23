# Research — Spec 227 UI Re-Audit

**Date**: 2026-09-06

## Decisions

### Audit before UI mutation

- **Decision**: `surface-inventory-v2.md` is the prerequisite for all consolidation edits and
  names the surface being replaced plus its direct replacement.
- **Rationale**: Spec 227 prohibits consolidation without that row; the current source still has
  several compatibility and legacy entry points that must not be removed by assumption.
- **Alternatives considered**: Start with an obvious visual restyle. Rejected because it could
  leave a second data authority or move a control before the operator can find it.

### Preserve existing action and route authority

- **Decision**: presentation standardization calls `SharedUiWidgets` around existing actions and
  uses the existing `OpenWorkbenchTab` / `WorkbenchNavigator` routes.
- **Rationale**: `SharedUiWidgets` is explicitly presentation-only; existing routes preserve
  settings, shortcuts, and compatibility callers.
- **Alternatives considered**: introduce a new sidebar state manager. Rejected under the
  god-class freeze until Spec 228 selects an extraction boundary.

### Duplicate disposition is evidence-led

- **Decision**: weak-signal retains the Spec 194 owner recorded by the spec; minimap and Inspector
  replacements remain inventory decisions until the source matrix and operator walkthrough prove
  the authoritative route.
- **Rationale**: the source shows shared minimap rendering but multiple hosts, and the operator
  reports that teleport reliability differs by host. Source inspection alone cannot prove input
  behavior or visual reachability.
- **Alternatives considered**: retire all non-tiny minimaps immediately. Rejected because that
  would infer a runtime result that has not been witnessed.

### Evidence has two levels

- **Decision**: source-audit receipts and build output may establish source provenance; screenshots
  and interaction results remain explicit operator evidence.
- **Rationale**: this prevents a green build from being mistaken for a usable UI or reliable
  teleport interaction.
- **Alternatives considered**: mark the full inventory complete from source code. Rejected because
  US1 expressly requires screenshots.

## Source findings used by the first inventory

| Finding | Source evidence | Consequence |
|---|---|---|
| Four visible workbench roots are Quick, Inspector, Editor, and Archaeology. | `ViewerApp_Sidebars.cs` renders their top buttons; `WorkbenchNavigator.GetBottomTabLabels` provides their page labels. | Audit by those roots, then record legacy entry points separately. |
| Archaeology has six declared pages. | `WorkbenchNavigator.GetArchaeologyWorkbenchLabels`. | They form individual v2 rows rather than one undifferentiated panel. |
| Weak-signal is dispatched by Archaeology and still reachable from Editor Terrain Lab. | `ViewerApp_Sidebars.cs`: `DrawArchaeologyWorkbenchSubTabContent`, `DrawTerrainLabSubTab`. | V2 must record the duplication before a link/retirement edit. |
| Minimap rendering is shared but hosted by both sidebar and window paths. | `MinimapHelpers.RenderMinimapContent`; `ViewerApp_MinimapAndStatus.DrawMinimapWindow`; `ViewerApp_Sidebars.DrawUtilitiesMinimap`. | Source cannot select the teleport-authoritative host; operator matrix required. |
| Inspector context already uses `SharedUiWidgets` in several places. | `ViewerApp_Sidebars.DrawInspectorContextPage`; `Workbench/InspectorContentHost.cs`. | First standardization may reuse these primitives without a new style system. |

No external research or API contract is required for this desktop-only UI pass.
