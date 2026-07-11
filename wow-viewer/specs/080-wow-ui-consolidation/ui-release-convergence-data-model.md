# UI Release Convergence Data Model

## UiSurfaceRecord

One row per user-visible surface or route in the release inventory.

| Field | Meaning |
|---|---|
| `surface_id` | Stable identifier used by audit and manual tests. |
| `label` | User-visible name. |
| `source_method` | Active viewer draw or action method. |
| `entry_points` | Menu, toolbar, bottom bar, hotkey, or frame routes. |
| `tabbed_route` | Draw destination when `_useTabUi` is true. |
| `legacy_route` | Draw destination when legacy/dockspace mode is active. |
| `runtime_prerequisites` | Required loaded model/world/tool state. |
| `owner` | Canonical frame, sidebar, or global window. |
| `status` | `working`, `misrouted`, `missing`, `placeholder`, `duplicate`, `retired`, or `disabled-with-reason`. |
| `predecessor_specs` | Historical source plans. |
| `manual_proof` | Required release test and evidence path. |

## UiRouteResult

One execution result for a surface in a named UI mode.

| Field | Meaning |
|---|---|
| `surface_id` | Links to `UiSurfaceRecord`. |
| `ui_mode` | `tabbed` or `legacy`. |
| `input_state` | Model/world/client prerequisites used. |
| `result` | `passed`, `disabled-explained`, or `failed`. |
| `evidence` | Screenshot/log path and reviewer note. |

## ConverterCardState

The user-facing state for a 073b command card: command identity, input
contract, availability, in-progress flag, last exit status, and bounded output
summary. The command itself remains owned by `WowViewer.Tool.Converter`.

## UiPerformanceSample

One fixed-camera measurement for a named surface configuration.

| Field | Meaning |
|---|---|
| `scenario_id` | Stable baseline case, such as `minimap-open` or `pm4-overlay-on`. |
| `ui_mode` | `tabbed` or `legacy`. |
| `surface_toggles` | Exact minimap/overlay/object/UI state. |
| `cpu_frame_ms` | Total CPU frame duration. |
| `ui_ms` | Major ImGui surface draw duration when instrumented. |
| `renderer_costs` | Existing terrain/WMO/MDX/liquid/overlay/asset-load measurements. |
| `draw_pressure` | Existing WMO/MDX and overlay submission counts. |
| `minimap_activity` | Cache hit/miss, decode/upload, and load-budget activity. |
| `memory` | Process, managed heap, MPQ cache, and world raw-cache counters. |
| `evidence` | Runtime Stats capture and reviewer note. |
