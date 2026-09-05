# UI Surface Inventory — Spec 223 Phase 0

**Created**: 2026-09-04
**Method**: source audit of the ImGui shell (`ViewerApp*.cs` partials, `Workbench/`), cross-checked
against the running viewer (operator screenshots). Every row needs a disposition before any
consolidation change lands (FR-1). Rows marked ⏳ need runtime confirmation during the walkthrough.

## 1. Shell structure (verified)

- **Top bar tabs** = `WorkspaceMode`: **Viewer / Editor / Archaeology**
  ([`ViewerApp_Workspaces.cs`](../../src/viewer/WoWViewer/ViewerApp_Workspaces.cs) — `SetWorkspaceMode`
  maps Editor→`WorkbenchTab.Editor`, Archaeology→`WorkbenchTab.Archaeology`, Viewer→`WorkbenchTab.Quick`).
- **Right sidebar "workbench" destinations** = [`WorkbenchTab`](../../src/viewer/WoWViewer/Workbench/WorkbenchTab.cs):
  Quick (F1), Inspect (F2), Scene (F3), Utilities (F4), Experimental, Editor (F6), Archaeology (F5).
  Page selectors per destination in [`WorkbenchNavigator`](../../src/viewer/WoWViewer/Workbench/WorkbenchNavigator.cs).
- **Left sidebar** (`##LeftSidebar` / legacy `##LegacyLeftSidebar`), **right sidebar** (`##RightSidebar`),
  **bottom bar** (`##BottomBar`), **status bar** (`##statusbar`), toolbars (`##Toolbar`,
  `##CenteredTerrainToolbar`), dockspace host (legacy, off when `_useTabUi`).
- **Legacy dual mode**: `_useTabUi=false` restores the pre-069 dockspace + floating windows. The
  menu item already disables switching back once tab UI is on. Disposition: **retire the legacy
  path** once 223 lands (it doubles every surface's test matrix).

## 2. Right-sidebar workbench destinations (verified)

| Destination | Pages | Content today | Owner spec | Disposition |
|---|---|---|---|---|
| Quick | — | Camera speed/FOV, ADT detail tiles, time-of-day, Fog Start/End, LIT fog, reset camera, wireframe, hide chrome, theme | 069/080 | **Keep, upgrade** (US5: per-profile top controls; Fog End already here — keep first-class) |
| Inspect | Context, Scene Investigation, MCNK/ADT, World Context, Archeology, Animations, Actions | Selection details, investigation tools | 080/156 | **Keep; becomes the authoritative object inspector** (US2) |
| Scene | Placements, LOD | Placement lists, LOD controls | 080 | Keep; merge with Inspect candidates during dedup |
| Utilities | Minimap, Log, Perf, Render Quality, Taxi, Capture, Asset Catalog, Runtime Stats, Lighting, Audio | 10 utility pages | 080/107/147 | Keep; Render Quality duplicates Settings>Render Quality (see §5) |
| Experimental | Terrain Lab, PM4, Converters, Population | Terrain Lab (Clipboard/Analysis/MCNK/Stratigraphy/Export/Tools), PM4 (7 pages), converter launchers | 080/195/130 | **Reorganize**: Terrain Lab is a tile-detail duplicate cluster; Converters are editor-write features → Editor profile after split |
| Editor | — | Editor workspace task nav (Terrain/Objects/PM4 Evidence/Inspect/Publish) | 197 | **Merge into Archaeology first** (US4), then split true-editor back |
| Archaeology | Weak Signal & Stratigraphy, UniqueId Timeline, Layers & Provenance, Playback & Capture, PM4 Analysis | Analysis surfaces | 194/149/144 | **Becomes the merged home** (US4); gains Cartography (222) |

## 3. Left sidebar sections (verified)

| Section | File:line | Owner | Disposition |
|---|---|---|---|
| World Overview (×2 — tab UI + legacy) | Sidebars:545, 606 | 080 | Merge the two implementations into one |
| File Browser | Sidebars:611 | — | Keep (navigation) |
| World Maps (×2) | Sidebars:617; Workspaces:164 | 080/197 | Merge duplicates |
| Phase Map Layers | PhaseLayers:22 | 222 | **Move to right sidebar under Cartography** (222-T109); delete here |
| Editor Workspace task nav | Workspaces:118 | 197 | Merge with Archaeology (US4) |
| Chunk Clipboard (×2) | Workspaces:201; Sidebars:3887 (floating) | 195/169 | One home only (US2) |
| Model Info | Workspaces:286 | — | Candidate duplicate of Inspect>Animations/Context |
| Camera | Workspaces:290 | — | Duplicate of Quick camera controls — retire to pointer |

## 4. Floating windows (verified — FR-7 targets)

| Window | File:line | Owner | Disposition |
|---|---|---|---|
| Minimap (+ fullscreen variant) | MinimapAndStatus:530,648; ViewerApp:9661 | 147 | Keep as window (map-canvas needs space) but single implementation; the ×3 call sites merge |
| Settings | Settings:15 | — | Keep (the one window that is reliably recallable) |
| MCNK Explorer | Investigation:287 | 080 | **Tile-detail duplicate #1** → fold into authoritative tile inspector |
| ADT Chunk Investigation / WL Liquid / LIT Lighting | Investigation:208,310,505 | 148/143 | Analysis pages → Archaeology |
| Terrain Workbench | Sidebars:1816 | 195 | Retire after 222 Cartography parity (222-T111) |
| Chunk Clipboard (floating) | Sidebars:3887 | 169 | Merge into merged profile's clipboard section |
| UniqueId Archaeology / Weak Signal Amplifier | Sidebars:4058,4075 | 149/194 | → Archaeology destination pages |
| Terrain Analysis | TerrainAnalysis:25 | 194 | → Archaeology |
| Synthesized Terrain Minimap Export | SynthesizedMinimapExport:41 | 222-T109a | → Cartography right-sidebar tab (floating window retired) |
| Camera Path | CameraPaths:118 | 144 | → Utilities>Capture page |
| Capture Automation | CaptureAutomation:234 | 144 | → Utilities>Capture |
| Perf / Render Quality (floating) | Pm4Utilities:917; RenderQuality:60 | 201 | Perf → Utilities>Perf (exists); Render Quality floating duplicates Settings>Render Quality — retire one |
| PM4 Alignment / Correlation / Object Match | Pm4Utilities:1148,1840,2244 | 130/046 | → Experimental>PM4 pages (already tabbed — dedupe the floating entry points) |
| Log Viewer | LogViewer:22 | — | → Utilities>Log (already exists as page — retire floating) |
| Train V7 / Build ML Dataset / Map Converter / WMO Converter / Terrain Texture Transfer | MlTraining:437; ViewerApp:6975,7473,7772,7975 | 192/221 | Editor-write features → Editor profile after split; entries stay as launchers |
| Import Alpha Masks / MCCV / Heightmaps | ViewerApp:3240,3395,3627 | 197 | Editor profile |
| WDL Preview / Map Preview | WdlPreview:47,79 | 108 | → Utilities or Archaeology page |
| SceneHoverAssetOverlay / TerrainChunkHoverOverlay | ViewerApp:14967; Investigation:704 | 156 | **Tile-detail duplicates #2/#3** — hover overlay becomes a summary that links to the authoritative inspector |
| Select Client Build / Rosetta / Map Converter dialogs | ClientDialogs:42,330 | — | Keep (modal dialogs are legitimate) |

## 5. The tile-detail duplicate cluster (US2 target — the operator's "3–5 ways")

Verified distinct surfaces that show details about the selected tile/chunk:

1. **Selected Object inspector** (right sidebar, Sidebars:2265) — selection details.
2. **SceneHoverAssetOverlay** (ViewerApp:14967) — hover summary card.
3. **MCNK Explorer** (floating, Investigation:287) — per-MCNK byte-level details.
4. **ADT Chunk Investigation** (Investigation:208) + **TerrainChunkHoverOverlay** (Investigation:704).
5. **Terrain Lab > MCNK / Analysis / Stratigraphy** (Experimental pages, TerrainBottomTab).
6. **Inspector tabs** (Sidebars:939 `##InspectorTabs`) — legacy right-sidebar tab set.

**Disposition**: designate **one** authoritative per object type (tile/chunk → one surface; WMO →
one; doodad → one; model → one; PM4 object → PM4 Selection page). The others become pointers or
are absorbed. Exact mapping decided in plan phase against this list.

## 6. Known missing control (US3)

- **WMO doodad-set switching**: MODS/MODD data is parsed (`WmoDoodadSetSummaryReader*` tests exist)
  and the selected-WMO inspector exists, but no UI exposes set switching. Add to the authoritative
  WMO detail surface.

## 7. Quick tab today (US5 baseline, verified Sidebars:5614)

Camera speed, FOV, ADT detail tiles, time-of-day, Fog Start/End, LIT fog toggle, reset camera,
wireframe, hide chrome, theme. **Already contains Fog End** — the gap is that (a) it exists only
in the Viewer profile's Quick, not per-profile, and (b) Settings>Fog Defaults duplicates it
(Settings:33). Per FR-6 keep one implementation, mirror everywhere.

## 8. Operator decisions (2026-09-04, answering §7 questions)

- **Q1 — merged profile name: keep "Archaeology".** The merged Editor+Archaeology profile is
  called Archaeology; the later split moves true-editor features into a dedicated Editor profile.
- **Q2: KEEP the legacy non-tab UI** (`_useTabUi=false`). It remains useful until the 3D object
  HUD (Spec 212) is functional — which is still MIA. Do not retire it in this spec; both shell
  modes stay maintained.
- **Q3: ONE unified object inspector.** All object inspection surfaces consolidate into a NEW
  right-sidebar **Inspector** tool with per-type sections: **ADT / MDX / M2 / WMO / PM4**, plus
  **WL\*** liquid inspection. Everything duplicated (MCNK Explorer, ADT Chunk Investigation,
  hover overlays, Terrain Lab MCNK pages, legacy InspectorTabs, PM4 detail pages) folds into it.
  The Inspector must be built HUD-ready — structured so it can convert into the 3D object HUD
  (Spec 212) later or now, whichever makes sense during implementation.