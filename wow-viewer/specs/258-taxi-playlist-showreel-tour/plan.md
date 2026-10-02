# Plan — Spec 258: Taxi Route Playlists & Dynamic Marketing Showreel Tour

## Architectural Strategy

This design provides a unified solution for multi-route taxi playlists and real-time marketing showreel telemetry overlays without video editing software.

```
                          ┌────────────────────────┐
                          │   ViewerApp (Shell)    │
                          └───────────┬────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌───────────────────┐      ┌─────────────────────────┐  ┌─────────────────────┐
│TaxiPlaylistService│◄────►│RecordingCoordinator     │  │ShowreelOverlay      │
│(Manages playlist, │      │Service                  │  │Service              │
│ chains, advance)  │      │(FFmpeg video piping,    │  │(Live coords, zones, │
└────────┬──────────┘      │ rate pacing, session)   │  │ landmarks, badges)  │
         │                 └─────────────────────────┘  └──────────┬──────────┘
         ▼                                                         ▼
┌───────────────────┐                                   ┌─────────────────────┐
│TaxiPanelService   │                                   │ImGui Foreground     │
│(UI controls,      │                                   │DrawList Overlay     │
│ playlist manager) │                                   └─────────────────────┘
└───────────────────┘
```

---

## 1. Taxi Route Playlist System

### `TaxiPlaylistService` (`src/viewer/WoWViewer/Workbench/Services/TaxiAreaPoi/TaxiPlaylistService.cs`)
- **State**:
  - `List<TaxiPlaylistItem> Items`: ordered list of routes (`PathId`, `FromNodeId`, `ToNodeId`, `FromName`, `ToName`).
  - `int CurrentIndex`: index of currently executing route segment.
  - `bool IsPlaying`: whether playlist playback is active.
  - `bool IsRecording`: whether recording across the playlist is active.
  - `bool Loop`: optional repeating loop.
- **Key Methods**:
  - `void AddRoute(TaxiPathLoader.TaxiRoute route, string fromName, string toName)`
  - `void RemoveAt(int index)`
  - `void MoveUp(int index)` / `void MoveDown(int index)`
  - `void Clear()`
  - `bool BuildAutoChain(int startNodeId, int hopCount)`: discovers connected outgoing routes from node to node to build an unbroken cross-continent tour.
  - `bool StartPlaylist(bool recordVideo, RecordingRequest? videoOptions = null)`: mounts actor, attaches camera to route 0, starts continuous video recording if requested.
  - `void Update(double dt)`: monitors route progress. When current route reaches destination node (travel >= length * 0.98f), cleanly transitions to route index + 1 without stopping the recording!
  - `void StopPlaylist(string reason)`: concludes playlist and finalizes video recording.

---

## 2. Dynamic Showreel HUD & Marketing Overlays

### `ShowreelOverlayService` (`src/viewer/WoWViewer/Workbench/Services/Recording/ShowreelOverlayService.cs`)
- **Live World Telemetry**:
  - Position: $(X, Y, Z)$ formatted in WoW engine units.
  - Orientation: Heading in degrees ($0^\circ - 360^\circ$) + 8-point compass ($N, NE, E, SE, S, SW, W, NW$) + Pitch.
  - Geography: Map Name (`Azeroth`, `Kalimdor`), Map ID, ADT Tile Coordinates $[X, Y]$ calculated via standard formula `32 - pos / 533.33333f`, MCNK chunk coordinates.
  - Flight Status: Active route name, progress percentage, segment indicator ($1/4$).
- **Cinematic Zone Entry Banners**:
  - Subscribes to `CurrentAreaLookup` / `CurrentAreaName` from `AreaContextService`.
  - When zone or subzone changes: triggers a title card animation:
    - Large typography: Primary Zone (e.g. `ELWYNN FOREST` or `WESTFALL`)
    - Subtitle: Subzone / Region (e.g. `Goldshire` or `Sentinel Hill`)
    - Area badge: Area ID & map ID
    - Smooth alpha fade-in (0.5s), display duration (3.5s), fade-out (0.8s).
- **Proximity Landmark Callouts**:
  - Scans `TaxiLoader.Nodes` within proximity ($< 350$ yards) in front of the camera.
  - Shows discovery badge: `Flight Master: Sentinel Hill (Node #12) • 120 yd`.
- **Engine Technical Badges (Patreon Highlights)**:
  - Rotates informative callouts highlighting the viewer's engine feats:
    - `WoW Format Engine: Authentic 1.12.1 DBC Flight Mechanics`
    - `Terrain System: 64-bit ADT Hole Punching & High-Res Liquids`
    - `Renderer: Native M2 GPU Skinning & Instanced Geometry`

---

## 3. UI & CLI Integration

- **UI (`TaxiPanelService.Sidebars.cs`)**:
  - Collapsible header: "Playlist & Marketing Tour".
  - List of playlist routes with remove/reorder controls.
  - "Add Current Route to Playlist" button.
  - "Build Connected Tour from Node" button.
  - "Play Playlist" and "Record Playlist Video" buttons.
  - Checkboxes for Showreel HUD features (Telemetry, Zone Banners, Landmark Callouts, Engine Badges).
- **CLI Options (`StartupAutomationService.cs`)**:
  - `--record-taxi-playlist <RouteIdsCsv>` (e.g. `1,2,3`)
  - `--record-taxi-chain <StartNodeId>,<Hops>` (e.g. `2,4` - chains 4 hops starting at node 2)
  - `--showreel-hud` (enables showreel HUD overlay during recording)
  - `--showreel-zone-banners` (true/false)
  - `--showreel-telemetry` (true/false)
  - `--showreel-landmarks` (true/false)
  - `--showreel-engine-badges` (true/false)

---

## 4. Architectural Boundaries

- No new fields in `WorldScene.cs` or `ViewerApp.cs` (AGENTS.md §10).
- Registered in `IViewerAppHost` and `ViewerApp_Host.cs`.
- Rendered on `ImGui.GetForegroundDrawList()` during UI pass so it captures into video recording without showing main developer sidebars.
- Strictly adheres to <= 2,000 line budget.
