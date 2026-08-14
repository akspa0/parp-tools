# Alpha 0.5.3 Area Overlay Ghidra Evidence

Last verified: 2026-08-14 against the open Ghidra `WoWClient.exe` program in project `0.5.3`.

This note records the native data boundary used by Spec 149's resident Zone/SubZone overlay. It does
not claim that the client exposes complete polygon boundaries through MCNK.

## Native anchors

- `CMapChunk::Create @ 0x00698e10` stores `this->zoneId = *(uint *)(param_1 + 0x40)`. The function
  receives the MCNK payload after its 8-byte chunk header, so this is the file MCNK area value at
  payload offset `0x38`, matching the viewer's `TerrainChunkData.AreaId` source (`Unknown3`).
- `AreaTableRec::Read @ 0x00585e20` reads the 0.5.3 AreaTable row sequentially. The first fields are
  `AreaNumber`, `ContinentID`, `ParentAreaNum`, then flags/audio fields and localized string offsets.
- `WowClientDB<AreaTableRec>::Load @ 0x0055cf30` loads `DBFilesClient\\AreaTable.dbc`, validates the
  `WDBC` header, expects 22 columns and an 88-byte row, and indexes records by their database ID.
- `Script_GetZoneText @ 0x004e3620`, `Script_GetSubZoneText @ 0x004e3640`, and
  `Script_GetMinimapZoneText @ 0x004e3660` return separate native UI strings. The status-bar-style
  distinction is therefore ZoneText versus SubZoneText, not a single display-name field.

## Overlay contract

The 0.5.3 native evidence supplies area identity per resident `CMapChunk`, not a complete area polygon.
The viewer therefore groups currently resident chunk bounds after the existing map-aware
`AreaTableService.ResolveArea` path:

- Zone groups use the resolved parent identity when present, otherwise the canonical current-area ID.
- Subzone groups use the canonical current-area identity only when it is distinct from the parent.
- Keys include map, kind, and canonical identity, so duplicate names do not merge accidentally.
- Missing or unresolved rows are counted but do not produce guessed geometry or labels.
- The render overlay uses resident chunk footprint outlines and one pin/label per group; it is opt-in and
  refreshes from a renderer residency revision rather than scanning unloaded terrain.

This is a faithful resident-coverage visualization. It must not be described as an authoritative full
zone polygon until a separate native boundary source is proven.
