# Ghidra Note: Why WDL Does Not Render On Top of Terrain Tiles

Date: 2026-02-13  
Binary: `WoWClient.exe` (Alpha 0.5.3 / build 3368)

---

## Direct Answer

The client avoids visible WDL-over-terrain overlap through **two primary mechanisms**:

1. **Selection/culling rules** in `CullHorizon` only include suitable low-detail horizon areas (and exclude loaded/blocked tile regions).
2. **Render pass order** draws WDL horizon first, then draws regular terrain chunks afterward, so normal terrain depth wins where chunk geometry exists.

No explicit polygon offset/depth-bias path was identified in the traced WDL render functions.

---

## 1) Pass Order Evidence

`CWorldScene::Render` (`0x0066a9d0`) call order includes:

- `RenderHorizon` (`0x0066daf0`) — WDL low-detail pass
- `RenderChunks` (`0x0066de50`) — regular terrain tiles

Because chunk terrain is rendered after horizon, it naturally covers/overrides horizon geometry where both project to the same screen region.

---

## 2) WDL Visibility Selection (CullHorizon)

`CullSortTable` (`0x0066ca50`) calls `CullHorizon` (`0x0066cad0`) to build `visAreaLowList`.

Key behaviors in `CullHorizon`:

- Expands area-rect window and clamps to valid 64x64 map bounds.
- Iterates candidate `areaLowTable` entries.
- Requires entry-level checks before adding to visible list:
  - loaded/blocked flag check (`entry + 0x8c4` path seen in decomp)
  - frustum test (`FrustumCull` call)
  - clip-buffer test (`ClipBufferCull` call)

Only passing areas are linked into low-detail visible list for horizon rendering.

---

## 3) RenderHorizon / RenderAreaLow State Behavior

### `RenderHorizon` (`0x0066daf0`)

- Saves projection.
- Builds/sets a horizon projection.
- Temporarily changes viewport.
- Renders visible low areas.
- Restores projection and viewport.

This keeps horizon draw behavior isolated to its dedicated pass.

### `CMap::RenderAreaLow` (`0x0069f360`)

Observed state setup includes disabling/changing blend/lighting/culling for low-detail draw, then submitting low mesh.

Notably, in this traced path:

- No explicit polygon offset API usage was found.
- No explicit depth-bias setup was found.

So overlap control is primarily from **what gets selected** and **when it is rendered**.

---

## 4) Practical Interpretation for External Viewers

To emulate client behavior and avoid WDL on top of high-detail terrain:

1. Maintain a visibility/culling gate for low-detail areas (don’t render all WDL cells blindly).
2. Render WDL horizon pass **before** regular chunk terrain pass.
3. Render full terrain tiles after horizon so high-detail tiles dominate final depth image.
4. Keep WDL as fallback/distant layer rather than a co-equal surface pass.

---

## 5) Address Reference

- `CWorldScene::Render` — `0x0066a9d0`
- `CullSortTable` — `0x0066ca50`
- `CullHorizon` — `0x0066cad0`
- `RenderHorizon` — `0x0066daf0`
- `RenderChunks` — `0x0066de50`
- `CMap::RenderAreaLow` — `0x0069f360`

---

If needed, next step is a strict frame-step trace documenting exact depth state values before/after `RenderHorizon` and `RenderChunks` in one table.