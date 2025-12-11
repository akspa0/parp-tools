# Per-Tile Detail Pages - Future Implementation

## Status: ⏳ DISABLED (Pending Design)

Per-tile detail pages have been temporarily disabled while we design the uniqueID timeline selector and patched ADT export functionality.

---

## What Was Disabled

### Files:
- `ViewerAssets/tile.html` → renamed to `tile.html.disabled`
- `ViewerAssets/js/main.js` → `openTileViewer()` function commented out

### Behavior:
- **Before**: Clicking a tile navigated to `tile.html?map=X&row=Y&col=Z&version=V`
- **After**: Clicking a tile just logs to console (no navigation)

---

## Why Disabled

We need to figure out:
1. **UniqueID Timeline Selector** - How to visualize and select object ID ranges across versions
2. **Patched ADT Export** - How to write modified ADT files with selected object ranges
3. **UI/UX** - What controls and workflow make sense for this feature

Current main viewer is sufficient for:
- ✅ Viewing terrain overlays
- ✅ Comparing versions
- ✅ Exploring object placements
- ✅ Debugging area mappings

Per-tile detail is needed for:
- ⏳ Fine-grained object selection (by uniqueID range)
- ⏳ Exporting patched ADT files
- ⏳ Timeline-based filtering

---

## Planned Features (When Re-Enabled)

### 1. UniqueID Timeline Selector

**Concept**: Visual timeline showing object additions/removals across versions

```
Version:  0.5.3      0.5.4      0.5.5      0.6.0
          ├──────────┼──────────┼──────────┤
Objects:  [====A====][==A+B==][A+B+C][B+C]
          
Legend:
  A = Objects 1-100
  B = Objects 101-150
  C = Objects 151-200
```

**UI Ideas**:
- Slider with version markers
- Checkbox groups for object ID ranges
- "Select all from version X" buttons
- Range input (min/max UID)

**Backend**:
- Parse ADT placement data per version
- Group objects by uniqueID ranges
- Track adds/removes between versions
- Generate timeline metadata

---

### 2. Patched ADT Export

**Goal**: Export modified ADT files with selected object ranges

**Workflow**:
1. User selects tile on main map
2. Opens per-tile detail page
3. Adjusts uniqueID range sliders
4. Previews objects in/out
5. Clicks "Export Patched ADT"
6. Downloads .adt file with filtered objects

**Technical Challenges**:
- **ADT Writing**: Need to serialize WDT/ADT/WMO group chunks correctly
- **Offset Recalculation**: MCNK offsets, chunk sizes must be recalculated
- **Reference Integrity**: M2/WMO references must remain valid
- **Format Variations**: Alpha vs LK ADT formats differ

**Existing Code**:
- `AlphaWDTAnalysisTool` has ADT reading logic
- `AlphaWdtAnalyzer.Core.Export.AdtWotlkWriter` has partial writing logic
- May need to port/refactor for WoWRollback

---

### 3. Per-Tile Detail Page Design

**Layout Concept**:

```
┌────────────────────────────────────────────────┐
│  ← Back to Map    Tile [31,34] - Azeroth      │
├────────────────────────────────────────────────┤
│                                                │
│  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  Minimap        │  │  Object List        │ │
│  │  (512x512)      │  │                     │ │
│  │                 │  │  □ Tree (UID 1245)  │ │
│  │                 │  │  □ Rock (UID 1246)  │ │
│  │                 │  │  □ House (UID 1247) │ │
│  └─────────────────┘  │  ...                │ │
│                       └─────────────────────┘ │
│  ┌──────────────────────────────────────────┐ │
│  │  UniqueID Timeline Selector              │ │
│  │  ────────●───────────●────────────●──    │ │
│  │  0.5.3   0.5.4      0.5.5       0.6.0    │ │
│  │                                           │ │
│  │  Range: [1000] to [1500]                 │ │
│  │  [ Export Patched ADT ]                  │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  Version Comparison:                          │
│  ┌──────────┬──────────┬──────────┐          │
│  │  0.5.3   │  0.5.5   │   Diff   │          │
│  ├──────────┼──────────┼──────────┤          │
│  │  125 M2  │  142 M2  │  +17 M2  │          │
│  │   38 WMO │   38 WMO │   +0 WMO │          │
│  └──────────┴──────────┴──────────┘          │
└────────────────────────────────────────────────┘
```

**Components Needed**:
- Minimap display (similar to main viewer)
- Object list with filtering
- Timeline slider/selector
- Version comparison table
- Export button with progress indicator

---

## Implementation Phases

### Phase 1: Data Collection
- [ ] Extract uniqueID ranges from ADT placement data
- [ ] Build per-tile object metadata (id, type, position, version)
- [ ] Generate timeline JSON files

### Phase 2: UI Prototyping
- [ ] Design timeline selector component
- [ ] Mockup per-tile page layout
- [ ] Test UX with sample data

### Phase 3: ADT Writing
- [ ] Research ADT format specification
- [ ] Implement ADT serializer (or port from AlphaWDTAnalysisTool)
- [ ] Test with simple modifications (remove 1 object)
- [ ] Validate in WoW client

### Phase 4: Integration
- [ ] Re-enable tile.html
- [ ] Connect timeline selector to object filtering
- [ ] Wire up export button
- [ ] Test end-to-end workflow

### Phase 5: Polish
- [ ] Add validation (warn if missing refs)
- [ ] Progress indicators for export
- [ ] Error handling
- [ ] Documentation

---

## Technical Notes

### UniqueID Data Structure

**Per-Tile Metadata** (to be generated):
```json
{
  "map": "Azeroth",
  "tile": { "row": 31, "col": 34 },
  "versions": [
    {
      "version": "0.5.3.3368",
      "objects": [
        {
          "uid": 1245,
          "type": "m2",
          "model": "Tree.mdx",
          "position": [123.4, 456.7, 89.0]
        }
      ],
      "uid_range": { "min": 1000, "max": 1500 }
    }
  ]
}
```

### ADT Export API (Conceptual)

```csharp
public class AdtPatcher
{
    public byte[] PatchAdt(
        string inputAdtPath,
        int minUid,
        int maxUid)
    {
        // 1. Parse ADT
        var adt = AdtReader.Read(inputAdtPath);
        
        // 2. Filter placements
        adt.M2Placements = adt.M2Placements
            .Where(p => p.UniqueId >= minUid && p.UniqueId <= maxUid)
            .ToList();
        
        // 3. Recalculate offsets
        adt.RecalculateOffsets();
        
        // 4. Serialize
        return AdtWriter.Write(adt);
    }
}
```

---

## Re-Enabling Checklist

When ready to implement:

1. **Rename file**: `tile.html.disabled` → `tile.html`
2. **Uncomment code**: Restore `openTileViewer()` in `main.js`
3. **Implement features**:
   - [ ] Timeline selector component
   - [ ] Object metadata generation
   - [ ] ADT patching logic
4. **Test thoroughly**:
   - [ ] Timeline interaction
   - [ ] Export functionality
   - [ ] Patched ADT loads in WoW client
5. **Update docs**:
   - [ ] User guide for per-tile workflow
   - [ ] Developer docs for ADT format
   - [ ] Known limitations

---

## Current Workaround

For now, users can:
- ✅ View terrain overlays on main map
- ✅ Toggle object markers on/off
- ✅ See object counts in console logs
- ✅ Use browser dev tools to inspect placement data

For manual ADT patching:
- Use AlphaWDTAnalysisTool directly
- Export to LK format with specific flags
- Manually edit placement chunks

---

## Summary

Per-tile detail pages are **temporarily disabled** to focus on:
1. Main viewer features (terrain overlays ✅)
2. Multi-map support (in progress ⏳)
3. Data consolidation (planned 📋)

Will be **re-enabled** when:
1. UniqueID timeline selector is designed
2. ADT patching logic is implemented
3. UX workflow is validated

**Status**: File disabled, code commented, ready to restore when needed! 🎯
