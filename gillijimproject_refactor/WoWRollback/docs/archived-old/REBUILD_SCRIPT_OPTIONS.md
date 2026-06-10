# rebuild-and-regenerate.ps1 Options

## Granular Refresh Switches

The script now supports three levels of refresh control:

### 🔄 `-RefreshCache` (Full Rebuild)
**What it does:**
- Deletes LK ADT cache
- Deletes analysis CSVs
- Runs AlphaWDTAnalysisTool to convert Alpha WDT → LK ADTs
- Extracts CSVs from LK ADTs
- Regenerates viewer overlays

**When to use:**
- Source Alpha WDT files changed
- LK ADT cache corrupted
- Complete rebuild needed

**Example:**
```powershell
.\rebuild-and-regenerate.ps1 -Maps @("Azeroth") -RefreshCache
```

---

### 📊 `-RefreshAnalysis` (CSV Rebuild - Fast!)
**What it does:**
- ✅ Keeps existing LK ADTs (NO re-conversion!)
- ❌ Deletes old analysis CSVs
- ✅ Re-extracts CSVs from existing LK ADTs using current code
- ⚡ **Skips expensive LK ADT generation** (52 min → 5 min for all maps!)

**How it works:**
- Runs AlphaWDTAnalyzer WITHOUT `--export-adt` flag
- Points tool to existing LK ADT cache
- Tool reads cached LK ADTs and extracts fresh CSVs
- No ADT conversion = **10x faster!**

**When to use:**
- CSV extraction code was fixed (like the AreaID bug fix)
- LK ADTs are correct but CSVs are wrong
- Much faster than full rebuild

**Example:**
```powershell
.\rebuild-and-regenerate.ps1 -Maps @("Azeroth") -RefreshAnalysis -AlphaRoot ..\test_data\
```

---

### 🎨 `-RefreshOverlays` (Viewer Rebuild)
**What it does:**
- Keeps existing LK ADTs and CSVs
- Deletes viewer overlay JSON files
- Regenerates overlays from existing CSVs

**When to use:**
- Overlay generation code was fixed
- CSVs are correct but viewer overlays are wrong
- Fastest refresh option

**Example:**
```powershell
.\rebuild-and-regenerate.ps1 -Maps @("Azeroth") -RefreshOverlays
```

---

## Decision Tree

```
Do you need to rebuild?
│
├─ Alpha WDT changed or LK ADTs corrupted?
│  └─ YES → Use -RefreshCache (full rebuild)
│
├─ CSV extraction code fixed (e.g., AreaID bug)?
│  └─ YES → Use -RefreshAnalysis (keep ADTs, rebuild CSVs)
│
├─ Viewer overlay code fixed?
│  └─ YES → Use -RefreshOverlays (keep CSVs, rebuild overlays)
│
└─ Nothing changed?
   └─ Run without flags (reuses all cached data)
```

---

## Current Bug Fix Scenario

**Problem:** Viewer shows "Unknown Area -286331154" (Alpha AreaIDs)

**Root Cause:** Old CSVs have Alpha AreaIDs, not LK AreaIDs

**Solution:** Use `-RefreshAnalysis`
```powershell
.\rebuild-and-regenerate.ps1 `
  -Maps @("Azeroth") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\ `
  -RefreshAnalysis
```

This will:
1. ✅ Keep existing LK ADTs (fast!)
2. ❌ Delete old CSVs with Alpha AreaIDs
3. ✅ Re-extract CSVs from LK ADTs using fixed code
4. ✅ Generate CSVs with proper LK AreaIDs
5. ✅ Rebuild viewer overlays with correct area names

---

## All Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-Maps` | string[] | `@("DeadminesInstance")` | Map names to process |
| `-Versions` | string[] | `@("0.5.3.3368","0.5.5.3494")` | Versions to compare |
| `-AlphaRoot` | string | *(required)* | Path to Alpha WoW client data |
| `-ConvertedAdtRoot` | string | *(optional)* | Pre-converted LK ADT directory |
| `-CacheRoot` | string | `"cached_maps"` | Cache directory for LK ADTs |
| `-RefreshCache` | switch | off | Force full rebuild (ADTs + CSVs) |
| `-RefreshAnalysis` | switch | off | Rebuild CSVs only (keep ADTs) |
| `-RefreshOverlays` | switch | off | Rebuild viewer overlays only |
| `-Serve` | switch | off | Auto-start HTTP server after build |
| `-UseNewViewerAssets` | switch | off | Use WoWRollback.Viewer project assets |

---

## Performance Comparison

| Refresh Type | LK ADT Conversion | CSV Extraction | Overlay Generation | Azeroth Time (est) |
|-------------|-------------------|----------------|--------------------|--------------------|
| None (cache hit) | ⏭️ Skip | ⏭️ Skip | ⏭️ Skip | ~30s |
| `-RefreshOverlays` | ⏭️ Skip | ⏭️ Skip | ✅ Run | ~2 min |
| `-RefreshAnalysis` | ⏭️ Skip | ✅ Run | ✅ Run | ~5 min |
| `-RefreshCache` | ✅ Run | ✅ Run | ✅ Run | ~15 min |

*Times approximate for Azeroth (755 tiles)*
