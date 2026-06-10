# Alpha WDT Conversion - Quick Reference

## ⚠️ Current Scope: Terrain-Only

**Working**: Dungeon maps with top-level WMO data (e.g., RazorfenDowns)  
**Not Working**: Outdoor maps with per-tile WMO/M2 data (e.g., Kalidar)

See `Known-Limitations.md` for details.

## 🔴 CRITICAL: Must-Fix Issues

### 1. WMO Name Counting (CRASH FIX)

**Problem**: Client crashes in `CMap::CreateMapObjDef` if count is wrong

**Solution**: Alpha client counts by splitting on null terminators

```
"name\0" → ["name", ""] = 2 parts (not 1!)

if (wmoNames.Count > 0)
    nMapObjNames = wmoNames.Count + 1;  // +1 for trailing empty
else
    nMapObjNames = 0;
```

### 2. MHDR.offsInfo (MISSING FIELD)

**Problem**: Client can't find MCIN chunk

**Solution**: Set to 0 (MCIN immediately after MHDR.data)

```c
mhdr.offsInfo = 0;  // MCIN is first sub-chunk
```

### 3. MCNK.Radius (CULLING ISSUE)

**Problem**: Was 0, should be calculated from terrain

**Solution**: Calculate bounding sphere from MCVT heights

```
minH = min(mcvtHeights)
maxH = max(mcvtHeights)
heightRange = maxH - minH
horizontalRadius = 23.57  // chunk diagonal / 2
radius = sqrt(horizontalRadius² + (heightRange/2)²)
```

### 4. MCNK.MclqOffset (ACCESS_VIOLATION)

**Problem**: Was 0, client reads from wrong location

**Solution**: Point to end of sub-chunks

```
mclqOffset = mcvtSize + mcnrSize + mclySize + mcrfSize + mcshSize + mcalSize
```

### 5. MONM Chunk (CRASH FIX)

**Problem**: Was empty, client needs actual WMO names

**Solution**: Read from source LK WDT MWMO chunk

```
wmoNames = ReadMWMO(lkWdtPath)
monmData = BuildNullTerminatedStrings(wmoNames)
```

---

## 📋 Structure Checklist

### Top-Level WDT
- [ ] MVER version = 18
- [ ] MPHD.nMapObjNames uses split-by-null counting (+1)
- [ ] MPHD.offsMapObjNames points to MONM
- [ ] MAIN entries: offset = MHDR letters, size = MHDR-to-first-MCNK
- [ ] MONM contains actual WMO names (not empty!)
- [ ] MODF chunk exists (even if empty)

### Per-Tile MHDR
- [ ] offsInfo = 0 (points to MCIN)
- [ ] All offsets relative to MHDR.data (+8 from MHDR letters)
- [ ] offsTex, offsDoo, offsMob point to correct chunks

### Per-Chunk MCNK
- [ ] radius calculated from MCVT
- [ ] mclqOffset = end of sub-chunks
- [ ] mcvtOffset = 0 (immediately after header)
- [ ] All sub-chunk offsets relative to MCNK.data

### Sub-Chunks
- [ ] MCVT: NO header, 580 bytes raw data
- [ ] MCNR: NO header, 448 bytes raw data
- [ ] MCLY: HAS header (letters + size)
- [ ] MCRF: HAS header
- [ ] MCSH: HAS header
- [ ] MCAL: HAS header

---

## 🐛 Common Bugs

| Symptom | Cause | Fix |
|---------|-------|-----|
| Crash in CreateMapObjDef | Wrong nMapObjNames count | Add +1 for trailing empty |
| Can't find MCIN | Missing offsInfo | Set to 0 |
| Terrain doesn't render | Radius = 0 | Calculate from heights |
| ACCESS_VIOLATION | MclqOffset = 0 | Point to end of sub-chunks |
| Crash reading WMO data | Empty MONM | Read from source MWMO |
| Wrong tile data | MAIN.size is total | Use MHDR-to-first-MCNK |

---

## 🔍 Verification

### Quick Test
```bash
# Pack a map
./converter pack-monolithic --lk-wdt source.wdt --map MapName

# Compare with real Alpha
./converter compare-alpha --reference real_alpha.wdt --test packed.wdt

# Check first difference - should be in MAIN or later (not MPHD!)
```

### In-Game Test
1. Copy packed WDT to Alpha client
2. Load map in 0.5.3 client
3. Verify:
   - ✅ No crash on load
   - ✅ Terrain visible
   - ✅ Can move around
   - ✅ No ACCESS_VIOLATION

---

## 📐 Key Formulas

### WMO Count
```
nMapObjNames = (wmoNames.Count > 0) ? wmoNames.Count + 1 : 0
```

### Radius
```
radius = sqrt(23.57² + ((maxHeight - minHeight) / 2)²)
```

### MAIN Entry Size
```
size = offsetToFirstMCNK - offsetToMHDR
```

### MCNK Sub-Chunk Offset
```
offset = previousOffset + previousSize + (previousSize % 2)  // Align to 2 bytes
```

---

## 🎯 Implementation Priority

1. **CRITICAL** - WMO name counting (prevents crash)
2. **CRITICAL** - MONM data (prevents crash)
3. **CRITICAL** - MclqOffset (prevents ACCESS_VIOLATION)
4. **HIGH** - MHDR.offsInfo (prevents chunk read failure)
5. **MEDIUM** - Radius calculation (prevents culling issues)
6. **LOW** - Other fields (cosmetic or unused)

---

## 📚 Full Documentation

See `Alpha-WDT-Conversion-Spec.md` for complete technical specification.
