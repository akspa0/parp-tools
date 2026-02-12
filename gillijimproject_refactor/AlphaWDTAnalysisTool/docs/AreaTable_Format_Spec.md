# Alpha AreaTable Format Specification

**Ghidra-verified from 0.5.3 `wowclient.exe` (Build 3368)**

---

## DBC Record Structure

**File**: `DBFilesClient\AreaTable.dbc`  
**Record Size**: **88 bytes (0x58)**

Based on decompilation of `LoadAreaTable` @ `00666c30`:

```c
struct AreaTableRec {
  uint32_t m_ID;           // 0x00 - Record ID (primary key)
  uint32_t m_ContinentID;  // 0x04 - Map/Continent ID
  uint32_t m_AreaNumber;   // 0x08 - PACKED: (area << 16) | subArea
  // ... remaining fields (name, flags, etc.)
  // Total: 0x58 (88) bytes
};
```

---

## AreaNumber Packing (CRITICAL)

The `m_AreaNumber` field is a **packed uint32**:

```c
uint32_t m_AreaNumber;
uint16_t area    = (m_AreaNumber >> 16) & 0xFFFF;  // Upper 16 bits
uint16_t subArea = m_AreaNumber & 0xFFFF;          // Lower 16 bits
```

### Example Values
| Raw m_AreaNumber | area (zone) | subArea |
|:---|:---|:---|
| `0x00010000` | 1 | 0 |
| `0x000C0001` | 12 | 1 |
| `0x00120005` | 18 | 5 |

---

## Hash Table Storage

Areas are loaded into `AREAHASHOBJECT` hash table at runtime:

```c
struct AREAHASHKEY {
  uint32_t continent;    // ContinentID from DBC
  uint32_t area;         // Upper 16 bits of m_AreaNumber
  uint32_t subArea;      // Lower 16 bits of m_AreaNumber
};

struct AREAHASHOBJECT {
  uint32_t hashKey;      // Derived from area
  // ... link pointers ...
  AREAHASHKEY key;       // 0x14: continent, 0x18: area, 0x1C: subArea
  uint32_t continent;    // Copy of ContinentID
  uint32_t area;         // Copy of area
  uint32_t subArea;      // Copy of subArea
  AreaTableRec* rec;     // Pointer to DBC record
};
```

---

## MCNK Area ID Field

**Location in Alpha MCNK**: `mcnk_header + 0x38` (`Unknown3`)

The MCNK stores area as packed `zone<<16|sub`:

```c
// Reading area from Alpha MCNK:
uint32_t packedArea = *(uint32_t*)(mcnk + 8 + 0x38);
uint16_t zone = packedArea >> 16;
uint16_t sub  = packedArea & 0xFFFF;
```

---

## Runtime Area Lookup

`CMap::QueryAreaId(float x, float y)` @ `00687af0`:

1. Convert world coords to tile index
2. Look up in 64x64 `areaTable` array
3. Return AreaID from `CMapArea->texIdTable[0x18]` (offset 0x60)

```c
// Coord to tile index:
uint tileX = (uint)ROUND(scale * (x - offset));
uint tileY = (uint)ROUND(scale * (y - offset));

// Array lookup:
CMapArea** tile = areaTable[(tileY >> 4) * 64 + (tileX >> 4)];
if (tile == NULL) return 0;

CMapArea* chunk = tile[(tileY & 0xF) * 16 + (tileX & 0xF) + 0x4A7];
return chunk->texIdTable[0x18];
```

---

## Crosswalk Matching Implications

When matching Alpha AreaIDs to LK (3.3.5a):

1. **Parse correctly**: Split the packed `m_AreaNumber` into `zone` and `subArea`
2. **Match by composite key**: `(ContinentID, zone, subArea)` - NOT just the raw uint32
3. **Handle zone=0**: Some areas have zone=0 (world areas), subArea carries the ID
4. **Name fallback**: Use `AreaName_Lang` for fuzzy matching when IDs don't exist

---

## LK (3.3.5a) AreaTable Differences

LK `AreaTable.dbc` uses **explicit `ParentAreaID`** field instead of packed zone/sub:

```c
struct AreaTableRec_LK {
  uint32_t ID;
  uint32_t ContinentID;
  uint32_t ParentAreaID;   // Replaces packed zone/sub
  uint32_t AreaBit;
  uint32_t Flags;
  // ... etc
};
```

**Matching Rule**: Alpha's `(zone, subArea)` → LK's `(ParentAreaID, ID)` hierarchy
