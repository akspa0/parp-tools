# Library Reference Guide - Learning from Existing Tools

**Goal**: Leverage existing WoW data libraries to guide our pre-CASC viewer implementation.

**Focus**: Alpha/Vanilla/BC/WotLK (pre-CASC versions only)  
**Future**: CASC support via CascLib (deferred)

---

## 📚 Available Libraries

### **1. Warcraft.NET** (`lib/Warcraft.NET`)

**Purpose**: .NET library for reading WoW file formats

**What We Can Learn**:
- ADT/WDT chunk parsing patterns
- MDDF/MODF structure definitions
- Flags enum definitions
- Coordinate system transformations
- Chunk validation logic

**Key Files to Study**:
- `Warcraft.NET/Files/ADT/` - ADT parsing
- `Warcraft.NET/Files/WDT/` - WDT parsing
- `Warcraft.NET/Files/Structures/` - Data structures
- `Warcraft.NET/Files/Enums/` - Flag definitions

**How to Use**:
```csharp
// Reference Warcraft.NET structures when implementing our parsers
// Example: MDDF flags from Warcraft.NET
namespace Warcraft.NET.Files.ADT.Chunks
{
    public enum MDDFFlags : ushort
    {
        // Study these flag definitions!
        // Use them to populate our flag decoder
    }
}
```

**Action Items**:
- [ ] Review MDDF/MODF flag enums
- [ ] Compare chunk parsing approaches
- [ ] Study coordinate transformations
- [ ] Reference structure layouts

---

### **2. wow.tools.local** (`lib/wow.tools.local`)

**Purpose**: Local version of wow.tools for offline file analysis

**What We Can Learn**:
- File format documentation
- Chunk structure examples
- Parsing strategies
- Validation logic

**Key Components**:
- Database schema definitions
- File format parsers
- Chunk readers

**How to Use**:
- Cross-reference our parsing with wow.tools.local
- Validate our chunk interpretations
- Learn from their error handling

**Action Items**:
- [ ] Review ADT/WDT parsing logic
- [ ] Study chunk validation approaches
- [ ] Reference error handling patterns

---

### **3. wow.export** (`lib/wow.export`)

**Purpose**: WoW model/texture export tool

**What We Can Learn**:
- M2 model parsing
- WMO object parsing
- Texture extraction
- Asset path resolution

**Key Features**:
- Model export pipeline
- Material/texture handling
- LOD management

**How to Use**:
- Reference for model path interpretation
- Learn asset type detection
- Study texture coordinate handling

**Action Items**:
- [ ] Review M2/WMO parsing
- [ ] Study asset path resolution
- [ ] Reference texture handling (future feature)

---

### **4. CascLib** (`lib/wow.tools.local/CascLib`) - **FUTURE ONLY**

**Purpose**: CASC (Content Addressable Storage Container) support for modern WoW

**Status**: ⚠️ **DEFERRED** - Not current priority

**When to Use**: After pre-CASC viewer is complete and stable

**Scope for Future**:
- Legion+ map support
- Modern client compatibility
- Encrypted file handling

**Current Action**: 🚫 **DO NOT IMPLEMENT** - Focus on pre-CASC only

---

## 🎯 Pre-CASC Focus (Current Work)

### **Target Versions**:
- ✅ **Alpha** (0.5.3 - 0.6.0)
- ✅ **Vanilla** (1.x)
- ✅ **Burning Crusade** (2.x)
- ✅ **Wrath of the Lich King** (3.x)
- ⚠️ **Cataclysm** (4.x) - Partial support (pre-CASC 4.0.x only)

### **File Formats We Need**:
- ✅ WDT (World Data Table)
- ✅ ADT (Area Data Tiles)
- ✅ M2 (Models) - metadata only for viewer
- ✅ WMO (World Map Objects) - metadata only for viewer
- ✅ DBC (Database Client files)
- ❌ BLP (textures) - Future enhancement
- ❌ Bone/animation data - Not needed for viewer

---

## 🔍 How to Leverage Libraries

### **Step 1: Flag Definitions**

**Reference**: `Warcraft.NET/Files/ADT/Chunks/MDDF.cs` (or similar)

```csharp
// Our implementation - populate from Warcraft.NET study
public enum MDDFFlags : ushort
{
    // TODO: Extract from Warcraft.NET
    Unknown1 = 0x0001,
    Unknown2 = 0x0002,
    // ... etc
}

// Use in our flag decoder
public static string DecodeMDDFFlags(ushort flags)
{
    var parts = new List<string>();
    if ((flags & (ushort)MDDFFlags.Unknown1) != 0)
        parts.Add("Flag 0x0001");
    // ... etc
    return string.Join(", ", parts);
}
```

---

### **Step 2: Chunk Parsing**

**Reference**: `Warcraft.NET/Files/ADT/ADT.cs`

```csharp
// Learn from Warcraft.NET chunk reading pattern
public class AdtParser
{
    public AdtData Parse(Stream stream)
    {
        using var reader = new BinaryReader(stream);
        var data = new AdtData();
        
        while (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            // Pattern from Warcraft.NET
            var chunkId = new string(reader.ReadChars(4).Reverse().ToArray());
            var chunkSize = reader.ReadUInt32();
            var chunkData = reader.ReadBytes((int)chunkSize);
            
            switch (chunkId)
            {
                case "MVER": data.Version = ParseMVER(chunkData); break;
                case "MHDR": data.Header = ParseMHDR(chunkData); break;
                // ... following Warcraft.NET pattern
            }
        }
        
        return data;
    }
}
```

---

### **Step 3: Coordinate Transformations**

**Reference**: `Warcraft.NET` coordinate handling

```csharp
// Study Warcraft.NET's approach to coordinate systems
// Alpha uses different coords than LK
// Learn from library, adapt for our viewer

public static Vector3 TransformAlphaToViewer(Vector3 position)
{
    // Based on Warcraft.NET transformations
    // Adapt for Leaflet coordinate system
    return new Vector3(
        position.X,  // May need adjustments
        position.Y,
        position.Z
    );
}
```

---

## 📋 Study Checklist

### **Immediate (Before Implementation)**

**Flags Research**:
- [ ] Open `Warcraft.NET/Files/ADT/Chunks/MDDF.cs`
- [ ] Extract MDDF flag enum definitions
- [ ] Open `Warcraft.NET/Files/ADT/Chunks/MODF.cs`
- [ ] Extract MODF flag enum definitions
- [ ] Document all flags in `ADT_FLAGS_REFERENCE.md`
- [ ] Create color/shape mapping based on flags

**Chunk Structure**:
- [ ] Review `Warcraft.NET/Files/ADT/ADT.cs`
- [ ] Study MHDR chunk parsing
- [ ] Study MCNK chunk parsing
- [ ] Compare with our current implementation
- [ ] Identify any missing fields we should track

**Data Validation**:
- [ ] Review `wow.tools.local` validation logic
- [ ] Learn error handling patterns
- [ ] Implement similar validation in our code

---

### **Near-Term (During Development)**

**Model Metadata**:
- [ ] Review `wow.export` M2 parsing
- [ ] Learn model name extraction
- [ ] Study model flags/properties
- [ ] Implement metadata extraction for viewer

**WMO Metadata**:
- [ ] Review `wow.export` WMO parsing
- [ ] Study WMO placement transformations
- [ ] Learn WMO flags/properties

---

### **Future (Post-MVP)**

**CASC Support** (Deferred):
- [ ] Study `CascLib` API
- [ ] Plan integration strategy
- [ ] Implement Legion+ support
- [ ] Test with modern clients

**Texture Support** (Deferred):
- [ ] Review BLP parsing in libraries
- [ ] Plan texture layer visualization
- [ ] Implement minimap texture overlay

---

## 🚫 Out of Scope (Current Phase)

**Do NOT implement**:
- ❌ CASC file reading
- ❌ Legion+ map support
- ❌ Encrypted file handling
- ❌ Modern client formats
- ❌ BLP texture rendering (future enhancement)
- ❌ 3D model rendering (viewer is 2D map only)
- ❌ Animation data
- ❌ Particle systems (beyond metadata)

---

## 📖 Documentation Strategy

### **Create Reference Docs**:

**1. `ADT_FLAGS_REFERENCE.md`** (from Warcraft.NET)
```markdown
# ADT Flag Definitions

## MDDF Flags (from Warcraft.NET)
- 0x0001: [Name from library] - [Description]
- 0x0002: [Name from library] - [Description]
...

## MODF Flags (from Warcraft.NET)
- 0x0001: [Name from library] - [Description]
...
```

**2. `CHUNK_STRUCTURES.md`** (from libraries)
```markdown
# Chunk Structure Reference

## MDDF (M2 Doodad Definition)
Based on Warcraft.NET

Offset | Type   | Name       | Description
-------|--------|------------|-------------
0      | uint   | NameID     | Index into MMID
4      | uint   | UniqueID   | Unique object ID
8      | Vector3| Position   | X, Y, Z coordinates
20     | Vector3| Rotation   | Rotation angles
32     | ushort | Scale      | Scale factor
34     | ushort | Flags      | See MDDF_FLAGS
```

**3. `COORDINATE_SYSTEMS.md`** (from library transformations)
```markdown
# Coordinate System Transformations

## Alpha → Viewer
Based on Warcraft.NET coordinate handling

## WotLK → Viewer
Differences from Alpha...
```

---

## ✅ Success Criteria

### Research Phase
- [ ] All MDDF/MODF flags documented from Warcraft.NET
- [ ] Chunk structures validated against libraries
- [ ] Coordinate transformations understood
- [ ] Error handling patterns identified

### Implementation Phase
- [ ] Flags parsed and displayed correctly
- [ ] Chunks read following library patterns
- [ ] Coordinates match library transformations
- [ ] Validation logic similar to libraries

### Quality Checks
- [ ] Our parser output matches library output
- [ ] Flag interpretations align with libraries
- [ ] Coordinate systems consistent with libraries
- [ ] Error handling comprehensive like libraries

---

## 🎯 Immediate Action

**Before writing any code**:
1. Open `lib/Warcraft.NET/Files/ADT/Chunks/MDDF.cs`
2. Extract flag enum
3. Document in `ADT_FLAGS_REFERENCE.md`
4. Repeat for MODF
5. Cross-reference with `ADT_v18.md`

**Then proceed with implementation using library patterns as guide!** 🦀

---

**These libraries are our blueprint - learn from them, don't reinvent!**
