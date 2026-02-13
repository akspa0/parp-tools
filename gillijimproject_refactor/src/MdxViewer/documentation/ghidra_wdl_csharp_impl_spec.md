# WDL C# Implementation Spec (Drop-in) — Based on Ghidra Findings

Scope: practical parser + runtime mesh path matching WoW Alpha 0.5.3 client behavior.

Related analysis doc: [ghidra_wdl_read_render_report.md](ghidra_wdl_read_render_report.md)

---

## 1) File Contract (What to Parse)

Expected chunk flow used by client:

1. `MVER` chunk
2. `uint32 version` (must be `0x12`)
3. `MAOF` chunk
4. `uint32 offsets[4096]` (`64 x 64` table, absolute file offsets)
5. For each nonzero offset: `MARE` chunk with `int16 heights[545]`

### Constants

```csharp
public static class WdlConstants
{
    public const uint FourCC_MVER = 0x4D564552; // "MVER"
    public const uint FourCC_MAOF = 0x4D414F46; // "MAOF"
    public const uint FourCC_MARE = 0x4D415245; // "MARE"

    public const int VersionExpected = 0x12;

    public const int GridSize = 64;
    public const int CellCount = GridSize * GridSize; // 4096

    public const int MARE_HeightCount = 545;          // 17*17 + 16*16
    public const int MARE_Bytes = MARE_HeightCount * 2; // int16[]

    public const int AreaTriangles = 16 * 16 * 4;     // 1024
    public const int AreaIndexCount = AreaTriangles * 3; // 3072 (0xC00)
}
```

---

## 2) C# Runtime Structures

```csharp
public readonly record struct WdlChunkHeader(uint FourCC, uint Size);

public sealed class WdlFile
{
    public int Version { get; init; }
    public uint[] MaofOffsets { get; init; } = new uint[WdlConstants.CellCount];
    public WdlAreaLow?[] Areas { get; init; } = new WdlAreaLow?[WdlConstants.CellCount];
}

public sealed class WdlAreaLow
{
    // Raw converted heights from MARE int16 -> float
    public float[] Heights { get; init; } = new float[WdlConstants.MARE_HeightCount];

    // Useful precomputed culling data (client computes bounds)
    public float MinZ { get; set; }
    public float MaxZ { get; set; }

    // Optional: world-space bounding center/radius or AABB
    public System.Numerics.Vector3 BoundsMin { get; set; }
    public System.Numerics.Vector3 BoundsMax { get; set; }
}
```

---

## 3) Reader Utility

```csharp
public sealed class EndianBinaryReader : IDisposable
{
    private readonly BinaryReader _reader;
    public EndianBinaryReader(Stream stream) => _reader = new BinaryReader(stream);

    public long Position { get => _reader.BaseStream.Position; set => _reader.BaseStream.Position = value; }
    public long Length => _reader.BaseStream.Length;

    public uint ReadUInt32() => _reader.ReadUInt32();
    public int ReadInt32() => _reader.ReadInt32();
    public short ReadInt16() => _reader.ReadInt16();
    public byte[] ReadBytes(int count) => _reader.ReadBytes(count);

    public WdlChunkHeader ReadChunkHeader() => new(ReadUInt32(), ReadUInt32());

    public void Dispose() => _reader.Dispose();
}
```

---

## 4) Strict Parse Logic (Client-Matching)

```csharp
public static class WdlReader
{
    public static WdlFile Read(string filePath)
    {
        using var fs = File.OpenRead(filePath);
        using var br = new EndianBinaryReader(fs);

        var file = new WdlFile();

        // 1) MVER
        var mver = br.ReadChunkHeader();
        if (mver.FourCC != WdlConstants.FourCC_MVER)
            throw new InvalidDataException($"Expected MVER, got 0x{mver.FourCC:X8}");

        // Client reads version directly after header
        int version = br.ReadInt32();
        if (version != WdlConstants.VersionExpected)
            throw new InvalidDataException($"Unsupported WDL version {version}, expected {WdlConstants.VersionExpected}");

        // 2) MAOF
        var maof = br.ReadChunkHeader();
        if (maof.FourCC != WdlConstants.FourCC_MAOF)
            throw new InvalidDataException($"Expected MAOF, got 0x{maof.FourCC:X8}");

        // Client behavior is effectively fixed 0x4000 read
        var offsets = new uint[WdlConstants.CellCount];
        for (int i = 0; i < offsets.Length; i++)
            offsets[i] = br.ReadUInt32();

        var areas = new WdlAreaLow?[WdlConstants.CellCount];

        // 3) Per-cell MARE by absolute offsets
        for (int i = 0; i < offsets.Length; i++)
        {
            uint offset = offsets[i];
            if (offset == 0)
                continue;

            if (offset >= br.Length)
                continue; // robust fallback; client logs/asserts

            br.Position = offset;

            var mare = br.ReadChunkHeader();
            if (mare.FourCC != WdlConstants.FourCC_MARE)
                continue; // robust fallback; client logs/asserts

            // Client reads 545 int16 heights
            var area = new WdlAreaLow();
            float minZ = float.PositiveInfinity;
            float maxZ = float.NegativeInfinity;

            for (int h = 0; h < WdlConstants.MARE_HeightCount; h++)
            {
                float z = br.ReadInt16();
                area.Heights[h] = z;
                if (z < minZ) minZ = z;
                if (z > maxZ) maxZ = z;
            }

            area.MinZ = minZ;
            area.MaxZ = maxZ;

            // Fill BoundsMin/BoundsMax after XY placement mapping in your world-space builder.
            areas[i] = area;
        }

        file = new WdlFile
        {
            Version = version,
            MaofOffsets = offsets,
            Areas = areas
        };

        return file;
    }
}
```

---

## 5) Area Geometry Generation (545 verts / 3072 indices)

Client generates low-detail vertices and fixed topology per area.

## Index expectations
- `16 x 16` cells
- `4` triangles per cell
- total `3072` indices

Because exact in-client vertex ordering comes from `CreateAreaLowDetailVertices` and `CreateAreaLowDetailIndices`, implement a deterministic layout and keep it stable between vertex/index builders.

### Recommended vertex layout (matches MARE split)
- First 289 values: outer lattice `17x17`
- Next 256 values: inner lattice `16x16` (cell centers)

```csharp
public readonly record struct WdlVertex(System.Numerics.Vector3 Position, uint Color);

public static class WdlMeshBuilder
{
    public static WdlVertex[] BuildVertices(WdlAreaLow area, int cellX, int cellY, float tileSize)
    {
        var verts = new WdlVertex[WdlConstants.MARE_HeightCount];

        // World origin of this low-area cell in your coordinate system
        float baseX = cellX * tileSize;
        float baseY = cellY * tileSize;

        // 0..288 : 17x17 corners
        int index = 0;
        for (int y = 0; y < 17; y++)
        {
            for (int x = 0; x < 17; x++)
            {
                float px = baseX + (x / 16f) * tileSize;
                float py = baseY + (y / 16f) * tileSize;
                float pz = area.Heights[index];
                verts[index] = new WdlVertex(new(px, py, pz), 0xFFFFFFFF);
                index++;
            }
        }

        // 289..544 : 16x16 centers
        for (int y = 0; y < 16; y++)
        {
            for (int x = 0; x < 16; x++)
            {
                float px = baseX + ((x + 0.5f) / 16f) * tileSize;
                float py = baseY + ((y + 0.5f) / 16f) * tileSize;
                float pz = area.Heights[index];
                verts[index] = new WdlVertex(new(px, py, pz), 0xFFFFFFFF);
                index++;
            }
        }

        return verts;
    }

    public static ushort[] BuildIndices()
    {
        var indices = new ushort[WdlConstants.AreaIndexCount];
        int k = 0;

        int Corner(int x, int y) => y * 17 + x;
        int Center(int x, int y) => 289 + y * 16 + x;

        // 4 triangles per cell around center
        for (int y = 0; y < 16; y++)
        {
            for (int x = 0; x < 16; x++)
            {
                ushort c  = (ushort)Center(x, y);
                ushort v00 = (ushort)Corner(x, y);
                ushort v10 = (ushort)Corner(x + 1, y);
                ushort v01 = (ushort)Corner(x, y + 1);
                ushort v11 = (ushort)Corner(x + 1, y + 1);

                indices[k++] = v00; indices[k++] = v10; indices[k++] = c;
                indices[k++] = v10; indices[k++] = v11; indices[k++] = c;
                indices[k++] = v11; indices[k++] = v01; indices[k++] = c;
                indices[k++] = v01; indices[k++] = v00; indices[k++] = c;
            }
        }

        return indices;
    }
}
```

---

## 6) Visibility / Culling Contract

For client-like behavior:

- Keep `WdlAreaLow?[] areaLowTable` length 4096.
- During cull pass, skip null entries.
- Use precomputed bounds (`MinZ/MaxZ` + XY extents) for frustum rejection.
- Build a visible list, then render only visible low areas.

Pseudo:

```csharp
foreach (int idx in candidateIndices)
{
    var area = areaLowTable[idx];
    if (area is null) continue;
    if (!FrustumIntersects(area.BoundsMin, area.BoundsMax)) continue;
    visible.Add((idx, area));
}
```

---

## 7) Runtime Toggle / Fallback

Client has low-detail enable gating. Mirror with a feature flag:

```csharp
if (settings.ShowLowDetailWdl)
    RenderVisibleWdlAreas();
```

If `.wdl` missing or invalid:
- keep `areaLowTable` empty/null
- continue rendering normal terrain path (no hard failure)

---

## 8) Validation Checklist

- [ ] First chunk is `MVER`
- [ ] Version == `0x12`
- [ ] Next chunk is `MAOF`
- [ ] Exactly 4096 offsets consumed
- [ ] Nonzero offsets seek to `MARE`
- [ ] Read exactly 545 int16 heights per valid MARE
- [ ] Mesh emits 545 vertices and 3072 indices
- [ ] Parser tolerates bad offsets/chunks without crashing

---

## 9) Optional Strict/Lenient Modes

Add parse mode:

- **Strict**: throw on first mismatch.
- **Lenient** (client-like runtime resilience): log and skip bad entries.

```csharp
public enum WdlParseMode { Strict, Lenient }
```

---

## 10) Integration Hook Points (MdxViewer)

Recommended minimal integration:

1. Load map -> attempt `WdlReader.Read(...)`.
2. Store in scene/map object as `WdlFile`.
3. During cull phase, evaluate `Areas[idx]` bounds.
4. During horizon pass, build/reuse area mesh and draw.
5. Keep regular terrain render unchanged.

---

If requested, I can generate concrete `WdlReader.cs` and `WdlMeshBuilder.cs` files in the project with this exact contract.