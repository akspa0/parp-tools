using System.Collections.Concurrent;
using System.Numerics;
using GillijimProject.WowFiles.Alpha;
using MdxViewer.Rendering;

namespace MdxViewer.Terrain;

/// <summary>
/// Bridges Alpha WDT/ADT/MCNK parsed data into GPU-ready <see cref="TerrainChunkData"/>.
/// Handles the Alpha-specific non-interleaved vertex layout and coordinate system.
/// Reuses existing gillijimproject-csharp parsers (WdtAlpha, AdtAlpha, McnkAlpha).
/// </summary>
/// <summary>
/// Parsed MDDF placement entry (MDX/M2 doodad placement in world).
/// </summary>
public struct MddfPlacement
{
    public int NameIndex;   // Index into WDT MDNM name table
    public int UniqueId;    // For dedup across tiles
    public Vector3 Position;
    public Vector3 Rotation; // Degrees
    public float Scale;      // 1024 = 1.0 in Alpha
}

/// <summary>
/// Parsed MODF placement entry (WMO placement in world).
/// </summary>
public struct ModfPlacement
{
    public int NameIndex;   // Index into WDT MONM name table
    public int UniqueId;    // For dedup across tiles
    public Vector3 Position;
    public Vector3 Rotation; // Degrees
    public Vector3 BoundsMin;
    public Vector3 BoundsMax;
    public ushort Flags;
}

/// <summary>
/// Result of loading a single tile — terrain chunks + per-tile placements.
/// </summary>
public class TileLoadResult
{
    public List<TerrainChunkData> Chunks { get; init; } = new();
    public List<MddfPlacement> MddfPlacements { get; init; } = new();
    public List<ModfPlacement> ModfPlacements { get; init; } = new();
}

public class AlphaTerrainAdapter : ITerrainAdapter
{
    private readonly string _wdtPath;
    private readonly WdtAlpha _wdt;
    private readonly List<int> _existingTiles;
    private readonly List<int> _adtOffsets;

    /// <summary>Texture names referenced across all loaded tiles (MTEX).</summary>
    public ConcurrentDictionary<(int tileX, int tileY), List<string>> TileTextures { get; } = new();

    /// <summary>MDX model name table from WDT MDNM.</summary>
    public IReadOnlyList<string> MdxModelNames { get; }

    /// <summary>WMO model name table from WDT MONM.</summary>
    public IReadOnlyList<string> WmoModelNames { get; }

    /// <summary>Collected MDDF placements from all loaded tiles (deduplicated by uniqueId).</summary>
    public List<MddfPlacement> MddfPlacements { get; } = new();

    /// <summary>Collected MODF placements from all loaded tiles (deduplicated by uniqueId).</summary>
    public List<ModfPlacement> ModfPlacements { get; } = new();

    // Track unique IDs to avoid duplicate placements across tiles
    private readonly object _placementLock = new();
    private readonly HashSet<int> _seenMddfIds = new();
    private readonly HashSet<int> _seenModfIds = new();

    /// <summary>WorldPosition of every loaded chunk (for diagnostics).</summary>
    public List<Vector3> LastLoadedChunkPositions { get; } = new();

    /// <summary>True if this is a WMO-only map (no terrain tiles).</summary>
    public bool IsWmoBased { get; }

    public AlphaTerrainAdapter(string wdtPath)
    {
        _wdtPath = wdtPath;
        _wdt = new WdtAlpha(wdtPath);
        _existingTiles = _wdt.GetExistingAdtsNumbers();
        _adtOffsets = _wdt.GetAdtOffsetsInMain();
        MdxModelNames = _wdt.GetMdnmFileNames();
        WmoModelNames = _wdt.GetMonmFileNames();
        IsWmoBased = _wdt.IsWmoBased;

        // For WMO-only maps, collect the WDT-level MODF placement
        if (IsWmoBased)
        {
            var wdtModf = _wdt.GetWdtModfRaw();
            if (wdtModf.Length > 0)
            {
                CollectModfPlacements(wdtModf);
                Console.WriteLine($"[TerrainAdapter] WMO-only map: {ModfPlacements.Count} WMO placements from WDT header");
            }
        }

        Console.WriteLine($"[TerrainAdapter] WDT loaded: {_existingTiles.Count} tiles, {MdxModelNames.Count} MDX names, {WmoModelNames.Count} WMO names, wmoBased={IsWmoBased}");

        // Placements are now collected per-tile in LoadTileWithPlacements() for lazy loading.
        // No upfront PreScan needed — placements stream in as tiles enter AOI.
    }

    /// <summary>
    /// Pre-scan all ADTs to collect MDDF/MODF placements without loading terrain geometry.
    /// This allows the asset manifest to be built before any tiles are loaded.
    /// </summary>
    private void PreScanPlacements()
    {
        int scanned = 0;
        foreach (int tileIdx in _existingTiles)
        {
            if (tileIdx < 0 || tileIdx >= _adtOffsets.Count || _adtOffsets[tileIdx] == 0) continue;
            try
            {
                var adt = new AdtAlpha(_wdtPath, _adtOffsets[tileIdx], tileIdx);
                CollectMddfPlacements(adt.GetMddfRaw());
                CollectModfPlacements(adt.GetModfRaw());
                scanned++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TerrainAdapter] PreScan error tile {tileIdx}: {ex.Message}");
            }
        }
        Console.WriteLine($"[TerrainAdapter] PreScan: {scanned} tiles → {MddfPlacements.Count} MDDF, {ModfPlacements.Count} MODF placements");
    }

    /// <summary>
    /// Returns the list of existing tile numbers (index = y*64+x).
    /// </summary>
    public IReadOnlyList<int> ExistingTiles => _existingTiles;

    /// <summary>
    /// Check if a tile exists at the given grid coordinates.
    /// </summary>
    public bool TileExists(int tileX, int tileY)
    {
        // Alpha WDT MAIN is column-major: index = x*64+y
        int idx = tileX * 64 + tileY;
        return idx >= 0 && idx < _adtOffsets.Count && _adtOffsets[idx] != 0;
    }

    /// <summary>
    /// Load all 256 chunks for a given tile, returning GPU-ready chunk data.
    /// Uses AdtAlpha + McnkAlpha parsers from gillijimproject-csharp.
    /// </summary>
    public List<TerrainChunkData> LoadTile(int tileX, int tileY)
    {
        var result = LoadTileWithPlacements(tileX, tileY);
        return result.Chunks;
    }

    /// <summary>
    /// Load a tile and return terrain chunks + per-tile MDDF/MODF placements.
    /// Placements are collected into the returned TileLoadResult AND into the global lists.
    /// </summary>
    public TileLoadResult LoadTileWithPlacements(int tileX, int tileY)
    {
        // Alpha WDT MAIN is column-major: index = x*64+y
        int tileIdx = tileX * 64 + tileY;
        if (tileIdx < 0 || tileIdx >= _adtOffsets.Count || _adtOffsets[tileIdx] == 0)
            return new TileLoadResult();

        // Use the existing AdtAlpha parser to get MCIN offsets and MTEX
        var adt = new AdtAlpha(_wdtPath, _adtOffsets[tileIdx], tileIdx);
        var mtexNames = adt.GetMtexTextureNames();
        TileTextures.TryAdd((tileX, tileY), mtexNames);

        var chunks = new List<TerrainChunkData>(256);

        // Use AdtAlpha's internal MCIN to get MCNK offsets (same pattern as ToAdtLk)
        var offsets = adt.GetMcnkOffsets();
        using var fs = File.OpenRead(_wdtPath);

        for (int i = 0; i < 256 && i < offsets.Count; i++)
        {
            int off = offsets[i];
            if (off <= 0) continue;

            try
            {
                var mcnk = new McnkAlpha(fs, off, headerSize: 0, adtNum: tileIdx);
                var chunkData = ExtractChunkData(mcnk, tileX, tileY, tileIdx);
                if (chunkData != null)
                    chunks.Add(chunkData);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TerrainAdapter] Error reading chunk {i} of tile ({tileX},{tileY}): {ex.Message}");
            }
        }

        // Collect MDDF/MODF placement entries from this ADT into per-tile lists (no dedup — always parse all)
        var tileMddf = new List<MddfPlacement>();
        var tileModf = new List<ModfPlacement>();
        ParseMddfEntries(adt.GetMddfRaw(), tileMddf);
        ParseModfEntries(adt.GetModfRaw(), tileModf);

        // Also add to global lists with dedup for backwards compat
        lock (_placementLock)
        {
            foreach (var p in tileMddf)
                if (_seenMddfIds.Add(p.UniqueId))
                    MddfPlacements.Add(p);
            foreach (var p in tileModf)
                if (_seenModfIds.Add(p.UniqueId))
                    ModfPlacements.Add(p);
        }

        // Diagnostic: print tile corner position for first tile loaded
        if (LastLoadedChunkPositions.Count <= 256)
        {
            float cornerX = WoWConstants.MapOrigin - tileX * WoWConstants.ChunkSize;
            float cornerY = WoWConstants.MapOrigin - tileY * WoWConstants.ChunkSize;
            Console.WriteLine($"[TerrainAdapter] Tile ({tileX},{tileY}) corner=({cornerX:F1}, {cornerY:F1})  wowY={cornerX:F1}  wowX={cornerY:F1}");
        }

        Console.WriteLine($"[TerrainAdapter] Tile ({tileX},{tileY}): {chunks.Count} chunks, {mtexNames.Count} textures, {tileMddf.Count} MDDF, {tileModf.Count} MODF");
        return new TileLoadResult { Chunks = chunks, MddfPlacements = tileMddf, ModfPlacements = tileModf };
    }

    /// <summary>
    /// Parse MDDF raw bytes into placement entries. Entry size = 36 bytes.
    /// Layout: nameIndex(4) uniqueId(4) posX(4) posZ(4) posY(4) rotX(4) rotZ(4) rotY(4) scale(2) flags(2)
    /// </summary>
    private bool _mddfDiagPrinted = false;
    private void CollectMddfPlacements(byte[] mddfData)
    {
        var temp = new List<MddfPlacement>();
        ParseMddfEntries(mddfData, temp);
        lock (_placementLock)
        {
            foreach (var p in temp)
                if (_seenMddfIds.Add(p.UniqueId))
                    MddfPlacements.Add(p);
        }
    }

    /// <summary>
    /// Parse MDDF entries into a list WITHOUT dedup. Always returns all placements in the data.
    /// </summary>
    private void ParseMddfEntries(byte[] mddfData, List<MddfPlacement> target)
    {
        const int entrySize = 36;
        for (int off = 0; off + entrySize <= mddfData.Length; off += entrySize)
        {
            int nameIdx = BitConverter.ToInt32(mddfData, off);
            int uniqueId = BitConverter.ToInt32(mddfData, off + 4);

            // Raw floats at file offsets
            float rawX = BitConverter.ToSingle(mddfData, off + 8);
            float rawZ = BitConverter.ToSingle(mddfData, off + 12); // height
            float rawY = BitConverter.ToSingle(mddfData, off + 16);
            float rotX = BitConverter.ToSingle(mddfData, off + 20);
            float rotZ = BitConverter.ToSingle(mddfData, off + 24);
            float rotY = BitConverter.ToSingle(mddfData, off + 28);
            ushort scale = BitConverter.ToUInt16(mddfData, off + 32);

            // Diagnostic: dump first 3 raw entries
            if (!_mddfDiagPrinted && MddfPlacements.Count < 3)
            {
                string name = nameIdx < MdxModelNames.Count ? Path.GetFileName(MdxModelNames[nameIdx]) : "?";
                Console.WriteLine($"[MDDF RAW] [{MddfPlacements.Count}] pos=({rawX:F2}, {rawZ:F2}, {rawY:F2}) rot=({rotX:F2}, {rotZ:F2}, {rotY:F2}) scale={scale}  model={name}");
                if (MddfPlacements.Count == 2) _mddfDiagPrinted = true;
            }

            // Convert to renderer coords: terrainX=wowY, terrainY=wowX (swap + subtract)
            target.Add(new MddfPlacement
            {
                NameIndex = nameIdx,
                UniqueId = uniqueId,
                Position = new Vector3(
                    WoWConstants.MapOrigin - rawY,
                    WoWConstants.MapOrigin - rawX,
                    rawZ),
                Rotation = new Vector3(rotX, rotY, rotZ),
                Scale = scale / 1024f
            });
        }
    }

    /// <summary>
    /// Parse MODF raw bytes into placement entries. Entry size = 64 bytes.
    /// Layout: nameIndex(4) uniqueId(4) pos(12) rot(12) bbMin(12) bbMax(12) flags(2) doodadSet(2) nameSet(2) pad(2)
    /// </summary>
    private void CollectModfPlacements(byte[] modfData)
    {
        var temp = new List<ModfPlacement>();
        ParseModfEntries(modfData, temp);
        lock (_placementLock)
        {
            foreach (var p in temp)
                if (_seenModfIds.Add(p.UniqueId))
                    ModfPlacements.Add(p);
        }
    }

    /// <summary>
    /// Parse MODF entries into a list WITHOUT dedup. Always returns all placements in the data.
    /// </summary>
    private void ParseModfEntries(byte[] modfData, List<ModfPlacement> target)
    {
        const int entrySize = 64;
        for (int off = 0; off + entrySize <= modfData.Length; off += entrySize)
        {
            int nameIdx = BitConverter.ToInt32(modfData, off);
            int uniqueId = BitConverter.ToInt32(modfData, off + 4);

            // Raw floats from file
            float rawX = BitConverter.ToSingle(modfData, off + 8);
            float rawZ = BitConverter.ToSingle(modfData, off + 12); // height
            float rawY = BitConverter.ToSingle(modfData, off + 16);
            float rotX = BitConverter.ToSingle(modfData, off + 20);
            float rotZ = BitConverter.ToSingle(modfData, off + 24);
            float rotY = BitConverter.ToSingle(modfData, off + 28);

            // Diagnostic: dump first 3 MODF raw entries
            if (ModfPlacements.Count < 3)
            {
                string mname = nameIdx < WmoModelNames.Count ? Path.GetFileName(WmoModelNames[nameIdx]) : "?";
                Console.WriteLine($"[MODF RAW] [{ModfPlacements.Count}] pos=({rawX:F2}, {rawZ:F2}, {rawY:F2}) rot=({rotX:F2}, {rotZ:F2}, {rotY:F2})  model={mname}");
            }
            float bbMinX = BitConverter.ToSingle(modfData, off + 32);
            float bbMinZ = BitConverter.ToSingle(modfData, off + 36);
            float bbMinY = BitConverter.ToSingle(modfData, off + 40);
            float bbMaxX = BitConverter.ToSingle(modfData, off + 44);
            float bbMaxZ = BitConverter.ToSingle(modfData, off + 48);
            float bbMaxY = BitConverter.ToSingle(modfData, off + 52);
            ushort flags = BitConverter.ToUInt16(modfData, off + 56);

            Vector3 position;
            Vector3 boundsMin, boundsMax;

            if (IsWmoBased)
            {
                // WMO-only maps: vertices are in WoW world coords (X, Y, Z with Z=up in file).
                // MODF file layout: pos=(X, Z, Y), bb=(X, Z, Y) — middle component is height.
                // WMO vertex file layout: (X, Y, Z) — Z is height.
                // So position = (rawX, rawY, rawZ=height) and BB = (bbX, bbY, bbZ=height).
                position = new Vector3(rawX, rawY, rawZ);
                boundsMin = new Vector3(
                    MathF.Min(bbMinX, bbMaxX), MathF.Min(bbMinY, bbMaxY), MathF.Min(bbMinZ, bbMaxZ));
                boundsMax = new Vector3(
                    MathF.Max(bbMinX, bbMaxX), MathF.Max(bbMinY, bbMaxY), MathF.Max(bbMinZ, bbMaxZ));
                if (ModfPlacements.Count < 3)
                    Console.WriteLine($"[MODF WMO-ONLY] pos=({position.X:F1},{position.Y:F1},{position.Z:F1}) bb=({boundsMin.X:F1},{boundsMin.Y:F1},{boundsMin.Z:F1})→({boundsMax.X:F1},{boundsMax.Y:F1},{boundsMax.Z:F1})  raw bb file: X({bbMinX:F1}..{bbMaxX:F1}) Z({bbMinZ:F1}..{bbMaxZ:F1}) Y({bbMinY:F1}..{bbMaxY:F1})");
            }
            else
            {
                // Normal terrain maps: convert to renderer coords
                // rendererX=MapOrigin-wowY, rendererY=MapOrigin-wowX, rendererZ=wowZ
                position = new Vector3(
                    WoWConstants.MapOrigin - rawY,
                    WoWConstants.MapOrigin - rawX,
                    rawZ);
                // Note: MapOrigin-min > MapOrigin-max, so swap min/max after conversion
                float rBBMinX = WoWConstants.MapOrigin - bbMaxY;
                float rBBMaxX = WoWConstants.MapOrigin - bbMinY;
                float rBBMinY = WoWConstants.MapOrigin - bbMaxX;
                float rBBMaxY = WoWConstants.MapOrigin - bbMinX;
                boundsMin = new Vector3(rBBMinX, rBBMinY, bbMinZ);
                boundsMax = new Vector3(rBBMaxX, rBBMaxY, bbMaxZ);
            }

            target.Add(new ModfPlacement
            {
                NameIndex = nameIdx,
                UniqueId = uniqueId,
                Position = position,
                Rotation = new Vector3(rotX, rotY, rotZ),
                BoundsMin = boundsMin,
                BoundsMax = boundsMax,
                Flags = flags
            });
        }
    }

    private TerrainChunkData? ExtractChunkData(McnkAlpha mcnk, int tileX, int tileY, int tileIdx)
    {
        int chunkX = mcnk.IndexX;
        int chunkY = mcnk.IndexY;

        // Extract heights (145 floats = 580 bytes, Alpha non-interleaved format)
        var heights = ExtractHeights(mcnk.McvtData);
        if (heights == null) return null;

        // Extract normals (145 × 3 signed bytes, Alpha non-interleaved format)
        var normals = ExtractNormals(mcnk.McnrData);

        // Extract layers from MCLY (16 bytes per layer)
        var layers = ExtractLayers(mcnk.MclyData, mcnk.NLayers);

        // Extract alpha maps from MCAL
        var alphaMaps = ExtractAlphaMaps(mcnk.McalData, mcnk.MclyData, mcnk.NLayers);

        // Extract MCSH shadow map (64×64 bits → 64×64 bytes)
        byte[]? shadowMap = ExtractShadowMap(mcnk.McshData, mcnk.McshSize);

        // Compute world position for this chunk in renderer coordinates.
        // WDT MAIN index = tileX*64+tileY (column-major).
        // Renderer coords match MODF: rendererX = MapOrigin - wowY, rendererY = MapOrigin - wowX
        float chunkSmall = WoWConstants.ChunkSize / 16f;
        float worldX = WoWConstants.MapOrigin - tileX * WoWConstants.ChunkSize - chunkY * chunkSmall;
        float worldY = WoWConstants.MapOrigin - tileY * WoWConstants.ChunkSize - chunkX * chunkSmall;

        LastLoadedChunkPositions.Add(new Vector3(worldX, worldY, 0f));

        // Extract MCLQ inline liquid data (type from MCNK header flags bits 2-5)
        var liquid = ExtractLiquid(mcnk.MclqData, mcnk.Header.Flags, tileX, tileY, chunkX, chunkY,
            new Vector3(worldX, worldY, 0f));

        return new TerrainChunkData
        {
            TileX = tileX,
            TileY = tileY,
            ChunkX = chunkX,
            ChunkY = chunkY,
            Heights = heights,
            Normals = normals,
            HoleMask = mcnk.Holes,
            Layers = layers,
            AlphaMaps = alphaMaps,
            ShadowMap = shadowMap,
            Liquid = liquid,
            WorldPosition = new Vector3(worldX, worldY, 0f)
        };
    }

    /// <summary>
    /// Extract 145 height floats from Alpha MCVT data, reordering from non-interleaved to interleaved.
    /// Alpha format: 81 outer vertices first, then 64 inner vertices.
    /// Interleaved format: row of 9 outer, row of 8 inner, alternating for 17 rows.
    /// </summary>
    private static float[]? ExtractHeights(byte[] mcvtData)
    {
        if (mcvtData == null || mcvtData.Length < 580) return null;

        var heights = new float[145];
        int destIdx = 0;

        // Alpha layout: [81 outer floats][64 inner floats]
        // Interleaved layout: 9 outer, 8 inner, 9 outer, 8 inner, ... 9 outer (17 rows total)
        for (int row = 0; row < 17; row++)
        {
            if (row % 2 == 0)
            {
                // Outer row (9 vertices)
                int outerRow = row / 2;
                for (int col = 0; col < 9; col++)
                {
                    int srcIdx = (outerRow * 9 + col) * 4; // Alpha: all 81 outer first
                    heights[destIdx++] = BitConverter.ToSingle(mcvtData, srcIdx);
                }
            }
            else
            {
                // Inner row (8 vertices)
                int innerRow = row / 2;
                for (int col = 0; col < 8; col++)
                {
                    int srcIdx = (81 + innerRow * 8 + col) * 4; // Alpha: 64 inner after 81 outer
                    heights[destIdx++] = BitConverter.ToSingle(mcvtData, srcIdx);
                }
            }
        }

        return heights;
    }

    /// <summary>
    /// Extract 145 normals from Alpha MCNR data, reordering from non-interleaved to interleaved.
    /// Each normal is 3 signed bytes (X, Z, Y in WoW coords), normalized to [-1,1].
    /// Alpha format: 81 outer normals first (243 bytes), then 64 inner normals (192 bytes).
    /// </summary>
    private static Vector3[] ExtractNormals(byte[] mcnrData)
    {
        var normals = new Vector3[145];

        if (mcnrData == null || mcnrData.Length < 435) // 145 * 3 = 435 minimum
        {
            // Default to up-facing normals
            for (int i = 0; i < 145; i++)
                normals[i] = Vector3.UnitZ;
            return normals;
        }

        int destIdx = 0;

        for (int row = 0; row < 17; row++)
        {
            if (row % 2 == 0)
            {
                // Outer row (9 normals)
                int outerRow = row / 2;
                for (int col = 0; col < 9; col++)
                {
                    int srcIdx = (outerRow * 9 + col) * 3;
                    normals[destIdx++] = DecodeNormal(mcnrData, srcIdx);
                }
            }
            else
            {
                // Inner row (8 normals)
                int innerRow = row / 2;
                for (int col = 0; col < 8; col++)
                {
                    int srcIdx = (81 * 3) + (innerRow * 8 + col) * 3;
                    normals[destIdx++] = DecodeNormal(mcnrData, srcIdx);
                }
            }
        }

        return normals;
    }

    private static Vector3 DecodeNormal(byte[] data, int offset)
    {
        if (offset + 2 >= data.Length) return Vector3.UnitZ;

        // MCNR stores normals as signed bytes: X, Z, Y (WoW convention)
        float nx = (sbyte)data[offset] / 127f;
        float nz = (sbyte)data[offset + 1] / 127f;
        float ny = (sbyte)data[offset + 2] / 127f;

        // Return as (X, Y, Z) in our coordinate system
        var n = new Vector3(nx, ny, nz);
        float len = n.Length();
        return len > 0.001f ? n / len : Vector3.UnitZ;
    }

    private static TerrainLayer[] ExtractLayers(byte[] mclyData, int nLayers)
    {
        if (mclyData == null || mclyData.Length < 16 || nLayers <= 0)
            return Array.Empty<TerrainLayer>();

        int count = Math.Min(nLayers, 4);
        count = Math.Min(count, mclyData.Length / 16);

        var layers = new TerrainLayer[count];
        for (int i = 0; i < count; i++)
        {
            int off = i * 16;
            layers[i] = new TerrainLayer
            {
                TextureIndex = BitConverter.ToInt32(mclyData, off),
                Flags = BitConverter.ToUInt32(mclyData, off + 4),
                AlphaOffset = BitConverter.ToUInt32(mclyData, off + 8),
                EffectId = BitConverter.ToUInt32(mclyData, off + 12)
            };
        }

        return layers;
    }

    /// <summary>
    /// Extract alpha maps from MCAL data. Layer 0 is always fully opaque (no alpha map).
    /// Each alpha map is 64×64 bytes (4096 bytes) for 8-bit, or 32×64 (2048 bytes) for 4-bit.
    /// </summary>
    private static Dictionary<int, byte[]> ExtractAlphaMaps(byte[] mcalData, byte[] mclyData, int nLayers)
    {
        var maps = new Dictionary<int, byte[]>();
        if (mcalData == null || mcalData.Length == 0 || nLayers <= 1)
            return maps;

        int offset = 0;
        for (int layer = 1; layer < nLayers && layer < 4; layer++)
        {
            if (layer * 16 > mclyData.Length) break;

            uint flags = BitConverter.ToUInt32(mclyData, layer * 16 + 4);
            bool isCompressed = (flags & 0x200) != 0;

            // Alpha 0.5.3 typically uses uncompressed 4-bit alpha (2048 bytes = 64×64 / 2)
            int alphaSize = isCompressed ? 4096 : 2048;
            if (offset + alphaSize > mcalData.Length)
            {
                // Try remaining data
                alphaSize = mcalData.Length - offset;
                if (alphaSize <= 0) break;
            }

            byte[] alpha;
            if (alphaSize == 2048)
            {
                // 4-bit alpha: expand to 8-bit (64×64)
                alpha = new byte[4096];
                for (int j = 0; j < Math.Min(2048, alphaSize); j++)
                {
                    byte packed = mcalData[offset + j];
                    alpha[j * 2] = (byte)((packed & 0x0F) * 17);     // low nibble → 0-255
                    alpha[j * 2 + 1] = (byte)((packed >> 4) * 17);   // high nibble → 0-255
                }
            }
            else
            {
                // 8-bit alpha: copy directly
                alpha = new byte[alphaSize];
                Array.Copy(mcalData, offset, alpha, 0, alphaSize);
            }

            maps[layer] = alpha;
            offset += alphaSize;
        }

        return maps;
    }

    /// <summary>
    /// Extract MCLQ inline liquid data from raw bytes (Ghidra-verified Alpha 0.5.3 format).
    /// NO FourCC header — data is inline within MCNK, referenced by ofsLiquid.
    /// Each instance: 8 (min/max) + 648 (81 verts × 8) + 64 (16 tile floats) + 84 (flows) = 804 bytes.
    /// Liquid type determined from MCNK header flags bits 2-5.
    /// Up to 4 liquid instances per chunk (one per type).
    /// Returns the first valid liquid instance found, or null.
    /// </summary>
    private static LiquidChunkData? ExtractLiquid(byte[] mclqData, int mcnkFlags, int tileX, int tileY,
        int chunkX, int chunkY, Vector3 worldPos)
    {
        if (mclqData == null || mclqData.Length < LiquidChunkData.InstanceSize)
            return null;

        // Determine liquid types from MCNK flags bits 2-5
        // Bit 2 (0x04) = Water, Bit 3 (0x08) = Ocean, Bit 4 (0x10) = Magma, Bit 5 (0x20) = Slime
        bool hasWater = (mcnkFlags & 0x04) != 0;
        bool hasOcean = (mcnkFlags & 0x08) != 0;
        bool hasMagma = (mcnkFlags & 0x10) != 0;
        bool hasSlime = (mcnkFlags & 0x20) != 0;

        if (!hasWater && !hasOcean && !hasMagma && !hasSlime)
            return null;

        // Determine which liquid type this is (first set bit)
        LiquidType liquidType;
        if (hasWater) liquidType = LiquidType.Water;
        else if (hasOcean) liquidType = LiquidType.Ocean;
        else if (hasMagma) liquidType = LiquidType.Magma;
        else liquidType = LiquidType.Slime;

        // Parse the first 804-byte liquid instance
        return ParseLiquidInstance(mclqData, 0, liquidType, tileX, tileY, chunkX, chunkY, worldPos);
    }

    /// <summary>
    /// Parse a single 804-byte MCLQ inline data instance.
    /// Layout: float minH, float maxH, {float height, uint32 data}[81], float tiles[16], uint32 nFlowvs, SWFlowv[2]
    /// </summary>
    private static LiquidChunkData? ParseLiquidInstance(byte[] data, int offset, LiquidType type,
        int tileX, int tileY, int chunkX, int chunkY, Vector3 worldPos)
    {
        if (offset + LiquidChunkData.InstanceSize > data.Length)
            return null;

        // Read min/max height range (8 bytes)
        float minHeight = BitConverter.ToSingle(data, offset); offset += 4;
        float maxHeight = BitConverter.ToSingle(data, offset); offset += 4;

        // Sanity check: invalid height range suggests bad data
        if (float.IsNaN(minHeight) || float.IsNaN(maxHeight) ||
            minHeight < -5000f || maxHeight > 5000f)
            return null;

        // Read 81 vertices (9×9 grid, 8 bytes each: float height + uint32 data)
        var heights = new float[81];
        var vertexData = new uint[81];
        for (int i = 0; i < 81; i++)
        {
            heights[i] = BitConverter.ToSingle(data, offset); offset += 4;
            vertexData[i] = BitConverter.ToUInt32(data, offset); offset += 4;
        }

        // Read 16 tile floats (4×4 grid, 64 bytes)
        var tileGrid = new float[16];
        for (int i = 0; i < 16; i++)
        {
            tileGrid[i] = BitConverter.ToSingle(data, offset); offset += 4;
        }

        // Skip flow data (4 bytes nFlowvs + 80 bytes flowvs = 84 bytes)
        // offset += 84; // Not needed, we don't use flow data for rendering

        return new LiquidChunkData
        {
            MinHeight = minHeight,
            MaxHeight = maxHeight,
            Heights = heights,
            VertexData = vertexData,
            TileGrid = tileGrid,
            Type = type,
            WorldPosition = worldPos,
            TileX = tileX,
            TileY = tileY,
            ChunkX = chunkX,
            ChunkY = chunkY
        };
    }

    /// <summary>
    /// Extract MCSH shadow map: 64×64 bits (512 bytes = 64 rows × 8 bytes/row).
    /// Each bit represents one cell: 1=shadowed, 0=lit.
    /// Expands to 64×64 bytes (0=lit, 255=shadowed) for GPU upload as R8 texture.
    /// </summary>
    private static byte[]? ExtractShadowMap(byte[] mcshData, int mcshSize)
    {
        if (mcshData == null || mcshData.Length == 0 || mcshSize <= 0)
            return null;

        // MCSH is 64 rows × 8 bytes/row = 512 bytes (64×64 bits)
        int rows = Math.Min(64, mcshSize / 8);
        if (rows == 0) return null;

        var shadow = new byte[64 * 64];
        for (int y = 0; y < rows; y++)
        {
            int srcRow = y * 8;
            for (int byteIdx = 0; byteIdx < 8 && srcRow + byteIdx < mcshData.Length; byteIdx++)
            {
                byte bits = mcshData[srcRow + byteIdx];
                for (int bit = 0; bit < 8; bit++)
                {
                    int x = byteIdx * 8 + bit;
                    if (x < 64)
                        shadow[y * 64 + x] = (byte)(((bits >> bit) & 1) * 255);
                }
            }
        }

        return shadow;
    }
}
