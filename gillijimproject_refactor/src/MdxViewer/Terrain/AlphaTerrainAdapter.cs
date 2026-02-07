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
    public Vector3 Position;
    public Vector3 Rotation; // Degrees
    public Vector3 BoundsMin;
    public Vector3 BoundsMax;
    public ushort Flags;
}

public class AlphaTerrainAdapter
{
    private readonly string _wdtPath;
    private readonly WdtAlpha _wdt;
    private readonly List<int> _existingTiles;
    private readonly List<int> _adtOffsets;

    /// <summary>Texture names referenced across all loaded tiles (MTEX).</summary>
    public Dictionary<(int tileX, int tileY), List<string>> TileTextures { get; } = new();

    /// <summary>MDX model name table from WDT MDNM.</summary>
    public IReadOnlyList<string> MdxModelNames { get; }

    /// <summary>WMO model name table from WDT MONM.</summary>
    public IReadOnlyList<string> WmoModelNames { get; }

    /// <summary>Collected MDDF placements from all loaded tiles (deduplicated by uniqueId).</summary>
    public List<MddfPlacement> MddfPlacements { get; } = new();

    /// <summary>Collected MODF placements from all loaded tiles (deduplicated by uniqueId).</summary>
    public List<ModfPlacement> ModfPlacements { get; } = new();

    // Track unique IDs to avoid duplicate placements across tiles
    private readonly HashSet<int> _seenMddfIds = new();
    private readonly HashSet<int> _seenModfIds = new();

    public AlphaTerrainAdapter(string wdtPath)
    {
        _wdtPath = wdtPath;
        _wdt = new WdtAlpha(wdtPath);
        _existingTiles = _wdt.GetExistingAdtsNumbers();
        _adtOffsets = _wdt.GetAdtOffsetsInMain();
        MdxModelNames = _wdt.GetMdnmFileNames();
        WmoModelNames = _wdt.GetMonmFileNames();

        Console.WriteLine($"[TerrainAdapter] WDT loaded: {_existingTiles.Count} tiles, {MdxModelNames.Count} MDX names, {WmoModelNames.Count} WMO names");
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
        // Alpha WDT MAIN is column-major: index = x*64+y
        int tileIdx = tileX * 64 + tileY;
        if (tileIdx < 0 || tileIdx >= _adtOffsets.Count || _adtOffsets[tileIdx] == 0)
            return new List<TerrainChunkData>();

        // Use the existing AdtAlpha parser to get MCIN offsets and MTEX
        var adt = new AdtAlpha(_wdtPath, _adtOffsets[tileIdx], tileIdx);
        var mtexNames = adt.GetMtexTextureNames();
        TileTextures[(tileX, tileY)] = mtexNames;

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

        // Collect MDDF/MODF placement entries from this ADT
        CollectMddfPlacements(adt.GetMddfRaw());
        CollectModfPlacements(adt.GetModfRaw());

        Console.WriteLine($"[TerrainAdapter] Tile ({tileX},{tileY}): {chunks.Count} chunks, {mtexNames.Count} textures");
        return chunks;
    }

    /// <summary>
    /// Parse MDDF raw bytes into placement entries. Entry size = 36 bytes.
    /// Layout: nameIndex(4) uniqueId(4) posX(4) posZ(4) posY(4) rotX(4) rotZ(4) rotY(4) scale(2) flags(2)
    /// </summary>
    private void CollectMddfPlacements(byte[] mddfData)
    {
        const int entrySize = 36;
        for (int off = 0; off + entrySize <= mddfData.Length; off += entrySize)
        {
            int nameIdx = BitConverter.ToInt32(mddfData, off);
            int uniqueId = BitConverter.ToInt32(mddfData, off + 4);

            if (!_seenMddfIds.Add(uniqueId)) continue; // deduplicate

            float posX = BitConverter.ToSingle(mddfData, off + 8);
            float posZ = BitConverter.ToSingle(mddfData, off + 12);
            float posY = BitConverter.ToSingle(mddfData, off + 16);
            float rotX = BitConverter.ToSingle(mddfData, off + 20);
            float rotZ = BitConverter.ToSingle(mddfData, off + 24);
            float rotY = BitConverter.ToSingle(mddfData, off + 28);
            ushort scale = BitConverter.ToUInt16(mddfData, off + 32);

            MddfPlacements.Add(new MddfPlacement
            {
                NameIndex = nameIdx,
                Position = new Vector3(posX, posY, posZ),
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
        const int entrySize = 64;
        for (int off = 0; off + entrySize <= modfData.Length; off += entrySize)
        {
            int nameIdx = BitConverter.ToInt32(modfData, off);
            int uniqueId = BitConverter.ToInt32(modfData, off + 4);

            if (!_seenModfIds.Add(uniqueId)) continue; // deduplicate

            float posX = BitConverter.ToSingle(modfData, off + 8);
            float posZ = BitConverter.ToSingle(modfData, off + 12);
            float posY = BitConverter.ToSingle(modfData, off + 16);
            float rotX = BitConverter.ToSingle(modfData, off + 20);
            float rotZ = BitConverter.ToSingle(modfData, off + 24);
            float rotY = BitConverter.ToSingle(modfData, off + 28);
            float bbMinX = BitConverter.ToSingle(modfData, off + 32);
            float bbMinZ = BitConverter.ToSingle(modfData, off + 36);
            float bbMinY = BitConverter.ToSingle(modfData, off + 40);
            float bbMaxX = BitConverter.ToSingle(modfData, off + 44);
            float bbMaxZ = BitConverter.ToSingle(modfData, off + 48);
            float bbMaxY = BitConverter.ToSingle(modfData, off + 52);
            ushort flags = BitConverter.ToUInt16(modfData, off + 56);

            ModfPlacements.Add(new ModfPlacement
            {
                NameIndex = nameIdx,
                Position = new Vector3(posX, posY, posZ),
                Rotation = new Vector3(rotX, rotY, rotZ),
                BoundsMin = new Vector3(bbMinX, bbMinY, bbMinZ),
                BoundsMax = new Vector3(bbMaxX, bbMaxY, bbMaxZ),
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

        // Compute world position for this chunk
        // WoW coordinate system: origin at center of map, Y increases northward, X increases westward
        float worldX = (32 - tileX) * WoWConstants.ChunkSize - chunkY * (WoWConstants.ChunkSize / 16f);
        float worldY = (32 - tileY) * WoWConstants.ChunkSize - chunkX * (WoWConstants.ChunkSize / 16f);

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
}
