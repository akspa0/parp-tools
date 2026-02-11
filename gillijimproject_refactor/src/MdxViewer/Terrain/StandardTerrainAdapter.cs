using System.Collections.Concurrent;
using System.Numerics;
using System.Text;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using WoWMapConverter.Core.Formats.LichKing;

namespace MdxViewer.Terrain;

/// <summary>
/// Terrain adapter for standard (LK/Cata+) WDT + split ADT files.
/// Reads from IDataSource (MPQ archives or loose files).
/// Produces <see cref="TerrainChunkData"/> compatible with the existing rendering pipeline.
/// </summary>
public class StandardTerrainAdapter : ITerrainAdapter
{
    private readonly IDataSource _dataSource;
    private readonly string _mapName;
    private readonly string _mapDir; // e.g. "World\\Maps\\Azeroth"
    private readonly List<int> _existingTiles;
    private readonly HashSet<int> _existingTileSet;
    private readonly uint _mphdFlags;
    private readonly bool _useBigAlpha;

    public ConcurrentDictionary<(int tileX, int tileY), List<string>> TileTextures { get; } = new();
    public IReadOnlyList<string> MdxModelNames => _mdxNames;
    public IReadOnlyList<string> WmoModelNames => _wmoNames;
    public List<MddfPlacement> MddfPlacements { get; } = new();
    public List<ModfPlacement> ModfPlacements { get; } = new();
    public bool IsWmoBased { get; }
    public List<Vector3> LastLoadedChunkPositions { get; } = new();
    public IReadOnlyList<int> ExistingTiles => _existingTiles;

    private readonly List<string> _mdxNames = new();
    private readonly List<string> _wmoNames = new();
    private readonly object _placementLock = new();
    private readonly HashSet<int> _seenMddfIds = new();
    private readonly HashSet<int> _seenModfIds = new();
    private readonly Dictionary<string, int> _mdxNameIndex = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, int> _wmoNameIndex = new(StringComparer.OrdinalIgnoreCase);

    public StandardTerrainAdapter(byte[] wdtBytes, string mapName, IDataSource dataSource)
    {
        _dataSource = dataSource;
        _mapName = mapName;
        _mapDir = $"World\\Maps\\{mapName}";

        // Parse MPHD
        _mphdFlags = ReadMphdFlags(wdtBytes);
        _useBigAlpha = (_mphdFlags & 0x4) != 0;

        // Parse MAIN chunk to enumerate tiles
        _existingTiles = ReadMainChunk(wdtBytes);
        _existingTileSet = new HashSet<int>(_existingTiles);

        // Check for WMO-only map (no terrain tiles but has MODF)
        IsWmoBased = _existingTiles.Count == 0;

        ViewerLog.Important(ViewerLog.Category.Terrain,
            $"Standard WDT: {_existingTiles.Count} tiles, MPHD=0x{_mphdFlags:X}, bigAlpha={_useBigAlpha}");
    }

    public bool TileExists(int tileX, int tileY)
    {
        int idx = tileX * 64 + tileY;
        return _existingTileSet.Contains(idx);
    }

    public TileLoadResult LoadTileWithPlacements(int tileX, int tileY)
    {
        var result = new TileLoadResult();
        if (!TileExists(tileX, tileY))
            return result;

        // Build ADT virtual paths
        // Standard WDT naming: tileIndex = tileX*64+tileY, file uses x=tileY, y=tileX
        // Actually for standard WDT, file naming is MapName_XX_YY.adt where XX=tileX, YY=tileY
        // per wowdev.wiki: MapName_X_Y.adt
        string basePath = $"{_mapDir}\\{_mapName}_{tileX}_{tileY}";
        string rootPath = $"{basePath}.adt";
        string texPath = $"{basePath}_tex0.adt";
        string objPath = $"{basePath}_obj0.adt";

        var adtBytes = _dataSource.ReadFile(rootPath);
        if (adtBytes == null || adtBytes.Length == 0)
        {
            ViewerLog.Trace($"[StandardADT] ADT not found or empty: {rootPath}");
            return result;
        }
        ViewerLog.Trace($"[StandardADT] Loaded {rootPath}: {adtBytes.Length} bytes, first4='{Encoding.ASCII.GetString(adtBytes, 0, Math.Min(4, adtBytes.Length))}'");

        // Optional split files (Cata+)
        var texBytes = _dataSource.FileExists(texPath) ? _dataSource.ReadFile(texPath) : null;
        var objBytes = _dataSource.FileExists(objPath) ? _dataSource.ReadFile(objPath) : null;

        try
        {
            ParseAdt(adtBytes, texBytes, objBytes, tileX, tileY, result);
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Terrain, $"Failed to parse ADT ({tileX},{tileY}): {ex.Message}");
        }

        return result;
    }

    private void ParseAdt(byte[] adtBytes, byte[]? texBytes, byte[]? objBytes,
        int tileX, int tileY, TileLoadResult result)
    {
        // Parse top-level chunks for MTEX, MMDX, MWMO
        var textures = new List<string>();
        var m2Names = new List<string>();
        var wmoNames = new List<string>();

        // Find MHDR to get MCIN offsets
        int mhdrOffset = FindChunk(adtBytes, "MHDR");
        if (mhdrOffset < 0)
        {
            ViewerLog.Info(ViewerLog.Category.Terrain, $"MHDR not found in ADT ({tileX},{tileY})");
            return;
        }

        // Parse top-level chunks
        for (int i = 0; i + 8 <= adtBytes.Length;)
        {
            string fcc = Encoding.ASCII.GetString(adtBytes, i, 4);
            int sz = BitConverter.ToInt32(adtBytes, i + 4);
            if (sz < 0) break;
            int dataStart = i + 8;
            int next = dataStart + sz + ((sz & 1) == 1 ? 1 : 0);
            if (dataStart + sz > adtBytes.Length) break;

            switch (fcc)
            {
                case "XETM": // MTEX reversed
                    textures.AddRange(ParseNullStrings(adtBytes, dataStart, sz));
                    break;
                case "XDMM": // MMDX reversed
                    m2Names.AddRange(ParseNullStrings(adtBytes, dataStart, sz));
                    break;
                case "OMWM": // MWMO reversed
                    wmoNames.AddRange(ParseNullStrings(adtBytes, dataStart, sz));
                    break;
            }

            if (next <= i) break;
            i = next;
        }

        // Store textures for this tile
        if (textures.Count > 0)
            TileTextures.TryAdd((tileX, tileY), textures);

        // Use MHDR to find MCIN
        var mhdr = new GillijimProject.WowFiles.Mhdr(adtBytes, mhdrOffset);
        int mhdrStart = mhdrOffset + 8;
        int mcinOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.McinOffset);
        if (mcinOff == 0)
        {
            ViewerLog.Info(ViewerLog.Category.Terrain, $"MCIN offset zero in ADT ({tileX},{tileY})");
            return;
        }

        int mcinAbsPos = mhdrStart + mcinOff;
        // Diagnostic: log MCIN position and first bytes
        if (mcinAbsPos + 8 <= adtBytes.Length)
        {
            string mcinSig = Encoding.ASCII.GetString(adtBytes, mcinAbsPos, 4);
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"MCIN at file pos {mcinAbsPos}: sig='{mcinSig}' (mhdrOff={mhdrOffset}, mhdrStart={mhdrStart}, mcinOff={mcinOff})");
        }

        var mcin = new GillijimProject.WowFiles.Mcin(adtBytes, mcinAbsPos);
        var mcnkOffsets = mcin.GetMcnkOffsets();

        float chunkSmall = WoWConstants.ChunkSize / 16f;
        var chunks = new List<TerrainChunkData>(256);

        // Diagnostic: log first 3 MCIN offsets
        ViewerLog.Important(ViewerLog.Category.Terrain,
            $"MCIN offsets[0..2]: {mcnkOffsets[0]}, {mcnkOffsets[1]}, {mcnkOffsets[2]} (fileLen={adtBytes.Length})");

        for (int ci = 0; ci < 256 && ci < mcnkOffsets.Count; ci++)
        {
            int off = mcnkOffsets[ci];
            if (off <= 0) continue;

            try
            {
                // Read MCNK size
                int mcnkSize = 0;
                if (off + 8 <= adtBytes.Length)
                {
                    string sig = Encoding.ASCII.GetString(adtBytes, off, 4);
                    if (sig == "KNCM") // MCNK reversed
                        mcnkSize = BitConverter.ToInt32(adtBytes, off + 4);
                    else if (ci == 0)
                        ViewerLog.Important(ViewerLog.Category.Terrain,
                            $"Chunk 0 at off={off}: sig='{sig}' (expected 'KNCM')");
                }
                if (mcnkSize <= 0)
                {
                    if (ci == 0)
                        ViewerLog.Important(ViewerLog.Category.Terrain,
                            $"Chunk 0: mcnkSize={mcnkSize}, off={off}, off+8<len={off + 8 <= adtBytes.Length}");
                    continue;
                }

                // Extract MCNK data (skip the 8-byte chunk header)
                int dataLen = Math.Min(mcnkSize, adtBytes.Length - off - 8);
                if (dataLen < 128) continue;
                var mcnkData = new byte[dataLen];
                Array.Copy(adtBytes, off + 8, mcnkData, 0, dataLen);

                var mcnk = new Mcnk(mcnkData);

                int chunkX = (int)mcnk.Header.IndexX;
                int chunkY = (int)mcnk.Header.IndexY;

                // Heights (already interleaved in LK format)
                float[]? heights = mcnk.Heightmap;
                if (ci == 0)
                    ViewerLog.Important(ViewerLog.Category.Terrain,
                        $"Chunk 0: idx=({chunkX},{chunkY}), heights={heights?.Length ?? -1}, heightOff=0x{mcnk.Header.HeightmapOffset:X}");
                if (heights == null || heights.Length < 145) continue;

                // Normals (interleaved in LK)
                var normals = ExtractNormals(mcnk.McnrData);

                // Layers
                var layers = ExtractLayers(mcnk.TextureLayers);

                // Alpha maps
                var alphaMaps = ExtractAlphaMaps(mcnk, _useBigAlpha);

                // Shadow map
                byte[]? shadowMap = ExtractShadowMap(mcnk.McshData);

                // Hole mask
                int holeMask = (int)mcnk.Header.Holes;

                // World position (LK uses interleaved format, same coordinate mapping)
                float worldX = WoWConstants.MapOrigin - tileX * WoWConstants.ChunkSize - chunkY * chunkSmall;
                float worldY = WoWConstants.MapOrigin - tileY * WoWConstants.ChunkSize - chunkX * chunkSmall;

                chunks.Add(new TerrainChunkData
                {
                    TileX = tileX,
                    TileY = tileY,
                    ChunkX = chunkX,
                    ChunkY = chunkY,
                    Heights = heights,
                    Normals = normals,
                    HoleMask = holeMask,
                    Layers = layers,
                    AlphaMaps = alphaMaps,
                    ShadowMap = shadowMap,
                    WorldPosition = new Vector3(worldX, worldY, 0f)
                });

                LastLoadedChunkPositions.Add(new Vector3(worldX, worldY, 0f));
            }
            catch (Exception ex)
            {
                ViewerLog.Debug(ViewerLog.Category.Terrain, $"Chunk {ci} parse error in ({tileX},{tileY}): {ex.Message}");
            }
        }

        result.Chunks.AddRange(chunks);

        // Parse MDDF/MODF placements from root ADT (or obj ADT if split)
        var placementSource = objBytes ?? adtBytes;
        CollectPlacements(placementSource, m2Names, wmoNames, result);
    }

    private Vector3[] ExtractNormals(byte[]? mcnrData)
    {
        var normals = new Vector3[145];

        if (mcnrData != null && mcnrData.Length >= 435)
        {
            // LK MCNR is already interleaved (9-8-9-8 pattern)
            for (int i = 0; i < 145; i++)
            {
                int off = i * 3;
                float nx = (sbyte)mcnrData[off] / 127f;
                float nz = (sbyte)mcnrData[off + 1] / 127f;
                float ny = (sbyte)mcnrData[off + 2] / 127f;
                var n = new Vector3(nx, ny, nz);
                float len = n.Length();
                normals[i] = len > 0.001f ? n / len : Vector3.UnitZ;
            }
            return normals;
        }

        for (int i = 0; i < 145; i++)
            normals[i] = Vector3.UnitZ;
        return normals;
    }

    private static TerrainLayer[] ExtractLayers(List<MclyEntry>? mclyEntries)
    {
        if (mclyEntries == null || mclyEntries.Count == 0)
            return Array.Empty<TerrainLayer>();

        var layers = new TerrainLayer[mclyEntries.Count];
        for (int i = 0; i < mclyEntries.Count; i++)
        {
            var e = mclyEntries[i];
            layers[i] = new TerrainLayer
            {
                TextureIndex = (int)e.TextureId,
                Flags = (uint)e.Flags,
                AlphaOffset = e.AlphaMapOffset,
                EffectId = e.EffectId
            };
        }
        return layers;
    }

    private static Dictionary<int, byte[]> ExtractAlphaMaps(Mcnk mcnk, bool useBigAlpha)
    {
        var maps = new Dictionary<int, byte[]>();
        if (mcnk.AlphaMaps == null || mcnk.TextureLayers == null)
            return maps;

        for (int i = 1; i < mcnk.TextureLayers.Count && i < 4; i++)
        {
            try
            {
                var alpha = mcnk.AlphaMaps.GetAlphaMapForLayer(mcnk.TextureLayers[i], useBigAlpha);
                if (alpha != null && alpha.Length > 0)
                    maps[i] = alpha;
            }
            catch { }
        }

        return maps;
    }

    private static byte[]? ExtractShadowMap(byte[]? mcshData)
    {
        if (mcshData == null || mcshData.Length < 8) return null;

        // MCSH: 64 rows × 8 bytes = 512 bytes of shadow bits
        var shadow = new byte[64 * 64];
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                int byteIndex = y * 8 + (x / 8);
                int bitIndex = x % 8;
                if (byteIndex < mcshData.Length)
                {
                    bool isShadowed = (mcshData[byteIndex] & (1 << bitIndex)) != 0;
                    shadow[y * 64 + x] = isShadowed ? (byte)255 : (byte)0;
                }
            }
        }
        return shadow;
    }

    private void CollectPlacements(byte[] data, List<string> m2Names, List<string> wmoNames, TileLoadResult result)
    {
        // Find MDDF and MODF chunks
        for (int i = 0; i + 8 <= data.Length;)
        {
            string fcc = Encoding.ASCII.GetString(data, i, 4);
            int sz = BitConverter.ToInt32(data, i + 4);
            if (sz < 0) break;
            int dataStart = i + 8;
            int next = dataStart + sz + ((sz & 1) == 1 ? 1 : 0);
            if (dataStart + sz > data.Length) break;

            if (fcc == "FDDM" && sz >= 36) // MDDF reversed
            {
                ParseMddfChunk(data, dataStart, sz, m2Names, result);
            }
            else if (fcc == "FDOM" && sz >= 64) // MODF reversed
            {
                ParseModfChunk(data, dataStart, sz, wmoNames, result);
            }

            if (next <= i) break;
            i = next;
        }
    }

    private void ParseMddfChunk(byte[] data, int offset, int size, List<string> m2Names, TileLoadResult result)
    {
        int entrySize = 36;
        int count = size / entrySize;

        for (int i = 0; i < count; i++)
        {
            int pos = offset + i * entrySize;
            if (pos + entrySize > data.Length) break;

            uint nameId = BitConverter.ToUInt32(data, pos);
            uint uniqueId = BitConverter.ToUInt32(data, pos + 4);
            float px = BitConverter.ToSingle(data, pos + 8);
            float py = BitConverter.ToSingle(data, pos + 12);
            float pz = BitConverter.ToSingle(data, pos + 16);
            float rx = BitConverter.ToSingle(data, pos + 20);
            float ry = BitConverter.ToSingle(data, pos + 24);
            float rz = BitConverter.ToSingle(data, pos + 28);
            ushort scale = BitConverter.ToUInt16(data, pos + 32);

            lock (_placementLock)
            {
                if (!_seenMddfIds.Add((int)uniqueId)) continue;

                // Resolve name and get/create index
                string name = nameId < m2Names.Count ? m2Names[(int)nameId] : $"unknown_m2_{nameId}";
                int nameIdx = GetOrAddMdxName(name);

                var placement = new MddfPlacement
                {
                    NameIndex = nameIdx,
                    UniqueId = (int)uniqueId,
                    Position = new Vector3(
                        WoWConstants.MapOrigin - py,
                        WoWConstants.MapOrigin - px,
                        pz),
                    Rotation = new Vector3(rx, ry, rz),
                    Scale = scale / 1024f
                };

                MddfPlacements.Add(placement);
                result.MddfPlacements.Add(placement);
            }
        }
    }

    private void ParseModfChunk(byte[] data, int offset, int size, List<string> wmoNames, TileLoadResult result)
    {
        int entrySize = 64;
        int count = size / entrySize;

        for (int i = 0; i < count; i++)
        {
            int pos = offset + i * entrySize;
            if (pos + entrySize > data.Length) break;

            uint nameId = BitConverter.ToUInt32(data, pos);
            uint uniqueId = BitConverter.ToUInt32(data, pos + 4);
            float px = BitConverter.ToSingle(data, pos + 8);
            float py = BitConverter.ToSingle(data, pos + 12);
            float pz = BitConverter.ToSingle(data, pos + 16);
            float rx = BitConverter.ToSingle(data, pos + 20);
            float ry = BitConverter.ToSingle(data, pos + 24);
            float rz = BitConverter.ToSingle(data, pos + 28);
            float bminX = BitConverter.ToSingle(data, pos + 32);
            float bminY = BitConverter.ToSingle(data, pos + 36);
            float bminZ = BitConverter.ToSingle(data, pos + 40);
            float bmaxX = BitConverter.ToSingle(data, pos + 44);
            float bmaxY = BitConverter.ToSingle(data, pos + 48);
            float bmaxZ = BitConverter.ToSingle(data, pos + 52);
            ushort flags = BitConverter.ToUInt16(data, pos + 56);

            lock (_placementLock)
            {
                if (!_seenModfIds.Add((int)uniqueId)) continue;

                string name = nameId < wmoNames.Count ? wmoNames[(int)nameId] : $"unknown_wmo_{nameId}";
                int nameIdx = GetOrAddWmoName(name);

                var placement = new ModfPlacement
                {
                    NameIndex = nameIdx,
                    UniqueId = (int)uniqueId,
                    Position = new Vector3(
                        WoWConstants.MapOrigin - py,
                        WoWConstants.MapOrigin - px,
                        pz),
                    Rotation = new Vector3(rx, ry, rz),
                    BoundsMin = new Vector3(bminX, bminY, bminZ),
                    BoundsMax = new Vector3(bmaxX, bmaxY, bmaxZ),
                    Flags = flags
                };

                ModfPlacements.Add(placement);
                result.ModfPlacements.Add(placement);
            }
        }
    }

    private int GetOrAddMdxName(string name)
    {
        if (_mdxNameIndex.TryGetValue(name, out int idx)) return idx;
        idx = _mdxNames.Count;
        _mdxNames.Add(name);
        _mdxNameIndex[name] = idx;
        return idx;
    }

    private int GetOrAddWmoName(string name)
    {
        if (_wmoNameIndex.TryGetValue(name, out int idx)) return idx;
        idx = _wmoNames.Count;
        _wmoNames.Add(name);
        _wmoNameIndex[name] = idx;
        return idx;
    }

    // ── WDT Parsing ──

    private static uint ReadMphdFlags(byte[] wdtBytes)
    {
        int off = FindChunk(wdtBytes, "MPHD");
        if (off >= 0 && off + 12 < wdtBytes.Length)
            return BitConverter.ToUInt32(wdtBytes, off + 8);
        return 0;
    }

    private static List<int> ReadMainChunk(byte[] wdtBytes)
    {
        var tiles = new List<int>();
        int mainOff = FindChunk(wdtBytes, "MAIN");
        if (mainOff < 0) return tiles;

        int mainSize = BitConverter.ToInt32(wdtBytes, mainOff + 4);
        int mainData = mainOff + 8;

        if (mainSize < 64 * 64 * 8) return tiles;

        for (int i = 0; i < 64 * 64; i++)
        {
            int entryOff = mainData + (i * 8);
            if (entryOff + 8 > wdtBytes.Length) break;

            uint flags = BitConverter.ToUInt32(wdtBytes, entryOff);
            if (flags != 0)
                tiles.Add(i);
        }
        return tiles;
    }

    /// <summary>
    /// Find a chunk in LK format (reversed FourCC on disk).
    /// </summary>
    private static int FindChunk(byte[] bytes, string fourCC)
    {
        string reversed = new string(fourCC.Reverse().ToArray());

        for (int i = 0; i + 8 <= bytes.Length;)
        {
            string fcc = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            if (size < 0) break;

            if (fcc == reversed)
                return i;

            int next = i + 8 + size + ((size & 1) == 1 ? 1 : 0);
            if (next <= i) break;
            i = next;
        }
        return -1;
    }

    private static List<string> ParseNullStrings(byte[] data, int offset, int size)
    {
        var result = new List<string>();
        int start = offset;
        int end = offset + size;
        for (int i = offset; i < end; i++)
        {
            if (data[i] == 0)
            {
                if (i > start)
                    result.Add(Encoding.ASCII.GetString(data, start, i - start));
                start = i + 1;
            }
        }
        return result;
    }
}
