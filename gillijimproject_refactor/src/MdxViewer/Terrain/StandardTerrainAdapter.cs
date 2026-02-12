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

        // Diagnostic: dump first 5 tile indices and their decoded coordinates + filenames
        for (int di = 0; di < Math.Min(5, _existingTiles.Count); di++)
        {
            int idx = _existingTiles[di];
            int tx = idx / 64, ty = idx % 64; // tx=row(y), ty=col(x)
            string fn = $"{_mapDir}\\{_mapName}_{ty}_{tx}.adt"; // MapName_x_y = MapName_{col}_{row}
            bool exists = _dataSource.FileExists(fn);
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"  Tile[{di}]: rawIdx={idx}, tx={tx}(row), ty={ty}(col), file={fn}, exists={exists}");
        }
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
        // Raw row-major: tileX=row(y), tileY=col(x).
        // Ghidra-verified filename: MapName_x_y.adt = MapName_{tileY}_{tileX}.
        string basePath = $"{_mapDir}\\{_mapName}_{tileY}_{tileX}";
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
        // Parse top-level MTEX chunk for texture names
        var textures = new List<string>();

        // Find MHDR — all other chunks located via MHDR offsets (Ghidra-verified)
        int mhdrOffset = FindChunk(adtBytes, "MHDR");
        if (mhdrOffset < 0)
        {
            ViewerLog.Info(ViewerLog.Category.Terrain, $"MHDR not found in ADT ({tileX},{tileY})");
            return;
        }

        // Scan for MTEX (texture names still needed for terrain layers)
        for (int i = 0; i + 8 <= adtBytes.Length;)
        {
            string fcc = Encoding.ASCII.GetString(adtBytes, i, 4);
            int sz = BitConverter.ToInt32(adtBytes, i + 4);
            if (sz < 0) break;
            int dataStart = i + 8;
            int next = dataStart + sz + ((sz & 1) == 1 ? 1 : 0);
            if (dataStart + sz > adtBytes.Length) break;

            if (fcc == "XETM") // MTEX reversed
            {
                textures.AddRange(ParseNullStrings(adtBytes, dataStart, sz));
                break; // Only need MTEX
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

                // Diagnostic: log first chunk of each tile
                if (ci == 0)
                {
                    var pos = mcnk.Header.Position;
                    string posStr = pos != null && pos.Length >= 3
                        ? $"Z={pos[0]:F1}, X={pos[1]:F1}, Y={pos[2]:F1}"
                        : "null";
                    ViewerLog.Important(ViewerLog.Category.Terrain,
                        $"  Tile({tileX},{tileY}) chunk0: idx=({chunkX},{chunkY}), pos=[{posStr}], baseZ={((pos != null && pos.Length >= 1) ? pos[0] : 0f):F1}");
                }

                // Heights (already interleaved in LK format)
                // MCVT values are deltas from the chunk's base Z (Position[0]).
                float[]? heights = mcnk.Heightmap;
                if (heights == null || heights.Length < 145) continue;

                // Add base height from MCNK Position field (Z, X, Y order)
                float baseZ = (mcnk.Header.Position != null && mcnk.Header.Position.Length >= 1)
                    ? mcnk.Header.Position[0] : 0f;
                if (!float.IsNaN(baseZ) && MathF.Abs(baseZ) < 50000f && baseZ != 0f)
                {
                    for (int hi = 0; hi < heights.Length; hi++)
                        heights[hi] += baseZ;
                }

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

                // World position: tileX=row (north-south→rendererX), tileY=col (east-west→rendererY)
                // Same convention as Alpha adapter (tx=row, ty=col).
                float worldX = WoWConstants.MapOrigin - tileX * WoWConstants.ChunkSize - chunkY * chunkSmall;
                float worldY = WoWConstants.MapOrigin - tileY * WoWConstants.ChunkSize - chunkX * chunkSmall;

                // MCLQ inline liquid (per-chunk, legacy format used alongside MH2O)
                LiquidChunkData? liquid = null;
                uint mcnkFlagsRaw = (uint)mcnk.Header.Flags;
                bool hasLiquidFlags = (mcnkFlagsRaw & 0x3C) != 0;
                if (mcnk.MclqData != null && mcnk.MclqData.Length >= 8)
                {
                    liquid = ExtractMclq(mcnk.MclqData, mcnkFlagsRaw,
                        tileX, tileY, chunkX, chunkY,
                        new Vector3(worldX, worldY, 0f), baseZ);
                }
                // Diagnostic: log first few chunks with liquid flags
                if (hasLiquidFlags && ci < 4)
                {
                    var sb = new System.Text.StringBuilder();
                    sb.Append($"[MCLQ-DIAG] tile({tileX},{tileY}) chunk({chunkX},{chunkY}) flags=0x{mcnkFlagsRaw:X} ofsMclq=0x{mcnk.Header.OfsMclq:X}");
                    sb.Append($" baseZ={baseZ:F2} mclqData={mcnk.MclqData?.Length ?? -1}");
                    if (mcnk.MclqData != null && mcnk.MclqData.Length >= 16)
                    {
                        float v0 = BitConverter.ToSingle(mcnk.MclqData, 0);
                        float v1 = BitConverter.ToSingle(mcnk.MclqData, 4);
                        float v2 = BitConverter.ToSingle(mcnk.MclqData, 8);
                        float v3 = BitConverter.ToSingle(mcnk.MclqData, 12);
                        sb.Append($" raw[0..3]: {v0:F2} {v1:F2} {v2:F2} {v3:F2}");
                    }
                    sb.Append($" liquid={liquid != null}");
                    if (liquid != null)
                        sb.Append($" minH={liquid.MinHeight:F2} maxH={liquid.MaxHeight:F2} h[0]={liquid.Heights[0]:F2} h[40]={liquid.Heights[40]:F2}");
                    ViewerLog.Important(ViewerLog.Category.Terrain, sb.ToString());
                }

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
                    Liquid = liquid,
                    WorldPosition = new Vector3(worldX, worldY, 0f)
                });

                LastLoadedChunkPositions.Add(new Vector3(worldX, worldY, 0f));
            }
            catch (Exception ex)
            {
                ViewerLog.Debug(ViewerLog.Category.Terrain, $"Chunk {ci} parse error in ({tileX},{tileY}): {ex.Message}");
            }
        }

        int mclqCount = chunks.Count(c => c.Liquid != null);
        if (chunks.Count > 0)
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"ADT ({tileX},{tileY}): parsed {chunks.Count} chunks with heightmaps, {mclqCount} with MCLQ liquid");
        else
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"ADT ({tileX},{tileY}): WARNING - 0 chunks parsed (no heightmaps found)");

        result.Chunks.AddRange(chunks);

        // Ghidra-verified (FUN_007d6ef0): MH2O located via MHDR offset +0x28.
        // Only parse MH2O if no MCLQ liquid was already found (0.6.0 has MCLQ, not MH2O).
        if (mclqCount == 0)
            ParseMh2o(adtBytes, mhdrStart, mhdr, tileX, tileY, chunkSmall, result);

        // Ghidra-verified (FUN_007d6ef0): MDDF/MODF are located via MHDR offsets,
        // NOT by linear scan. Name resolution: MDDF.nameId → MMID[nameId] → byte offset into MMDX.
        CollectPlacementsViaMhdr(adtBytes, mhdrStart, mhdr, tileX, tileY, result);
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
        if (mcnk.TextureLayers == null || mcnk.TextureLayers.Count <= 1)
            return maps;

        // Detect format: if any layer has UseAlpha (0x100) or CompressedAlpha (0x200),
        // use 3.3.5 flag-based decode via Mcal. Otherwise use Alpha-style sequential 4-bit.
        bool hasLkFlags = false;
        for (int i = 1; i < mcnk.TextureLayers.Count && i < 4; i++)
        {
            if ((mcnk.TextureLayers[i].Flags & (MclyFlags.UseAlpha | MclyFlags.CompressedAlpha)) != 0)
            {
                hasLkFlags = true;
                break;
            }
        }

        if (hasLkFlags && mcnk.AlphaMaps != null)
        {
            // 3.3.5 path: per-layer offsets + flags (compressed, big alpha, etc.)
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
        }
        else if (mcnk.McalRawData != null && mcnk.McalRawData.Length > 0)
        {
            // Alpha-style (0.5.3/0.6.0): sequential 4-bit nibbles, 2048 bytes per layer
            int offset = 0;
            int nLayers = mcnk.TextureLayers.Count;
            for (int layer = 1; layer < nLayers && layer < 4; layer++)
            {
                int alphaSize = 2048; // 4-bit: 64×64 / 2
                if (offset + alphaSize > mcnk.McalRawData.Length)
                {
                    alphaSize = mcnk.McalRawData.Length - offset;
                    if (alphaSize <= 0) break;
                }

                var alpha = new byte[4096]; // 64×64 output
                for (int j = 0; j < Math.Min(2048, alphaSize); j++)
                {
                    byte packed = mcnk.McalRawData[offset + j];
                    alpha[j * 2] = (byte)((packed & 0x0F) * 17);
                    alpha[j * 2 + 1] = (byte)((packed >> 4) * 17);
                }

                maps[layer] = alpha;
                offset += alphaSize;
            }
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

    /// <summary>
    /// Extract MCLQ inline liquid data from raw bytes.
    /// Supports both 3.3.5 single-instance format and 0.6.0 packed multi-instance format.
    /// 0.6.0: payload is packed instances (0x2D4 bytes each), one per liquid flag bit
    /// (0x04=River, 0x08=Ocean, 0x10=Magma, 0x20=Slime). Instance layout:
    ///   +0x000: minHeight (float), +0x004: maxHeight (float),
    ///   +0x008: 81 vertices × 8 bytes (648), +0x290: 64 tile flags, +0x2D0: trailing uint32.
    /// 3.3.5: single instance (8 + 81×8 + 64 = 720 bytes).
    /// </summary>
    private static LiquidChunkData? ExtractMclq(byte[] mclqData, uint mcnkFlags,
        int tileX, int tileY, int chunkX, int chunkY, Vector3 worldPos, float baseHeight)
    {
        if (mclqData == null || mclqData.Length < 8)
            return null;

        // Determine which liquid types are present from MCNK flags
        uint[] liquidBits = { 0x04, 0x08, 0x10, 0x20 };
        LiquidType[] liquidTypes = { LiquidType.Water, LiquidType.Ocean, LiquidType.Magma, LiquidType.Slime };

        // Check if this is a packed multi-instance format (0.6.0)
        // by counting how many liquid flag bits are set
        int instanceCount = 0;
        for (int b = 0; b < 4; b++)
            if ((mcnkFlags & liquidBits[b]) != 0) instanceCount++;

        if (instanceCount == 0)
            return null;

        // Try to parse packed instances (0x2D4 each)
        // If data is large enough for packed format, use it; otherwise fall back to single-instance
        int packedSize = instanceCount * 0x2D4;
        bool usePacked = mclqData.Length >= packedSize && instanceCount > 0;

        // For single-instance (3.3.5 or single liquid type), also accept 720-byte format
        if (!usePacked && mclqData.Length >= 720)
            usePacked = false; // use legacy single-instance path below

        int offset = 0;
        LiquidChunkData? result = null;

        for (int b = 0; b < 4; b++)
        {
            if ((mcnkFlags & liquidBits[b]) == 0) continue;

            if (offset + 8 > mclqData.Length) break;

            float minHeight = BitConverter.ToSingle(mclqData, offset + 0);
            float maxHeight = BitConverter.ToSingle(mclqData, offset + 4);

            if (float.IsNaN(minHeight) || float.IsNaN(maxHeight))
            {
                if (usePacked) { offset += 0x2D4; continue; }
                else break;
            }

            LiquidType liquidType = liquidTypes[b];

            // Build 9×9 height grid from per-vertex data or flat plane
            var heights = new float[81];
            bool hasPerVertex = false;

            if (offset + 8 + 81 * 8 <= mclqData.Length)
            {
                for (int i = 0; i < 81; i++)
                {
                    int voff = offset + 8 + i * 8;
                    float h = BitConverter.ToSingle(mclqData, voff);
                    if (float.IsNaN(h) || MathF.Abs(h) > 50000f)
                        h = 0f;
                    heights[i] = h;
                }
                // Check if per-vertex data is meaningful (not all near-zero)
                for (int i = 0; i < 81; i++)
                {
                    if (MathF.Abs(heights[i]) > 0.01f) { hasPerVertex = true; break; }
                }
            }

            if (!hasPerVertex)
            {
                // Flat plane at (min+max)/2 — these are absolute world Z
                float liquidLevel = (minHeight + maxHeight) * 0.5f;
                Array.Fill(heights, liquidLevel);
            }

            // Read 64 tile flags at offset + 0x290 (packed) or offset + 8 + 81*8 (legacy)
            int tileFlagsOff = usePacked ? (offset + 0x290) : (offset + 8 + 81 * 8);
            byte[] tileFlags = null;
            if (tileFlagsOff + 64 <= mclqData.Length)
            {
                tileFlags = new byte[64];
                Array.Copy(mclqData, tileFlagsOff, tileFlags, 0, 64);
            }

            // Check if all tiles hidden
            bool anyVisible = true;
            if (tileFlags != null)
            {
                anyVisible = false;
                for (int i = 0; i < 64; i++)
                {
                    if ((tileFlags[i] & 0x0F) != 0x0F) { anyVisible = true; break; }
                }
            }

            if (anyVisible && result == null)
            {
                result = new LiquidChunkData
                {
                    MinHeight = minHeight,
                    MaxHeight = maxHeight,
                    Heights = heights,
                    Type = liquidType,
                    WorldPosition = worldPos,
                    TileX = tileX,
                    TileY = tileY,
                    ChunkX = chunkX,
                    ChunkY = chunkY
                };
            }

            if (usePacked) offset += 0x2D4;
            else break; // single instance, done
        }

        return result;
    }

    private static byte[] StripMclqChunkHeaderIfPresent(byte[] mclqData)
    {
        // Some builds/paths provide MCLQ as a full chunk: [FourCC][uint32 size][payload...]
        // 0.6.0 client code (Ghidra) treats MCLQ like a normal chunk and uses payload at +8.
        if (mclqData.Length < 8)
            return mclqData;

        bool isMclq = mclqData[0] == (byte)'M' && mclqData[1] == (byte)'C' && mclqData[2] == (byte)'L' && mclqData[3] == (byte)'Q';
        bool isReversed = mclqData[0] == (byte)'Q' && mclqData[1] == (byte)'L' && mclqData[2] == (byte)'C' && mclqData[3] == (byte)'M';
        if (!isMclq && !isReversed)
            return mclqData;

        uint size = BitConverter.ToUInt32(mclqData, 4);
        if (size == 0)
            return Array.Empty<byte>();

        // Basic sanity: size must fit in the provided buffer (ignore optional padding byte).
        int available = mclqData.Length - 8;
        if (size > (uint)available)
            return mclqData;

        var payload = new byte[size];
        Buffer.BlockCopy(mclqData, 8, payload, 0, (int)size);
        return payload;
    }

    /// <summary>
    /// Parse MH2O liquid data via MHDR offset +0x28 (Ghidra-verified FUN_007d6ef0).
    /// MH2O header: 256 entries × 12 bytes (one per MCNK chunk).
    /// Each entry: { uint32 ofsInformation, uint32 layerCount, uint32 ofsRender }
    /// SMLiquidInstance: { uint16 liquidType, uint16 liquidObject, float min, float max,
    ///   uint8 xOfs, uint8 yOfs, uint8 width, uint8 height, uint32 ofsMask, uint32 ofsHeightmap }
    /// </summary>
    private void ParseMh2o(byte[] adt, int mhdrStart, GillijimProject.WowFiles.Mhdr mhdr,
        int tileX, int tileY, float chunkSmall, TileLoadResult result)
    {
        int mh2oOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.Mh2oOffset);
        if (mh2oOff == 0)
        {
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"  Tile({tileX},{tileY}) MH2O: MHDR offset is 0 (no MH2O)");
            return;
        }

        int mh2oAbs = mhdrStart + mh2oOff;
        if (mh2oAbs + 8 > adt.Length)
        {
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"  Tile({tileX},{tileY}) MH2O: abs pos {mh2oAbs} beyond file ({adt.Length})");
            return;
        }

        // Check FourCC at the position
        string mh2oSig = Encoding.ASCII.GetString(adt, mh2oAbs, 4);
        int mh2oSize = BitConverter.ToInt32(adt, mh2oAbs + 4);
        ViewerLog.Important(ViewerLog.Category.Terrain,
            $"  Tile({tileX},{tileY}) MH2O: mhdrOff={mh2oOff}, absPos={mh2oAbs}, sig='{mh2oSig}', size={mh2oSize}");

        // MH2O chunk: skip FourCC(4) + size(4) to get data start
        int mh2oDataStart = mh2oAbs + 8;
        if (mh2oSize <= 0 || mh2oDataStart + mh2oSize > adt.Length) return;

        int liquidCount = 0;
        int skippedNoLayer = 0;

        // 256 header entries, 12 bytes each (3 × uint32)
        for (int ci = 0; ci < 256; ci++)
        {
            int headerPos = mh2oDataStart + ci * 12;
            if (headerPos + 12 > adt.Length) break;

            uint ofsInformation = BitConverter.ToUInt32(adt, headerPos);
            uint layerCount = BitConverter.ToUInt32(adt, headerPos + 4);
            // uint ofsRender = BitConverter.ToUInt32(adt, headerPos + 8); // not needed for basic rendering

            if (layerCount == 0 || ofsInformation == 0)
            {
                if (ci < 3 && layerCount == 0 && ofsInformation != 0)
                    skippedNoLayer++;
                continue;
            }

            // SMLiquidInstance is at mh2oDataStart + ofsInformation
            int instPos = mh2oDataStart + (int)ofsInformation;
            if (instPos + 24 > adt.Length) continue;

            ushort liquidTypeId = BitConverter.ToUInt16(adt, instPos);
            // ushort liquidObject = BitConverter.ToUInt16(adt, instPos + 2);
            float minHeight = BitConverter.ToSingle(adt, instPos + 4);
            float maxHeight = BitConverter.ToSingle(adt, instPos + 8);
            byte xOffset = adt[instPos + 12];
            byte yOffset = adt[instPos + 13];
            byte width = adt[instPos + 14];
            byte height = adt[instPos + 15];
            uint ofsMask = BitConverter.ToUInt32(adt, instPos + 16);
            uint ofsHeightmap = BitConverter.ToUInt32(adt, instPos + 20);

            if (width == 0 || height == 0) continue;
            if (width > 8 || height > 8) continue; // sanity

            // Chunk indices: ci = y*16+x (row-major)
            int chunkX = ci % 16; // column within tile
            int chunkY = ci / 16; // row within tile

            // Build 9×9 height grid for the liquid surface
            // Default to flat at maxHeight
            var heights = new float[81];
            for (int i = 0; i < 81; i++)
                heights[i] = maxHeight;

            // Read per-vertex heights if available
            if (ofsHeightmap != 0)
            {
                int heightPos = mh2oDataStart + (int)ofsHeightmap;
                int vertCount = (width + 1) * (height + 1);
                if (heightPos + vertCount * 4 <= adt.Length)
                {
                    // Fill sub-rect heights into the 9×9 grid
                    for (int vy = 0; vy <= height; vy++)
                    {
                        for (int vx = 0; vx <= width; vx++)
                        {
                            int srcIdx = vy * (width + 1) + vx;
                            int dstIdx = (yOffset + vy) * 9 + (xOffset + vx);
                            if (dstIdx < 81)
                                heights[dstIdx] = BitConverter.ToSingle(adt, heightPos + srcIdx * 4);
                        }
                    }
                }
            }

            // Classify liquid type
            LiquidType liqType = (liquidTypeId & 0x3) switch
            {
                0 => LiquidType.Water,
                1 => LiquidType.Ocean,
                2 => LiquidType.Magma,
                3 => LiquidType.Slime,
                _ => LiquidType.Water
            };

            // World position: same formula as terrain chunks
            // tileX=row, tileY=col (raw row-major convention)
            float worldX = WoWConstants.MapOrigin - tileX * WoWConstants.ChunkSize - chunkY * chunkSmall;
            float worldY = WoWConstants.MapOrigin - tileY * WoWConstants.ChunkSize - chunkX * chunkSmall;

            var liquid = new LiquidChunkData
            {
                MinHeight = minHeight,
                MaxHeight = maxHeight,
                Heights = heights,
                Type = liqType,
                WorldPosition = new Vector3(worldX, worldY, 0f),
                TileX = tileX,
                TileY = tileY,
                ChunkX = chunkX,
                ChunkY = chunkY
            };

            // Attach to matching terrain chunk if possible (never overwrite existing MCLQ liquid)
            var matchingChunk = result.Chunks.FirstOrDefault(c =>
                c.TileX == tileX && c.TileY == tileY && c.ChunkX == chunkX && c.ChunkY == chunkY);
            if (matchingChunk != null && matchingChunk.Liquid == null)
                matchingChunk.Liquid = liquid;

            liquidCount++;
        }

        if (liquidCount > 0)
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"  Tile({tileX},{tileY}) MH2O: {liquidCount} liquid chunks");
    }

    /// <summary>
    /// Ghidra-verified (FUN_007d6ef0): locate MDDF/MODF via MHDR offsets.
    /// Resolve names via MMID/MWID → MMDX/MWMO byte offsets.
    /// </summary>
    private void CollectPlacementsViaMhdr(byte[] adt, int mhdrStart,
        GillijimProject.WowFiles.Mhdr mhdr, int tileX, int tileY, TileLoadResult result)
    {
        // Read MHDR offsets (all relative to mhdrStart)
        int mmdxOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.MmdxOffset);
        int mmidOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.MmidOffset);
        int mwmoOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.MwmoOffset);
        int mwidOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.MwidOffset);
        int mddfOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.MddfOffset);
        int modfOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.ModfOffset);

        // Resolve absolute positions (MHDR offset + 8 to skip chunk header = data start)
        // Ghidra: pbVar1 + *(pbVar1 + offset) + 8
        int mmdxAbs = mmdxOff != 0 ? mhdrStart + mmdxOff : -1;
        int mmidAbs = mmidOff != 0 ? mhdrStart + mmidOff : -1;
        int mwmoAbs = mwmoOff != 0 ? mhdrStart + mwmoOff : -1;
        int mwidAbs = mwidOff != 0 ? mhdrStart + mwidOff : -1;
        int mddfAbs = mddfOff != 0 ? mhdrStart + mddfOff : -1;
        int modfAbs = modfOff != 0 ? mhdrStart + modfOff : -1;

        // Parse MMDX string block (null-terminated strings)
        byte[]? mmdxData = null;
        if (mmdxAbs >= 0 && mmdxAbs + 8 <= adt.Length)
        {
            int mmdxSize = BitConverter.ToInt32(adt, mmdxAbs + 4);
            int mmdxDataStart = mmdxAbs + 8;
            if (mmdxSize > 0 && mmdxDataStart + mmdxSize <= adt.Length)
                mmdxData = new ReadOnlySpan<byte>(adt, mmdxDataStart, mmdxSize).ToArray();
        }

        // Parse MMID (array of uint32 offsets into MMDX)
        List<uint>? mmidEntries = null;
        if (mmidAbs >= 0 && mmidAbs + 8 <= adt.Length)
        {
            int mmidSize = BitConverter.ToInt32(adt, mmidAbs + 4);
            int mmidDataStart = mmidAbs + 8;
            if (mmidSize > 0 && mmidDataStart + mmidSize <= adt.Length)
            {
                int count = mmidSize / 4;
                mmidEntries = new List<uint>(count);
                for (int i = 0; i < count; i++)
                    mmidEntries.Add(BitConverter.ToUInt32(adt, mmidDataStart + i * 4));
            }
        }

        // Parse MWMO string block
        byte[]? mwmoData = null;
        if (mwmoAbs >= 0 && mwmoAbs + 8 <= adt.Length)
        {
            int mwmoSize = BitConverter.ToInt32(adt, mwmoAbs + 4);
            int mwmoDataStart = mwmoAbs + 8;
            if (mwmoSize > 0 && mwmoDataStart + mwmoSize <= adt.Length)
                mwmoData = new ReadOnlySpan<byte>(adt, mwmoDataStart, mwmoSize).ToArray();
        }

        // Parse MWID (array of uint32 offsets into MWMO)
        List<uint>? mwidEntries = null;
        if (mwidAbs >= 0 && mwidAbs + 8 <= adt.Length)
        {
            int mwidSize = BitConverter.ToInt32(adt, mwidAbs + 4);
            int mwidDataStart = mwidAbs + 8;
            if (mwidSize > 0 && mwidDataStart + mwidSize <= adt.Length)
            {
                int count = mwidSize / 4;
                mwidEntries = new List<uint>(count);
                for (int i = 0; i < count; i++)
                    mwidEntries.Add(BitConverter.ToUInt32(adt, mwidDataStart + i * 4));
            }
        }

        // Parse MDDF
        if (mddfAbs >= 0 && mddfAbs + 8 <= adt.Length)
        {
            int mddfSize = BitConverter.ToInt32(adt, mddfAbs + 4);
            int mddfDataStart = mddfAbs + 8;
            if (mddfSize >= 36 && mddfDataStart + mddfSize <= adt.Length)
                ParseMddfViaMmid(adt, mddfDataStart, mddfSize, mmdxData, mmidEntries, result);
        }

        // Parse MODF
        if (modfAbs >= 0 && modfAbs + 8 <= adt.Length)
        {
            int modfSize = BitConverter.ToInt32(adt, modfAbs + 4);
            int modfDataStart = modfAbs + 8;
            if (modfSize >= 64 && modfDataStart + modfSize <= adt.Length)
                ParseModfViaMwid(adt, modfDataStart, modfSize, mwmoData, mwidEntries, result);
        }

        ViewerLog.Important(ViewerLog.Category.Terrain,
            $"  Tile({tileX},{tileY}) MHDR placements: mddf={mddfOff != 0}, modf={modfOff != 0}, " +
            $"mmid={mmidEntries?.Count ?? 0}, mwid={mwidEntries?.Count ?? 0}");
    }

    /// <summary>
    /// Resolve a name from xID[index] → byte offset into string block.
    /// </summary>
    private static string ResolveNameViaXid(uint nameId, List<uint>? xidEntries, byte[]? stringBlock)
    {
        if (xidEntries == null || stringBlock == null || nameId >= xidEntries.Count)
            return $"unknown_{nameId}";

        uint byteOffset = xidEntries[(int)nameId];
        if (byteOffset >= stringBlock.Length)
            return $"unknown_{nameId}";

        // Read null-terminated string
        int end = (int)byteOffset;
        while (end < stringBlock.Length && stringBlock[end] != 0)
            end++;
        return Encoding.ASCII.GetString(stringBlock, (int)byteOffset, end - (int)byteOffset);
    }

    private void ParseMddfViaMmid(byte[] data, int offset, int size,
        byte[]? mmdxData, List<uint>? mmidEntries, TileLoadResult result)
    {
        int entrySize = 36; // 0x24
        int count = size / entrySize;

        for (int i = 0; i < count; i++)
        {
            int pos = offset + i * entrySize;
            if (pos + entrySize > data.Length) break;

            uint nameId = BitConverter.ToUInt32(data, pos);
            uint uniqueId = BitConverter.ToUInt32(data, pos + 4);
            // MDDF position is (X, Z, Y): X=North, Z=Up(height), Y=West
            // Confirmed: LkToAlphaConverter passes positions unchanged → same layout as Alpha
            float rawX = BitConverter.ToSingle(data, pos + 8);   // North
            float rawZ = BitConverter.ToSingle(data, pos + 12);  // Up (height)
            float rawY = BitConverter.ToSingle(data, pos + 16);  // West
            // Rotation stored as (X, Z, Y) — same layout as position
            float rotX = BitConverter.ToSingle(data, pos + 20);
            float rotZ = BitConverter.ToSingle(data, pos + 24);
            float rotY = BitConverter.ToSingle(data, pos + 28);
            ushort scale = BitConverter.ToUInt16(data, pos + 32);

            lock (_placementLock)
            {
                string name = ResolveNameViaXid(nameId, mmidEntries, mmdxData);
                int nameIdx = GetOrAddMdxName(name);

                // Convert WoW coords to renderer: rendererX=MapOrigin-wowY, rendererY=MapOrigin-wowX, rendererZ=wowZ
                var placement = new MddfPlacement
                {
                    NameIndex = nameIdx,
                    UniqueId = (int)uniqueId,
                    Position = new Vector3(
                        WoWConstants.MapOrigin - rawY,
                        WoWConstants.MapOrigin - rawX,
                        rawZ),
                    Rotation = new Vector3(rotX, rotY, rotZ),
                    Scale = scale / 1024f
                };

                MddfPlacements.Add(placement);
                result.MddfPlacements.Add(placement);
            }
        }
    }

    private void ParseModfViaMwid(byte[] data, int offset, int size,
        byte[]? mwmoData, List<uint>? mwidEntries, TileLoadResult result)
    {
        int entrySize = 64; // 0x40
        int count = size / entrySize;

        for (int i = 0; i < count; i++)
        {
            int pos = offset + i * entrySize;
            if (pos + entrySize > data.Length) break;

            uint nameId = BitConverter.ToUInt32(data, pos);
            uint uniqueId = BitConverter.ToUInt32(data, pos + 4);
            // MODF position is (X, Z, Y): X=North, Z=Up(height), Y=West
            // Confirmed: LkToAlphaConverter passes positions unchanged → same layout as Alpha
            float rawX = BitConverter.ToSingle(data, pos + 8);   // North
            float rawZ = BitConverter.ToSingle(data, pos + 12);  // Up (height)
            float rawY = BitConverter.ToSingle(data, pos + 16);  // West
            // Rotation stored as (X, Z, Y)
            float rotX = BitConverter.ToSingle(data, pos + 20);
            float rotZ = BitConverter.ToSingle(data, pos + 24);
            float rotY = BitConverter.ToSingle(data, pos + 28);

            // Bounds stored as (X, Z, Y) pairs
            float bbMinX = BitConverter.ToSingle(data, pos + 32);
            float bbMinZ = BitConverter.ToSingle(data, pos + 36);
            float bbMinY = BitConverter.ToSingle(data, pos + 40);
            float bbMaxX = BitConverter.ToSingle(data, pos + 44);
            float bbMaxZ = BitConverter.ToSingle(data, pos + 48);
            float bbMaxY = BitConverter.ToSingle(data, pos + 52);
            ushort flags = BitConverter.ToUInt16(data, pos + 56);

            lock (_placementLock)
            {
                string name = ResolveNameViaXid(nameId, mwidEntries, mwmoData);
                int nameIdx = GetOrAddWmoName(name);

                // Convert WoW coords to renderer: rendererX=MapOrigin-wowY, rendererY=MapOrigin-wowX, rendererZ=wowZ
                // Note: MapOrigin-min > MapOrigin-max, so swap min/max after conversion
                var placement = new ModfPlacement
                {
                    NameIndex = nameIdx,
                    UniqueId = (int)uniqueId,
                    Position = new Vector3(
                        WoWConstants.MapOrigin - rawY,
                        WoWConstants.MapOrigin - rawX,
                        rawZ),
                    Rotation = new Vector3(rotX, rotY, rotZ),
                    BoundsMin = new Vector3(
                        WoWConstants.MapOrigin - bbMaxY,
                        WoWConstants.MapOrigin - bbMaxX,
                        bbMinZ),
                    BoundsMax = new Vector3(
                        WoWConstants.MapOrigin - bbMinY,
                        WoWConstants.MapOrigin - bbMinX,
                        bbMaxZ),
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
            {
                // Ghidra-verified (FUN_007b5950): MAIN is row-major [y][x].
                // Store raw index i = y*64+x. TerrainManager decodes:
                //   tx = i/64 = y (row), ty = i%64 = x (col).
                // This matches Alpha convention where tx=row, ty=col.
                tiles.Add(i);
            }
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
