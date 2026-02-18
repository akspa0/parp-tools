using System.Collections.Concurrent;
using System.Numerics;
using System.Text.Json;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using WoWMapConverter.Core.VLM;

namespace MdxViewer.Terrain;

/// <summary>
/// Loads a VLM dataset project folder and converts it into TerrainChunkData
/// for rendering in the viewer. VLM datasets are human-readable JSON exports
/// of WoW terrain data, produced by VlmDatasetExporter.
///
/// Expected folder structure:
///   project_root/
///     dataset/          ← per-tile JSON files (MapName_X_Y.json)
///     images/           ← minimap PNGs
///     shadows/          ← per-chunk shadow PNGs
///     masks/            ← per-chunk alpha mask PNGs
///     textures/         ← exported BLP→PNG textures
/// </summary>
public class VlmProjectLoader
{
    private readonly string _projectRoot;
    private readonly string _datasetDir;
    private readonly string _editsDatasetDir;

    /// <summary>Map name inferred from JSON filenames.</summary>
    public string MapName { get; private set; } = "VLM Project";

    /// <summary>All tile indices found in the dataset (tileX*64+tileY).</summary>
    public List<int> ExistingTiles { get; } = new();

    /// <summary>Tile coordinates for each existing tile.</summary>
    public List<(int tileX, int tileY)> TileCoords { get; } = new();

    /// <summary>Texture names per tile.</summary>
    public ConcurrentDictionary<(int, int), List<string>> TileTextures { get; } = new();

    /// <summary>Object placements from all tiles.</summary>
    public List<MddfPlacement> MddfPlacements { get; } = new();
    public List<ModfPlacement> ModfPlacements { get; } = new();

    /// <summary>MDX model names (from object placements).</summary>
    public List<string> MdxModelNames { get; } = new();

    /// <summary>WMO model names (from object placements).</summary>
    public List<string> WmoModelNames { get; } = new();

    /// <summary>Chunk positions for diagnostics.</summary>
    public List<Vector3> LastLoadedChunkPositions { get; } = new();

    // Dedup + thread safety for background loading
    private readonly object _placementLock = new();
    private readonly HashSet<uint> _seenMddfIds = new();
    private readonly HashSet<uint> _seenModfIds = new();
    private readonly Dictionary<string, int> _mdxNameIndex = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, int> _wmoNameIndex = new(StringComparer.OrdinalIgnoreCase);

    // Cached parsed JSON per tile
    private readonly Dictionary<(int, int), string> _tileJsonPaths = new();
    private int _loadDiagCount = 0;

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
    };

    private static readonly JsonSerializerOptions _jsonOptionsIndented = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
        WriteIndented = true
    };

    public VlmProjectLoader(string projectRoot)
    {
        _projectRoot = projectRoot;
        _datasetDir = Path.Combine(projectRoot, "dataset");
        _editsDatasetDir = Path.Combine(projectRoot, "edits", "dataset");

        if (!Directory.Exists(_datasetDir))
            throw new DirectoryNotFoundException($"VLM dataset directory not found: {_datasetDir}");

        ScanTiles();
    }

    public string ProjectRoot => _projectRoot;
    public string DatasetDir => _datasetDir;
    public string EditsDatasetDir => _editsDatasetDir;

    private void ScanTiles()
    {
        var jsonFiles = Directory.GetFiles(_datasetDir, "*.json")
            .Where(f => !Path.GetFileName(f).Equals("texture_database.json", StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f)
            .ToArray();

        if (jsonFiles.Length == 0)
            throw new FileNotFoundException($"No tile JSON files found in {_datasetDir}");

        foreach (var jsonPath in jsonFiles)
        {
            var name = Path.GetFileNameWithoutExtension(jsonPath);
            // Parse tile coords from filename: MapName_X_Y
            // The VLM exporter uses: x = tileIndex % 64 (= wowTileY), y = tileIndex / 64 (= wowTileX)
            // So filename X is actually WoW tileY, and filename Y is WoW tileX.
            var parts = name.Split('_');
            if (parts.Length >= 3 &&
                int.TryParse(parts[^2], out int fileX) &&
                int.TryParse(parts[^1], out int fileY))
            {
                int tileX = fileY;  // fileY = tileIndex / 64 = WoW tileX
                int tileY = fileX;  // fileX = tileIndex % 64 = WoW tileY
                if (MapName == "VLM Project")
                    MapName = string.Join("_", parts[..^2]);

                int idx = tileX * 64 + tileY;
                ExistingTiles.Add(idx);
                TileCoords.Add((tileX, tileY));
                _tileJsonPaths[(tileX, tileY)] = jsonPath;
            }
        }

        ViewerLog.Important(ViewerLog.Category.Vlm, $"Found {ExistingTiles.Count} tiles for map '{MapName}' in {_datasetDir}");
    }

    /// <summary>
    /// Load a single tile and return terrain chunk data + placements.
    /// </summary>
    public TileLoadResult LoadTile(int tileX, int tileY)
    {
        var result = new TileLoadResult();

        if (!TryGetEffectiveTileJsonPath(tileX, tileY, out var jsonPath))
            return result;

        VlmTrainingSample? sample;
        try
        {
            var json = File.ReadAllText(jsonPath);
            sample = JsonSerializer.Deserialize<VlmTrainingSample>(json, _jsonOptions);
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Vlm, $"Failed to parse {jsonPath}: {ex.Message}");
            return result;
        }

        if (sample?.TerrainData == null)
            return result;

        var data = sample.TerrainData;

        // Diagnostic: log first tile details
        if (_loadDiagCount < 2)
        {
            ViewerLog.Info(ViewerLog.Category.Vlm, $"Tile ({tileX},{tileY}): Heights={data.Heights?.Length ?? 0}, " +
                $"ChunkPositions={data.ChunkPositions?.Length ?? 0}, Textures={data.Textures?.Count ?? 0}, " +
                $"IsInterleaved={data.IsInterleaved}");
            if (data.Heights != null && data.Heights.Length > 0)
            {
                var h0 = data.Heights[0];
                ViewerLog.Debug(ViewerLog.Category.Vlm, $"  Heights[0]: idx={h0.ChunkIndex}, h.len={h0.Heights?.Length ?? 0}, " +
                    $"first5={string.Join(",", (h0.Heights ?? Array.Empty<float>()).Take(5).Select(v => v.ToString("F1")))}");
            }
            if (data.ChunkPositions != null && data.ChunkPositions.Length >= 3)
                ViewerLog.Debug(ViewerLog.Category.Vlm, $"  ChunkPos[0]: ({data.ChunkPositions[0]:F1}, {data.ChunkPositions[1]:F1}, {data.ChunkPositions[2]:F1})");
            _loadDiagCount++;
        }

        // Store texture names for this tile
        if (data.Textures != null && data.Textures.Count > 0)
            TileTextures.TryAdd((tileX, tileY), data.Textures);

        // Convert heights + layers into TerrainChunkData
        if (data.Heights != null)
        {
            foreach (var chunkHeights in data.Heights)
            {
                var chunkData = ConvertChunk(chunkHeights, data, tileX, tileY);
                if (chunkData != null)
                    result.Chunks.Add(chunkData);
            }
        }

        // Convert object placements
        if (data.Objects != null)
        {
            foreach (var obj in data.Objects)
            {
                if (obj.Category == "m2")
                    AddMddfPlacement(obj, result);
                else if (obj.Category == "wmo")
                    AddModfPlacement(obj, result);
            }
        }

        return result;
    }

    public bool TryGetEffectiveTileJsonPath(int tileX, int tileY, out string jsonPath)
    {
        jsonPath = string.Empty;
        if (!_tileJsonPaths.TryGetValue((tileX, tileY), out var basePath))
            return false;

        var editPath = Path.Combine(_editsDatasetDir, Path.GetFileName(basePath));
        jsonPath = File.Exists(editPath) ? editPath : basePath;
        return true;
    }

    public bool TryGetEditTileJsonPath(int tileX, int tileY, out string jsonPath)
    {
        jsonPath = string.Empty;
        if (!_tileJsonPaths.TryGetValue((tileX, tileY), out var basePath))
            return false;

        jsonPath = Path.Combine(_editsDatasetDir, Path.GetFileName(basePath));
        return true;
    }

    public bool TryGetTileJsonPath(int tileX, int tileY, out string jsonPath)
    {
        return _tileJsonPaths.TryGetValue((tileX, tileY), out jsonPath!);
    }

    public bool TryLoadRawSample(int tileX, int tileY, out VlmTrainingSample sample)
    {
        sample = default!;

        if (!TryGetEffectiveTileJsonPath(tileX, tileY, out var jsonPath))
            return false;

        try
        {
            var json = File.ReadAllText(jsonPath);
            var parsed = JsonSerializer.Deserialize<VlmTrainingSample>(json, _jsonOptions);
            if (parsed?.TerrainData == null)
                return false;
            sample = parsed;
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Save an edited tile sample to the edits folder (never overwrites source dataset JSON).
    /// </summary>
    public bool TrySaveEditedSample(int tileX, int tileY, VlmTrainingSample sample)
    {
        if (!TryGetEditTileJsonPath(tileX, tileY, out var jsonPath))
            return false;

        try
        {
            Directory.CreateDirectory(_editsDatasetDir);
            var json = JsonSerializer.Serialize(sample, _jsonOptionsIndented);
            File.WriteAllText(jsonPath, json);
            return true;
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Vlm, $"Failed to save tile ({tileX},{tileY}) JSON: {ex.Message}");
            return false;
        }
    }

    private TerrainChunkData? ConvertChunk(VlmChunkHeights chunkHeights, VlmTerrainData data, int tileX, int tileY)
    {
        if (chunkHeights.Heights == null || chunkHeights.Heights.Length < 145)
            return null;

        int chunkIndex = chunkHeights.ChunkIndex;
        int chunkX = chunkIndex % 16;
        int chunkY = chunkIndex / 16;

        // Heights: VLM stores in Alpha non-interleaved format (81 outer + 64 inner).
        // Check IsInterleaved flag to determine if we need to reorder.
        float[] heights;
        if (data.IsInterleaved)
        {
            // Already interleaved — use directly
            heights = chunkHeights.Heights;
        }
        else
        {
            // Non-interleaved (Alpha format): reorder to interleaved
            heights = ReorderToInterleaved(chunkHeights.Heights);
        }

        // Normals from chunk_layers (must reinterleave if non-interleaved, same as heights)
        Vector3[] normals = ExtractNormals(data, chunkIndex, !data.IsInterleaved);

        // Hole mask
        int holeMask = 0;
        if (data.Holes != null && chunkIndex < data.Holes.Length)
            holeMask = data.Holes[chunkIndex];

        // Layers
        var layers = ExtractLayers(data, chunkIndex);

        // Alpha maps
        var alphaMaps = ExtractAlphaMaps(data, chunkIndex);

        // Shadow map
        byte[]? shadowMap = ExtractShadowMap(data, chunkIndex);

        // World position from chunk_positions
        // VLM exporter stores positions already in WoW world coords:
        //   Alpha: posX = (32-tileX)*533 - idxX*33, posY = (32-tileY)*533 - idxY*33
        //   LK:    posX = Origin - y*Tile - IndexY*Chunk, posY = Origin - x*Tile - IndexX*Chunk
        // The native AlphaTerrainAdapter uses:
        //   worldX = MapOrigin - tileX*ChunkSize - chunkY*chunkSmall  (north-south)
        //   worldY = MapOrigin - tileY*ChunkSize - chunkX*chunkSmall  (east-west)
        // Use tile/chunk indices directly for consistency with native path.
        float chunkSmall = WoWConstants.ChunkSize / 16f;
        float worldX = WoWConstants.MapOrigin - tileX * WoWConstants.ChunkSize - chunkY * chunkSmall;
        float worldY = WoWConstants.MapOrigin - tileY * WoWConstants.ChunkSize - chunkX * chunkSmall;

        LastLoadedChunkPositions.Add(new Vector3(worldX, worldY, 0f));

        var liquid = ExtractLiquid(data, chunkIndex, tileX, tileY, chunkX, chunkY, new Vector3(worldX, worldY, 0f));

        return new TerrainChunkData
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
        };
    }

    private static LiquidChunkData? ExtractLiquid(VlmTerrainData data, int chunkIndex,
        int tileX, int tileY, int chunkX, int chunkY, Vector3 worldPos)
    {
        var liquids = data.Liquids;
        if (liquids == null || liquids.Length == 0)
            return null;

        var l = liquids.FirstOrDefault(x => x.ChunkIndex == chunkIndex);
        if (l == null)
            return null;

        LiquidType type = l.LiquidType switch
        {
            1 => LiquidType.Ocean,
            2 => LiquidType.Magma,
            3 => LiquidType.Slime,
            _ => LiquidType.Water
        };

        var heights = ConvertHeightsTo9x9(l.Heights, l.MinHeight);
        if (heights == null)
            return null;

        return new LiquidChunkData
        {
            MinHeight = l.MinHeight,
            MaxHeight = l.MaxHeight,
            Heights = heights,
            VertexData = new uint[81],
            TileGrid = new float[16],
            Type = type,
            WorldPosition = worldPos,
            TileX = tileX,
            TileY = tileY,
            ChunkX = chunkX,
            ChunkY = chunkY
        };
    }

    private static float[]? ConvertHeightsTo9x9(float[]? src, float fallbackHeight)
    {
        if (src == null || src.Length == 0)
        {
            var flat = new float[81];
            Array.Fill(flat, fallbackHeight);
            return flat;
        }

        if (src.Length >= 81)
        {
            var dst = new float[81];
            Array.Copy(src, dst, 81);
            return dst;
        }

        // MH2O can store partial sub-rect height grids (<= 9x9). Infer a small grid size and place it
        // in the top-left of a 9x9 buffer.
        int bestW = 0;
        int bestH = 0;
        for (int h = 2; h <= 9; h++)
        {
            if (src.Length % h != 0) continue;
            int w = src.Length / h;
            if (w < 2 || w > 9) continue;
            if (w * h == src.Length)
            {
                bestW = w;
                bestH = h;
                break;
            }
        }

        if (bestW == 0 || bestH == 0)
        {
            var flat = new float[81];
            Array.Fill(flat, fallbackHeight);
            return flat;
        }

        var outHeights = new float[81];
        Array.Fill(outHeights, fallbackHeight);

        for (int y = 0; y < bestH && y < 9; y++)
        {
            for (int x = 0; x < bestW && x < 9; x++)
            {
                int srcIdx = y * bestW + x;
                outHeights[y * 9 + x] = src[srcIdx];
            }
        }

        return outHeights;
    }

    /// <summary>
    /// Reorder Alpha non-interleaved heights (81 outer + 64 inner) to interleaved (9-8-9-8...).
    /// </summary>
    private static float[] ReorderToInterleaved(float[] src)
    {
        if (src.Length < 145) return src;

        var dst = new float[145];
        int di = 0;
        for (int row = 0; row < 17; row++)
        {
            if (row % 2 == 0)
            {
                // Outer row (9 vertices)
                int outerRow = row / 2;
                for (int col = 0; col < 9; col++)
                    dst[di++] = src[outerRow * 9 + col];
            }
            else
            {
                // Inner row (8 vertices)
                int innerRow = row / 2;
                for (int col = 0; col < 8; col++)
                    dst[di++] = src[81 + innerRow * 8 + col];
            }
        }
        return dst;
    }

    private static Vector3[] ExtractNormals(VlmTerrainData data, int chunkIndex, bool needsReinterleave)
    {
        var normals = new Vector3[145];

        // Try to get normals from chunk_layers
        var chunkLayer = data.ChunkLayers?.FirstOrDefault(c => c.ChunkIndex == chunkIndex);
        if (chunkLayer?.Normals != null && chunkLayer.Normals.Length >= 435)
        {
            if (needsReinterleave)
            {
                // MCNR in Alpha format is non-interleaved: 81 outer normals (243 bytes) then 64 inner normals (192 bytes).
                // Must reinterleave to match the height vertex layout (9-8-9-8... pattern).
                int destIdx = 0;
                for (int row = 0; row < 17; row++)
                {
                    if (row % 2 == 0)
                    {
                        int outerRow = row / 2;
                        for (int col = 0; col < 9; col++)
                        {
                            int srcIdx = (outerRow * 9 + col) * 3;
                            normals[destIdx++] = DecodeNormal(chunkLayer.Normals, srcIdx);
                        }
                    }
                    else
                    {
                        int innerRow = row / 2;
                        for (int col = 0; col < 8; col++)
                        {
                            int srcIdx = (81 + innerRow * 8 + col) * 3;
                            normals[destIdx++] = DecodeNormal(chunkLayer.Normals, srcIdx);
                        }
                    }
                }
            }
            else
            {
                // Already interleaved (LK format) — read linearly
                for (int i = 0; i < 145; i++)
                    normals[i] = DecodeNormal(chunkLayer.Normals, i * 3);
            }
            return normals;
        }

        // Default: up-facing normals
        for (int i = 0; i < 145; i++)
            normals[i] = Vector3.UnitZ;
        return normals;
    }

    private static Vector3 DecodeNormal(sbyte[] data, int offset)
    {
        if (offset + 2 >= data.Length) return Vector3.UnitZ;

        // MCNR stores normals as signed bytes: X, Z, Y (WoW convention)
        float nx = data[offset] / 127f;
        float nz = data[offset + 1] / 127f;
        float ny = data[offset + 2] / 127f;
        var n = new Vector3(nx, ny, nz);
        float len = n.Length();
        return len > 0.001f ? n / len : Vector3.UnitZ;
    }

    private static TerrainLayer[] ExtractLayers(VlmTerrainData data, int chunkIndex)
    {
        var chunkLayer = data.ChunkLayers?.FirstOrDefault(c => c.ChunkIndex == chunkIndex);
        if (chunkLayer?.Layers == null || chunkLayer.Layers.Length == 0)
            return Array.Empty<TerrainLayer>();

        var layers = new TerrainLayer[chunkLayer.Layers.Length];
        for (int i = 0; i < chunkLayer.Layers.Length; i++)
        {
            var src = chunkLayer.Layers[i];
            layers[i] = new TerrainLayer
            {
                TextureIndex = (int)src.TextureId,
                Flags = src.Flags,
                AlphaOffset = src.AlphaOffset,
                EffectId = src.EffectId
            };
        }
        return layers;
    }

    private static Dictionary<int, byte[]> ExtractAlphaMaps(VlmTerrainData data, int chunkIndex)
    {
        var maps = new Dictionary<int, byte[]>();

        var chunkLayer = data.ChunkLayers?.FirstOrDefault(c => c.ChunkIndex == chunkIndex);
        if (chunkLayer?.Layers == null) return maps;

        for (int i = 1; i < chunkLayer.Layers.Length; i++)
        {
            var layer = chunkLayer.Layers[i];
            if (layer.AlphaBitsBase64 == null) continue;

            try
            {
                var raw = Convert.FromBase64String(layer.AlphaBitsBase64);

                byte[] alpha;
                if (raw.Length == 2048)
                {
                    // 4-bit alpha: expand to 8-bit (64×64)
                    alpha = new byte[4096];
                    for (int j = 0; j < 2048 && j < raw.Length; j++)
                    {
                        byte packed = raw[j];
                        alpha[j * 2] = (byte)((packed & 0x0F) * 17);
                        alpha[j * 2 + 1] = (byte)((packed >> 4) * 17);
                    }
                }
                else if (raw.Length >= 4096)
                {
                    alpha = new byte[4096];
                    Array.Copy(raw, alpha, 4096);
                }
                else
                {
                    continue;
                }

                maps[i] = alpha;
            }
            catch { }
        }

        return maps;
    }

    private static byte[]? ExtractShadowMap(VlmTerrainData data, int chunkIndex)
    {
        var bits = data.ShadowBits?.FirstOrDefault(s => s.ChunkIndex == chunkIndex);
        if (bits?.BitsBase64 == null) return null;

        try
        {
            var raw = Convert.FromBase64String(bits.BitsBase64);
            if (raw.Length == 0) return null;

            // Expand MCSH bits to bytes (64×64).
            // Format: 64 rows × 8 bytes/row = 512 bytes. Each bit: 1=shadowed, 0=lit.
            // Output: 0=lit (black shadow disabled), 255=shadowed (for shader: shadow=1.0 darkens).
            // Matches ShadowMapService.ReadShadow polarity.
            var shadow = new byte[64 * 64];
            for (int y = 0; y < 64; y++)
            {
                for (int x = 0; x < 64; x++)
                {
                    int byteIndex = y * 8 + (x / 8);
                    int bitIndex = x % 8;

                    if (byteIndex < raw.Length)
                    {
                        bool isShadowed = (raw[byteIndex] & (1 << bitIndex)) != 0;
                        shadow[y * 64 + x] = isShadowed ? (byte)255 : (byte)0;
                    }
                }
            }
            return shadow;
        }
        catch { return null; }
    }

    private void AddMddfPlacement(VlmObjectPlacement obj, TileLoadResult result)
    {
        lock (_placementLock)
        {
            if (!_seenMddfIds.Add(obj.UniqueId)) return;

            int nameIdx = GetOrAddMdxName(obj.Name);
            var placement = new MddfPlacement
            {
                NameIndex = nameIdx,
                UniqueId = (int)obj.UniqueId,
                Position = new Vector3(
                    WoWConstants.MapOrigin - obj.Y,
                    WoWConstants.MapOrigin - obj.X,
                    obj.Z),
                Rotation = new Vector3(obj.RotX, obj.RotY, obj.RotZ),
                Scale = obj.Scale
            };

            MddfPlacements.Add(placement);
            result.MddfPlacements.Add(placement);
        }
    }

    private void AddModfPlacement(VlmObjectPlacement obj, TileLoadResult result)
    {
        lock (_placementLock)
        {
            if (!_seenModfIds.Add(obj.UniqueId)) return;

            int nameIdx = GetOrAddWmoName(obj.Name);
            var placement = new ModfPlacement
            {
                NameIndex = nameIdx,
                UniqueId = (int)obj.UniqueId,
                Position = new Vector3(
                    WoWConstants.MapOrigin - obj.Y,
                    WoWConstants.MapOrigin - obj.X,
                    obj.Z),
                Rotation = new Vector3(obj.RotX, obj.RotY, obj.RotZ),
                BoundsMin = obj.BoundsMin != null && obj.BoundsMin.Length >= 3
                    ? new Vector3(obj.BoundsMin[0], obj.BoundsMin[1], obj.BoundsMin[2])
                    : Vector3.Zero,
                BoundsMax = obj.BoundsMax != null && obj.BoundsMax.Length >= 3
                    ? new Vector3(obj.BoundsMax[0], obj.BoundsMax[1], obj.BoundsMax[2])
                    : Vector3.Zero,
                Flags = 0
            };

            ModfPlacements.Add(placement);
            result.ModfPlacements.Add(placement);
        }
    }

    private int GetOrAddMdxName(string name)
    {
        if (_mdxNameIndex.TryGetValue(name, out int idx)) return idx;
        idx = MdxModelNames.Count;
        MdxModelNames.Add(name);
        _mdxNameIndex[name] = idx;
        return idx;
    }

    private int GetOrAddWmoName(string name)
    {
        if (_wmoNameIndex.TryGetValue(name, out int idx)) return idx;
        idx = WmoModelNames.Count;
        WmoModelNames.Add(name);
        _wmoNameIndex[name] = idx;
        return idx;
    }

    /// <summary>
    /// Try to resolve a texture path to a PNG file in the project's textures/ directory.
    /// </summary>
    public string? ResolveTexturePath(string textureName)
    {
        // VLM exporter writes tileset textures to tilesets/ folder as flat PNGs
        var pngName = Path.ChangeExtension(Path.GetFileName(textureName), ".png");

        // Try tilesets/ first (where the exporter writes them)
        var pngPath = Path.Combine(_projectRoot, "tilesets", pngName);
        if (File.Exists(pngPath)) return pngPath;

        // Try textures/ as fallback
        pngPath = Path.Combine(_projectRoot, "textures", pngName);
        if (File.Exists(pngPath)) return pngPath;

        // Try with full relative path under tilesets/
        var relPath = textureName.Replace('\\', '/').Replace('/', Path.DirectorySeparatorChar);
        pngPath = Path.Combine(_projectRoot, "tilesets", Path.ChangeExtension(relPath, ".png"));
        if (File.Exists(pngPath)) return pngPath;

        return null;
    }
}
