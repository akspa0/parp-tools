using System.Numerics;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Rendering;
using WoWMapConverter.Core.Formats.Liquids;

namespace MdxViewer.Terrain;

/// <summary>
/// Loads WLW/WLQ/WLM loose liquid project files from the data source.
/// These files contain raw liquid heightmaps that are NOT read by the WoW client
/// but preserve liquid data for bodies of water that may have been deleted from
/// map tiles. Useful for visualization and manual tile reconstruction.
/// </summary>
public class WlLiquidLoader
{
    public sealed class WlTransformSettings
    {
        public bool Enabled { get; set; } = true;
        public bool SwapXYBeforeRotation { get; set; } = false;
        public Vector3 RotationDegrees { get; set; } = Vector3.Zero;
        public Vector3 Translation { get; set; } = Vector3.Zero;
    }

    // Tunable WL->renderer transform (editor-only files, no client runtime reference).
    public static WlTransformSettings TransformSettings { get; } = new();

    private readonly IDataSource _dataSource;
    private readonly string _mapName;
    private readonly string _mapDir;

    /// <summary>All loaded WL liquid bodies for this map.</summary>
    public List<WlLiquidBody> Bodies { get; } = new();

    /// <summary>Whether any WL files were found and loaded.</summary>
    public bool HasData => Bodies.Count > 0;

    public WlLiquidLoader(IDataSource dataSource, string mapName)
    {
        _dataSource = dataSource;
        _mapName = mapName;
        _mapDir = $"World\\Maps\\{mapName}";
    }

    /// <summary>
    /// Discover and load all WLW/WLQ/WLM files for the current map.
    /// Call once after the data source is ready.
    /// </summary>
    public void LoadAll()
    {
        Bodies.Clear();
        int loaded = 0;

        foreach (string ext in new[] { ".wlw", ".wlm", ".wlq" })
        {
            var files = _dataSource.GetFileList(ext);
            foreach (var path in files)
            {
                // Filter to current map directory
                if (!path.Replace('/', '\\').StartsWith(_mapDir, StringComparison.OrdinalIgnoreCase))
                    continue;

                try
                {
                    var data = _dataSource.ReadFile(path);
                    if (data == null || data.Length < 16) continue;

                    using var ms = new MemoryStream(data);
                    using var br = new BinaryReader(ms);
                    var wl = WlFile.Read(br, path);

                    if (wl.Blocks.Count == 0) continue;

                    var body = ConvertToLiquidBody(wl, path);
                    if (body != null)
                    {
                        Bodies.Add(body);
                        loaded++;
                    }
                }
                catch (Exception ex)
                {
                    ViewerLog.Error(ViewerLog.Category.Terrain, $"[WlLoader] Failed to load {path}: {ex.Message}");
                }
            }
        }

        if (loaded > 0)
            ViewerLog.Info(ViewerLog.Category.Terrain, $"[WlLoader] Loaded {loaded} WL liquid bodies for map '{_mapName}'");
    }

    /// <summary>
    /// Convert a parsed WlFile into a renderable WlLiquidBody.
    /// </summary>
    private static WlLiquidBody? ConvertToLiquidBody(WlFile wl, string sourcePath)
    {
        if (wl.Blocks.Count == 0) return null;

        string name = Path.GetFileNameWithoutExtension(sourcePath);

        // Map WL liquid type to our LiquidType enum
        LiquidType liquidType = wl.Header.LiquidType switch
        {
            WlLiquidType.Ocean => LiquidType.Ocean,
            WlLiquidType.Magma => LiquidType.Magma,
            WlLiquidType.Slime => LiquidType.Slime,
            _ => LiquidType.Water // StillWater, River, FastWater all render as water
        };

        // Collect all vertices from all blocks
        var allVertices = new List<Vector3>();
        var allIndices = new List<int>();

        foreach (var block in wl.Blocks)
        {
            int baseIdx = allVertices.Count;

            // WL blocks have 16 vertices in a 4x4 grid, stored in reverse order
            // (index 15 = lower-right corner first in file)
            // WL files are editor-only; apply configurable 3D transform for alignment.
            for (int i = 0; i < 16; i++)
            {
                var v = ApplyWlTransform(block.Vertices[i]);
                allVertices.Add(v);
            }

            // Build 3x3 quads from the 4x4 grid (reversed index order)
            // File order: 15,14,13,12 / 11,10,9,8 / 7,6,5,4 / 3,2,1,0
            // We need to build quads from the grid in standard row-major order
            // Remap: grid[row,col] = Vertices[15 - (row*4 + col)]
            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    int tl = 15 - (row * 4 + col);
                    int tr = 15 - (row * 4 + col + 1);
                    int bl = 15 - ((row + 1) * 4 + col);
                    int br = 15 - ((row + 1) * 4 + col + 1);

                    // Two triangles per quad (reversed winding for correct backface culling)
                    allIndices.Add(baseIdx + tl);
                    allIndices.Add(baseIdx + tr);
                    allIndices.Add(baseIdx + bl);

                    allIndices.Add(baseIdx + tr);
                    allIndices.Add(baseIdx + br);
                    allIndices.Add(baseIdx + bl);
                }
            }
        }

        if (allVertices.Count == 0) return null;

        // Compute bounding box
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        foreach (var v in allVertices)
        {
            min = Vector3.Min(min, v);
            max = Vector3.Max(max, v);
        }

        return new WlLiquidBody
        {
            Name = name,
            SourcePath = sourcePath,
            Type = liquidType,
            FileType = wl.Header.FileType,
            Vertices = allVertices.ToArray(),
            Indices = allIndices.ToArray(),
            BoundsMin = min,
            BoundsMax = max,
            BlockCount = wl.Blocks.Count
        };
    }

    private static Vector3 ApplyWlTransform(Vector3 input)
    {
        var s = TransformSettings;
        if (!s.Enabled)
            return input;

        Vector3 p = input;
        if (s.SwapXYBeforeRotation)
            p = new Vector3(p.Y, p.X, p.Z);

        float rx = MathF.PI / 180f * s.RotationDegrees.X;
        float ry = MathF.PI / 180f * s.RotationDegrees.Y;
        float rz = MathF.PI / 180f * s.RotationDegrees.Z;

        var rotation = Matrix4x4.CreateRotationX(rx)
                     * Matrix4x4.CreateRotationY(ry)
                     * Matrix4x4.CreateRotationZ(rz);

        p = Vector3.Transform(p, rotation);
        p += s.Translation;
        return p;
    }
}

/// <summary>
/// A single liquid body loaded from a WLW/WLQ/WLM file.
/// Contains pre-transformed vertices ready for GPU upload.
/// </summary>
public class WlLiquidBody
{
    public string Name { get; init; } = "";
    public string SourcePath { get; init; } = "";
    public LiquidType Type { get; init; }
    public WlFileType FileType { get; init; }
    public Vector3[] Vertices { get; init; } = Array.Empty<Vector3>();
    public int[] Indices { get; init; } = Array.Empty<int>();
    public Vector3 BoundsMin { get; init; }
    public Vector3 BoundsMax { get; init; }
    public int BlockCount { get; init; }
}
