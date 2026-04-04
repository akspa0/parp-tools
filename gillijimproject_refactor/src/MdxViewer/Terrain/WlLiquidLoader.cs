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
    public enum WlBodyGroupingMode
    {
        FileWelded,
        PlaneWelded,
        BlockUnwelded,
    }

    public sealed class WlTransformSettings
    {
        public bool Enabled { get; set; } = true;
        public bool SwapXYBeforeRotation { get; set; } = false;
        public Vector3 RotationDegrees { get; set; } = Vector3.Zero;
        public Vector3 Translation { get; set; } = Vector3.Zero;
        public WlBodyGroupingMode GroupingMode { get; set; } = WlBodyGroupingMode.PlaneWelded;
        public float PlaneHeightTolerance { get; set; } = 0.5f;
    }

    // Tunable WL->renderer transform (editor-only files, no client runtime reference).
    public static WlTransformSettings TransformSettings { get; } = new();

    private readonly IDataSource _dataSource;
    private readonly string _mapName;
    private readonly string _mapDir;

    /// <summary>All loaded WL liquid bodies for this map.</summary>
    public List<WlLiquidBody> Bodies { get; } = new();

    public int SourceFileCount { get; private set; }

    public int TotalBlockCount { get; private set; }

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
        SourceFileCount = 0;
        TotalBlockCount = 0;
        int loaded = 0;

        foreach (string ext in new[] { ".wlw", ".wlm", ".wlq", ".wll" })
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

                    var bodies = ConvertToLiquidBodies(wl, path);
                    if (bodies.Count > 0)
                    {
                        Bodies.AddRange(bodies);
                        loaded += bodies.Count;
                        SourceFileCount++;
                        TotalBlockCount += wl.Blocks.Count;
                    }
                }
                catch (Exception ex)
                {
                    ViewerLog.Error(ViewerLog.Category.Terrain, $"[WlLoader] Failed to load {path}: {ex.Message}");
                }
            }
        }

        if (loaded > 0)
            ViewerLog.Info(ViewerLog.Category.Terrain,
                $"[WlLoader] Loaded {loaded} WL liquid bodies from {SourceFileCount} WL files ({TotalBlockCount} blocks) for map '{_mapName}'");
    }

    /// <summary>
    /// Convert a parsed WlFile into one or more renderable WlLiquidBody entries.
    /// </summary>
    private static List<WlLiquidBody> ConvertToLiquidBodies(WlFile wl, string sourcePath)
    {
        var bodies = new List<WlLiquidBody>();
        if (wl.Blocks.Count == 0)
            return bodies;

        string name = Path.GetFileNameWithoutExtension(sourcePath);

        // Map WL liquid type to our LiquidType enum
        LiquidType liquidType = wl.Header.LiquidType switch
        {
            WlLiquidType.Ocean => LiquidType.Ocean,
            WlLiquidType.Magma => LiquidType.Magma,
            WlLiquidType.Slime => LiquidType.Slime,
            _ => LiquidType.Water // StillWater, River, FastWater all render as water
        };

        var buildBlocks = BuildBodyBlocks(wl);
        if (buildBlocks.Count == 0)
            return bodies;

        List<List<WlBodyBlock>> groupedBlocks = TransformSettings.GroupingMode switch
        {
            WlBodyGroupingMode.BlockUnwelded => BuildPerBlockGroups(buildBlocks),
            WlBodyGroupingMode.PlaneWelded => BuildPlaneGroups(buildBlocks, TransformSettings.PlaneHeightTolerance),
            _ => new List<List<WlBodyBlock>> { buildBlocks },
        };

        for (int bodyIndex = 0; bodyIndex < groupedBlocks.Count; bodyIndex++)
        {
            WlLiquidBody? body = BuildLiquidBody(
                name,
                sourcePath,
                liquidType,
                wl.Header.FileType,
                groupedBlocks[bodyIndex],
                bodyIndex,
                groupedBlocks.Count);

            if (body != null)
                bodies.Add(body);
        }

        return bodies;
    }

    private static WlLiquidBody? BuildLiquidBody(
        string sourceName,
        string sourcePath,
        LiquidType liquidType,
        WlFileType fileType,
        List<WlBodyBlock> blocks,
        int bodyIndex,
        int totalBodies)
    {
        if (blocks.Count == 0)
            return null;

        var allVertices = new List<Vector3>();
        var allIndices = new List<int>();

        foreach (WlBodyBlock block in blocks)
        {
            int baseIdx = allVertices.Count;

            for (int i = 0; i < 16; i++)
                allVertices.Add(block.TransformedVertices[i]);

            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    int tl = 15 - (row * 4 + col);
                    int tr = 15 - (row * 4 + col + 1);
                    int bl = 15 - ((row + 1) * 4 + col);
                    int br = 15 - ((row + 1) * 4 + col + 1);

                    allIndices.Add(baseIdx + tl);
                    allIndices.Add(baseIdx + tr);
                    allIndices.Add(baseIdx + bl);

                    allIndices.Add(baseIdx + tr);
                    allIndices.Add(baseIdx + br);
                    allIndices.Add(baseIdx + bl);
                }
            }
        }

        if (allVertices.Count == 0)
            return null;

        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        foreach (var v in allVertices)
        {
            min = Vector3.Min(min, v);
            max = Vector3.Max(max, v);
        }

        float heightMin = float.MaxValue;
        float heightMax = float.MinValue;
        float heightSum = 0f;
        float coordMinX = float.MaxValue;
        float coordMaxX = float.MinValue;
        float coordMinY = float.MaxValue;
        float coordMaxY = float.MinValue;
        int metadataNonZeroMin = int.MaxValue;
        int metadataNonZeroMax = int.MinValue;
        var metadataPatterns = new HashSet<int>();
        var sourceBlockIndices = new int[blocks.Count];

        for (int blockIndex = 0; blockIndex < blocks.Count; blockIndex++)
        {
            WlBodyBlock block = blocks[blockIndex];
            sourceBlockIndices[blockIndex] = block.BlockIndex;
            heightMin = MathF.Min(heightMin, block.MinHeight);
            heightMax = MathF.Max(heightMax, block.MaxHeight);
            heightSum += block.AverageHeight;
            coordMinX = MathF.Min(coordMinX, block.CoordX);
            coordMaxX = MathF.Max(coordMaxX, block.CoordX);
            coordMinY = MathF.Min(coordMinY, block.CoordY);
            coordMaxY = MathF.Max(coordMaxY, block.CoordY);
            metadataNonZeroMin = Math.Min(metadataNonZeroMin, block.NonZeroMetadataWords);
            metadataNonZeroMax = Math.Max(metadataNonZeroMax, block.NonZeroMetadataWords);
            metadataPatterns.Add(block.MetadataPatternHash);
        }

        string groupLabel = TransformSettings.GroupingMode switch
        {
            WlBodyGroupingMode.BlockUnwelded => $"block {sourceBlockIndices[0]:D3}",
            WlBodyGroupingMode.PlaneWelded => $"plane {bodyIndex + 1:D2}",
            _ => "file",
        };

        string bodyName = TransformSettings.GroupingMode == WlBodyGroupingMode.FileWelded && totalBodies == 1
            ? sourceName
            : $"{sourceName} [{groupLabel}]";

        return new WlLiquidBody
        {
            BodyKey = $"{sourcePath}#{TransformSettings.GroupingMode}:{bodyIndex}",
            Name = bodyName,
            SourcePath = sourcePath,
            Type = liquidType,
            FileType = fileType,
            GroupingMode = TransformSettings.GroupingMode,
            GroupLabel = groupLabel,
            Vertices = allVertices.ToArray(),
            Indices = allIndices.ToArray(),
            BoundsMin = min,
            BoundsMax = max,
            BlockCount = blocks.Count,
            SourceBlockIndices = sourceBlockIndices,
            MinHeight = heightMin,
            MaxHeight = heightMax,
            AverageHeight = heightSum / blocks.Count,
            CoordMinX = coordMinX,
            CoordMaxX = coordMaxX,
            CoordMinY = coordMinY,
            CoordMaxY = coordMaxY,
            MetadataPatternCount = metadataPatterns.Count,
            MetadataNonZeroMin = metadataNonZeroMin == int.MaxValue ? 0 : metadataNonZeroMin,
            MetadataNonZeroMax = metadataNonZeroMax == int.MinValue ? 0 : metadataNonZeroMax
        };
    }

    private static List<WlBodyBlock> BuildBodyBlocks(WlFile wl)
    {
        var blocks = new List<WlBodyBlock>(wl.Blocks.Count);
        for (int blockIndex = 0; blockIndex < wl.Blocks.Count; blockIndex++)
        {
            WlBlock source = wl.Blocks[blockIndex];
            var transformedVertices = new Vector3[source.Vertices.Length];
            for (int vertexIndex = 0; vertexIndex < source.Vertices.Length; vertexIndex++)
                transformedVertices[vertexIndex] = ApplyWlTransform(source.Vertices[vertexIndex]);

            float[] heights = source.GetHeights4x4();
            float minX = float.MaxValue;
            float maxX = float.MinValue;
            float minY = float.MaxValue;
            float maxY = float.MinValue;
            float minHeight = float.MaxValue;
            float maxHeight = float.MinValue;
            float heightSum = 0f;

            foreach (Vector3 vertex in source.Vertices)
            {
                minX = MathF.Min(minX, vertex.X);
                maxX = MathF.Max(maxX, vertex.X);
                minY = MathF.Min(minY, vertex.Y);
                maxY = MathF.Max(maxY, vertex.Y);
                minHeight = MathF.Min(minHeight, vertex.Z);
                maxHeight = MathF.Max(maxHeight, vertex.Z);
                heightSum += vertex.Z;
            }

            var hash = new HashCode();
            int nonZeroMetadataWords = 0;
            for (int i = 0; i < source.Data.Length; i++)
            {
                hash.Add(source.Data[i]);
                if (source.Data[i] != 0)
                    nonZeroMetadataWords++;
            }

            blocks.Add(new WlBodyBlock
            {
                BlockIndex = blockIndex,
                CoordX = source.CoordX,
                CoordY = source.CoordY,
                TransformedVertices = transformedVertices,
                Heights4x4 = heights,
                MinX = minX,
                MaxX = maxX,
                MinY = minY,
                MaxY = maxY,
                MinHeight = minHeight,
                MaxHeight = maxHeight,
                AverageHeight = heightSum / source.Vertices.Length,
                NonZeroMetadataWords = nonZeroMetadataWords,
                MetadataPatternHash = hash.ToHashCode()
            });
        }

        return blocks;
    }

    private static List<List<WlBodyBlock>> BuildPerBlockGroups(List<WlBodyBlock> blocks)
    {
        var groups = new List<List<WlBodyBlock>>(blocks.Count);
        foreach (WlBodyBlock block in blocks)
            groups.Add(new List<WlBodyBlock> { block });
        return groups;
    }

    private static List<List<WlBodyBlock>> BuildPlaneGroups(List<WlBodyBlock> blocks, float heightTolerance)
    {
        if (blocks.Count <= 1)
            return new List<List<WlBodyBlock>> { new List<WlBodyBlock>(blocks) };

        const float xyTolerance = 0.5f;
        var visited = new bool[blocks.Count];
        var groups = new List<List<WlBodyBlock>>();
        var queue = new Queue<int>();

        for (int startIndex = 0; startIndex < blocks.Count; startIndex++)
        {
            if (visited[startIndex])
                continue;

            visited[startIndex] = true;
            queue.Enqueue(startIndex);
            var group = new List<WlBodyBlock>();

            while (queue.Count > 0)
            {
                int currentIndex = queue.Dequeue();
                WlBodyBlock current = blocks[currentIndex];
                group.Add(current);

                for (int candidateIndex = 0; candidateIndex < blocks.Count; candidateIndex++)
                {
                    if (visited[candidateIndex])
                        continue;

                    if (!AreBlocksConnectedOnPlane(current, blocks[candidateIndex], xyTolerance, heightTolerance))
                        continue;

                    visited[candidateIndex] = true;
                    queue.Enqueue(candidateIndex);
                }
            }

            group.Sort((left, right) => left.BlockIndex.CompareTo(right.BlockIndex));
            groups.Add(group);
        }

        groups.Sort((left, right) => left[0].BlockIndex.CompareTo(right[0].BlockIndex));
        return groups;
    }

    private static bool AreBlocksConnectedOnPlane(WlBodyBlock left, WlBodyBlock right, float xyTolerance, float heightTolerance)
    {
        if (MathF.Abs(left.MaxX - right.MinX) <= xyTolerance && RangesOverlap(left.MinY, left.MaxY, right.MinY, right.MaxY, xyTolerance))
            return EdgeHeightsMatch(left.Heights4x4, right.Heights4x4, WlBlockAdjacencyMode.RightToLeft, heightTolerance);

        if (MathF.Abs(right.MaxX - left.MinX) <= xyTolerance && RangesOverlap(left.MinY, left.MaxY, right.MinY, right.MaxY, xyTolerance))
            return EdgeHeightsMatch(left.Heights4x4, right.Heights4x4, WlBlockAdjacencyMode.LeftToRight, heightTolerance);

        if (MathF.Abs(left.MaxY - right.MinY) <= xyTolerance && RangesOverlap(left.MinX, left.MaxX, right.MinX, right.MaxX, xyTolerance))
            return EdgeHeightsMatch(left.Heights4x4, right.Heights4x4, WlBlockAdjacencyMode.BottomToTop, heightTolerance);

        if (MathF.Abs(right.MaxY - left.MinY) <= xyTolerance && RangesOverlap(left.MinX, left.MaxX, right.MinX, right.MaxX, xyTolerance))
            return EdgeHeightsMatch(left.Heights4x4, right.Heights4x4, WlBlockAdjacencyMode.TopToBottom, heightTolerance);

        return false;
    }

    private static bool RangesOverlap(float minA, float maxA, float minB, float maxB, float tolerance)
    {
        return maxA + tolerance >= minB && maxB + tolerance >= minA;
    }

    private static bool EdgeHeightsMatch(
        float[] heightsA,
        float[] heightsB,
        WlBlockAdjacencyMode adjacencyMode,
        float heightTolerance = 0.5f)
    {
        for (int i = 0; i < 4; i++)
        {
            float a = adjacencyMode switch
            {
                WlBlockAdjacencyMode.RightToLeft => heightsA[i * 4 + 3],
                WlBlockAdjacencyMode.LeftToRight => heightsA[i * 4],
                WlBlockAdjacencyMode.BottomToTop => heightsA[12 + i],
                WlBlockAdjacencyMode.TopToBottom => heightsA[i],
                _ => heightsA[i]
            };

            float b = adjacencyMode switch
            {
                WlBlockAdjacencyMode.RightToLeft => heightsB[i * 4],
                WlBlockAdjacencyMode.LeftToRight => heightsB[i * 4 + 3],
                WlBlockAdjacencyMode.BottomToTop => heightsB[i],
                WlBlockAdjacencyMode.TopToBottom => heightsB[12 + i],
                _ => heightsB[i]
            };

            if (MathF.Abs(a - b) > heightTolerance)
                return false;
        }

        return true;
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

    private sealed class WlBodyBlock
    {
        public int BlockIndex { get; init; }
        public float CoordX { get; init; }
        public float CoordY { get; init; }
        public Vector3[] TransformedVertices { get; init; } = Array.Empty<Vector3>();
        public float[] Heights4x4 { get; init; } = Array.Empty<float>();
        public float MinX { get; init; }
        public float MaxX { get; init; }
        public float MinY { get; init; }
        public float MaxY { get; init; }
        public float MinHeight { get; init; }
        public float MaxHeight { get; init; }
        public float AverageHeight { get; init; }
        public int NonZeroMetadataWords { get; init; }
        public int MetadataPatternHash { get; init; }
    }

    private enum WlBlockAdjacencyMode
    {
        RightToLeft,
        LeftToRight,
        BottomToTop,
        TopToBottom,
    }
}

/// <summary>
/// A single liquid body loaded from a WLW/WLQ/WLM file.
/// Contains pre-transformed vertices ready for GPU upload.
/// </summary>
public class WlLiquidBody
{
    public string BodyKey { get; init; } = "";
    public string Name { get; init; } = "";
    public string SourcePath { get; init; } = "";
    public LiquidType Type { get; init; }
    public WlFileType FileType { get; init; }
    public WlLiquidLoader.WlBodyGroupingMode GroupingMode { get; init; }
    public string GroupLabel { get; init; } = "";
    public Vector3[] Vertices { get; init; } = Array.Empty<Vector3>();
    public int[] Indices { get; init; } = Array.Empty<int>();
    public Vector3 BoundsMin { get; init; }
    public Vector3 BoundsMax { get; init; }
    public int BlockCount { get; init; }
    public int[] SourceBlockIndices { get; init; } = Array.Empty<int>();
    public float MinHeight { get; init; }
    public float MaxHeight { get; init; }
    public float AverageHeight { get; init; }
    public float CoordMinX { get; init; }
    public float CoordMaxX { get; init; }
    public float CoordMinY { get; init; }
    public float CoordMaxY { get; init; }
    public int MetadataPatternCount { get; init; }
    public int MetadataNonZeroMin { get; init; }
    public int MetadataNonZeroMax { get; init; }
}
