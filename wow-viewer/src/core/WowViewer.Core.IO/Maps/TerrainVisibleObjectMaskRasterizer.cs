using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>Result of rasterizing one transformed geometry triangle.</summary>
public readonly record struct ObjectTriangleRasterResult(
    int CandidatePixels,
    int VisiblePixels,
    int OccludedPixels,
    int TerrainUnknownPixels,
    int LiquidCoveredPixels,
    int LiquidHiddenPixels,
    int LiquidAboveSurfacePixels,
    int LiquidUnknownPixels);

/// <summary>
/// Exact raw-MCVT interpolation evidence for one dense raster sample. Dense
/// coordinates identify canonical raw vertices; absent source vertices retain
/// their identity and are represented with NaN Z rather than fabricated data.
/// </summary>
public readonly record struct TerrainSurfaceFragmentSample(
    float Elevation,
    int Vertex0X,
    int Vertex0Y,
    int Vertex1X,
    int Vertex1Y,
    int Vertex2X,
    int Vertex2Y,
    float Vertex0Z,
    float Vertex1Z,
    float Vertex2Z,
    bool Vertex0Present,
    bool Vertex1Present,
    bool Vertex2Present,
    float Weight0,
    float Weight1,
    float Weight2)
{
    public static TerrainSurfaceFragmentSample Unknown => new(
        float.NaN,
        -1, -1,
        -1, -1,
        -1, -1,
        float.NaN, float.NaN, float.NaN,
        false, false, false,
        float.NaN, float.NaN, float.NaN);
}

/// <summary>
/// Rasterizes transformed object triangles only where their interpolated world
/// elevation is above the exact raw MCVT terrain surface at that raster sample.
/// When raw liquid evidence is supplied, a fragment at or below the liquid
/// surface is excluded while a fragment above it remains explicit provenance.
/// It intentionally has no rectangle, circle, chunk-coverage, or placement
/// fallback route.
/// </summary>
public sealed class TerrainVisibleObjectMaskRasterizer
{
    public const int Size = TerrainVertexLattice.DenseGridSize;
    public const float DefaultTerrainClearance = 0.25f;

    private readonly TerrainVertexLattice _terrain;
    private readonly float _clearance;
    private readonly float[,]? _liquidMask;
    private readonly float[,]? _liquidHeight;

    public TerrainVisibleObjectMaskRasterizer(
        TerrainVertexLattice terrain,
        float clearance = DefaultTerrainClearance,
        float[,]? liquidMask = null,
        float[,]? liquidHeight = null)
    {
        _terrain = terrain ?? throw new ArgumentNullException(nameof(terrain));
        if (!float.IsFinite(clearance) || clearance < 0f)
            throw new ArgumentOutOfRangeException(nameof(clearance));
        if ((liquidMask is null) != (liquidHeight is null))
            throw new ArgumentException("Liquid mask and liquid height must be provided together.");
        if (liquidMask is not null
            && (liquidMask.GetLength(0) != Size || liquidMask.GetLength(1) != Size
                || liquidHeight!.GetLength(0) != Size || liquidHeight.GetLength(1) != Size))
        {
            throw new ArgumentException($"Liquid evidence must be {Size}x{Size}.");
        }
        _clearance = clearance;
        _liquidMask = liquidMask;
        _liquidHeight = liquidHeight;
    }

    /// <summary>
    /// Paint a triangle fragment-by-fragment.  A pixel is retained only when
    /// the triangle elevation at its center exceeds terrain elevation by the
    /// configured clearance. A known liquid surface independently hides only
    /// submerged fragments. Missing terrain or liquid evidence is counted and
    /// never treated as visible terrain.
    /// </summary>
    public ObjectTriangleRasterResult PaintTriangle(
        float[,] visibleMask,
        float[,] visibleTopElevation,
        float[,] visibleTerrainElevation,
        byte[,] visibleSource,
        Vector2 p0,
        Vector2 p1,
        Vector2 p2,
        float elevation0,
        float elevation1,
        float elevation2,
        ObjectGeometryPixelSource source)
        => PaintTriangleWithTrace(
            visibleMask,
            visibleTopElevation,
            visibleTerrainElevation,
            visibleSource,
            p0,
            p1,
            p2,
            new Vector3(p0.X, p0.Y, elevation0),
            new Vector3(p1.X, p1.Y, elevation1),
            new Vector3(p2.X, p2.Y, elevation2),
            elevation0,
            elevation1,
            elevation2,
            source,
            placementUniqueId: -1,
            assetIndex: -1,
            sourceTriangleIndex: -1,
            fragmentTrace: null);

    /// <summary>
    /// Strict rasterization overload that retains one trace row for every
    /// triangle/raster candidate. The trace is deliberately independent of
    /// the final mask union so overlaps and rejected fragments remain auditable.
    /// </summary>
    public ObjectTriangleRasterResult PaintTriangleWithTrace(
        float[,] visibleMask,
        float[,] visibleTopElevation,
        float[,] visibleTerrainElevation,
        byte[,] visibleSource,
        Vector2 p0,
        Vector2 p1,
        Vector2 p2,
        Vector3 world0,
        Vector3 world1,
        Vector3 world2,
        float elevation0,
        float elevation1,
        float elevation2,
        ObjectGeometryPixelSource source,
        int placementUniqueId,
        int assetIndex,
        int sourceTriangleIndex,
        ICollection<ObjectGeometryFragmentRecord>? fragmentTrace)
    {
        ValidateOutputShape(visibleMask, nameof(visibleMask));
        ValidateOutputShape(visibleTopElevation, nameof(visibleTopElevation));
        ValidateOutputShape(visibleTerrainElevation, nameof(visibleTerrainElevation));
        ValidateOutputShape(visibleSource, nameof(visibleSource));
        if (!float.IsFinite(elevation0) || !float.IsFinite(elevation1) || !float.IsFinite(elevation2))
            throw new ArgumentOutOfRangeException(nameof(elevation0), "Triangle elevations must be finite.");
        if (!IsFinite(world0) || !IsFinite(world1) || !IsFinite(world2))
            throw new ArgumentOutOfRangeException(nameof(world0), "Transformed object coordinates must be finite.");
        if (source == ObjectGeometryPixelSource.None)
            throw new ArgumentOutOfRangeException(nameof(source));

        float area = Edge(p0, p1, p2);
        if (MathF.Abs(area) < 0.0001f)
            return default;

        int minX = Math.Max(0, (int)MathF.Floor(MathF.Min(p0.X, MathF.Min(p1.X, p2.X))));
        int minY = Math.Max(0, (int)MathF.Floor(MathF.Min(p0.Y, MathF.Min(p1.Y, p2.Y))));
        int maxX = Math.Min(Size - 1, (int)MathF.Ceiling(MathF.Max(p0.X, MathF.Max(p1.X, p2.X))));
        int maxY = Math.Min(Size - 1, (int)MathF.Ceiling(MathF.Max(p0.Y, MathF.Max(p1.Y, p2.Y))));

        int candidates = 0;
        int visible = 0;
        int occluded = 0;
        int terrainUnknown = 0;
        int liquidCovered = 0;
        int liquidHidden = 0;
        int liquidAboveSurface = 0;
        int liquidUnknown = 0;
        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                Vector2 sample = new(x + 0.5f, y + 0.5f);
                float w0 = Edge(p1, p2, sample) / area;
                float w1 = Edge(p2, p0, sample) / area;
                float w2 = Edge(p0, p1, sample) / area;
                const float insideEpsilon = -0.00001f;
                if (w0 < insideEpsilon || w1 < insideEpsilon || w2 < insideEpsilon)
                    continue;

                candidates++;
                float objectElevation = (w0 * elevation0) + (w1 * elevation1) + (w2 * elevation2);
                Vector3 objectWorld = (w0 * world0) + (w1 * world1) + (w2 * world2);
                bool terrainKnown = TrySampleTerrainSurface(sample.X, sample.Y, out TerrainSurfaceFragmentSample terrainSample);
                if (!TrySampleLiquidSurface(y, x, out bool liquidPresent, out float liquidElevation))
                {
                    liquidUnknown++;
                    AppendTrace(
                        ObjectGeometryFragmentClassification.LiquidUnknown,
                        terrainSample,
                        float.NaN);
                    continue;
                }

                if (!terrainKnown)
                {
                    terrainUnknown++;
                    AppendTrace(
                        ObjectGeometryFragmentClassification.TerrainUnknown,
                        terrainSample,
                        liquidPresent ? liquidElevation : float.NaN);
                    continue;
                }

                if (liquidPresent)
                {
                    liquidCovered++;
                    if (objectElevation <= liquidElevation + _clearance)
                    {
                        liquidHidden++;
                        AppendTrace(
                            ObjectGeometryFragmentClassification.WaterHidden,
                            terrainSample,
                            liquidElevation);
                        continue;
                    }
                    liquidAboveSurface++;
                }

                if (objectElevation <= terrainSample.Elevation + _clearance)
                {
                    occluded++;
                    AppendTrace(
                        ObjectGeometryFragmentClassification.TerrainHidden,
                        terrainSample,
                        liquidPresent ? liquidElevation : float.NaN);
                    continue;
                }

                if (visibleMask[y, x] <= 0f || objectElevation > visibleTopElevation[y, x])
                {
                    visibleTopElevation[y, x] = objectElevation;
                    visibleTerrainElevation[y, x] = terrainSample.Elevation;
                    visibleSource[y, x] = (byte)source;
                }
                visibleMask[y, x] = 1f;
                visible++;
                AppendTrace(
                    ObjectGeometryFragmentClassification.TerrainVisible,
                    terrainSample,
                    liquidPresent ? liquidElevation : float.NaN);

                void AppendTrace(
                    ObjectGeometryFragmentClassification classification,
                    TerrainSurfaceFragmentSample terrain,
                    float liquidSurfaceElevation)
                {
                    if (fragmentTrace is null)
                        return;

                    fragmentTrace.Add(new ObjectGeometryFragmentRecord(
                        x,
                        y,
                        objectWorld.X,
                        objectWorld.Y,
                        objectWorld.Z,
                        objectElevation,
                        placementUniqueId,
                        assetIndex,
                        sourceTriangleIndex,
                        source,
                        classification,
                        terrain.Vertex0X,
                        terrain.Vertex0Y,
                        terrain.Vertex1X,
                        terrain.Vertex1Y,
                        terrain.Vertex2X,
                        terrain.Vertex2Y,
                        terrain.Vertex0Z,
                        terrain.Vertex1Z,
                        terrain.Vertex2Z,
                        terrain.Vertex0Present,
                        terrain.Vertex1Present,
                        terrain.Vertex2Present,
                        terrain.Weight0,
                        terrain.Weight1,
                        terrain.Weight2,
                        terrain.Elevation,
                        liquidSurfaceElevation));
                }
            }
        }

        return new ObjectTriangleRasterResult(
            candidates,
            visible,
            occluded,
            terrainUnknown,
            liquidCovered,
            liquidHidden,
            liquidAboveSurface,
            liquidUnknown);
    }

    private static bool IsFinite(Vector3 value)
        => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);

    /// <summary>
    /// Samples the canonical quincunx MCVT surface at a dense-grid position.
    /// The interpolation uses the actual four triangles around each inner MCVT
    /// node rather than a filled 257x257 display raster.
    /// </summary>
    public bool TrySampleTerrainSurface(float denseX, float denseY, out float elevation)
    {
        bool known = TrySampleTerrainSurface(denseX, denseY, out TerrainSurfaceFragmentSample sample);
        elevation = sample.Elevation;
        return known;
    }

    /// <summary>
    /// Samples the native MCVT surface and returns its three-node interpolation
    /// evidence even when one of those raw source vertices is unavailable.
    /// </summary>
    public bool TrySampleTerrainSurface(
        float denseX,
        float denseY,
        out TerrainSurfaceFragmentSample terrainSample)
    {
        terrainSample = TerrainSurfaceFragmentSample.Unknown;
        if (!float.IsFinite(denseX) || !float.IsFinite(denseY))
            return false;

        denseX = Math.Clamp(denseX, 0f, Size - 1f);
        denseY = Math.Clamp(denseY, 0f, Size - 1f);
        int chunkX = Math.Min((int)(denseX / TerrainVertexLattice.HalfStepsPerChunk), TerrainVertexLattice.ChunksPerAxis - 1);
        int chunkY = Math.Min((int)(denseY / TerrainVertexLattice.HalfStepsPerChunk), TerrainVertexLattice.ChunksPerAxis - 1);
        float localX = denseX - (chunkX * TerrainVertexLattice.HalfStepsPerChunk);
        float localY = denseY - (chunkY * TerrainVertexLattice.HalfStepsPerChunk);
        int cellX = Math.Min((int)(localX / 2f), 7);
        int cellY = Math.Min((int)(localY / 2f), 7);
        int originX = (chunkX * TerrainVertexLattice.HalfStepsPerChunk) + (cellX * 2);
        int originY = (chunkY * TerrainVertexLattice.HalfStepsPerChunk) + (cellY * 2);
        Vector2 sample = new(denseX, denseY);

        if (TryDescribeTerrainTriangle(
                new(originX + 1, originY + 1),
                new(originX + 2, originY),
                new(originX, originY),
                sample,
                out terrainSample,
                out bool complete))
        {
            return complete;
        }

        if (TryDescribeTerrainTriangle(
                new(originX + 1, originY + 1),
                new(originX + 2, originY + 2),
                new(originX + 2, originY),
                sample,
                out terrainSample,
                out complete))
        {
            return complete;
        }

        if (TryDescribeTerrainTriangle(
                new(originX + 1, originY + 1),
                new(originX, originY + 2),
                new(originX + 2, originY + 2),
                sample,
                out terrainSample,
                out complete))
        {
            return complete;
        }

        if (TryDescribeTerrainTriangle(
                new(originX + 1, originY + 1),
                new(originX, originY),
                new(originX, originY + 2),
                sample,
                out terrainSample,
                out complete))
        {
            return complete;
        }

        terrainSample = TerrainSurfaceFragmentSample.Unknown;
        return false;
    }

    private bool TryDescribeTerrainTriangle(
        Vector2 p0,
        Vector2 p1,
        Vector2 p2,
        Vector2 sample,
        out TerrainSurfaceFragmentSample terrainSample,
        out bool complete)
    {
        terrainSample = TerrainSurfaceFragmentSample.Unknown;
        complete = false;
        float area = Edge(p0, p1, p2);
        if (MathF.Abs(area) < 0.0001f)
            return false;
        float w0 = Edge(p1, p2, sample) / area;
        float w1 = Edge(p2, p0, sample) / area;
        float w2 = Edge(p0, p1, sample) / area;
        const float insideEpsilon = -0.00001f;
        if (w0 < insideEpsilon || w1 < insideEpsilon || w2 < insideEpsilon)
            return false;

        bool present0 = _terrain.TryGetVertexAtDenseCoordinates((int)p0.X, (int)p0.Y, out float z0);
        bool present1 = _terrain.TryGetVertexAtDenseCoordinates((int)p1.X, (int)p1.Y, out float z1);
        bool present2 = _terrain.TryGetVertexAtDenseCoordinates((int)p2.X, (int)p2.Y, out float z2);
        if (!present0)
            z0 = float.NaN;
        if (!present1)
            z1 = float.NaN;
        if (!present2)
            z2 = float.NaN;

        float elevation = present0 && present1 && present2
            ? (w0 * z0) + (w1 * z1) + (w2 * z2)
            : float.NaN;
        terrainSample = new TerrainSurfaceFragmentSample(
            elevation,
            (int)p0.X,
            (int)p0.Y,
            (int)p1.X,
            (int)p1.Y,
            (int)p2.X,
            (int)p2.Y,
            z0,
            z1,
            z2,
            present0,
            present1,
            present2,
            w0,
            w1,
            w2);
        complete = float.IsFinite(elevation);
        return true;
    }

    private bool TrySampleLiquidSurface(int y, int x, out bool liquidPresent, out float elevation)
    {
        liquidPresent = false;
        elevation = 0f;
        if (_liquidMask is null)
            return true;

        float mask = _liquidMask[y, x];
        if (!float.IsFinite(mask))
            return false;
        if (mask <= 0f)
            return true;

        liquidPresent = true;
        elevation = _liquidHeight![y, x];
        return float.IsFinite(elevation);
    }

    private static float Edge(Vector2 a, Vector2 b, Vector2 point)
        => ((point.X - a.X) * (b.Y - a.Y)) - ((point.Y - a.Y) * (b.X - a.X));

    private static void ValidateOutputShape(Array array, string name)
    {
        ArgumentNullException.ThrowIfNull(array);
        if (array.Rank != 2 || array.GetLength(0) != Size || array.GetLength(1) != Size)
            throw new ArgumentException($"Strict object-mask buffers must be {Size}x{Size}.", name);
    }
}
