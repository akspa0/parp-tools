using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

/// <summary>
/// Contract tests for the strict object target. These exercise the raw MCVT
/// quincunx surface directly: the 257x257 output grid is a raster coordinate
/// system, not a substituted terrain-height image.
/// </summary>
public sealed class TerrainVisibleObjectMaskRasterizerTests
{
    [Fact]
    public void TrySampleTerrainSurface_InterpolatesNativeMcvtTriangleInsteadOfDenseFill()
    {
        TerrainVertexLattice terrain = BuildFlatTerrain(10f);
        terrain.VertexZ[0, 0, TerrainVertexLattice.ResolveSampleIndex(1, 1)] = 20f;
        TerrainVisibleObjectMaskRasterizer rasterizer = new(terrain);

        Assert.True(rasterizer.TrySampleTerrainSurface(0.5f, 0.5f, out float elevation));

        // (0.5, 0.5) lies in the native MCVT triangle made from (0,0),
        // (1,1), and (2,0). Its center vertex contributes one half.
        Assert.Equal(15f, elevation, precision: 3);
    }

    [Fact]
    public void PaintTriangle_KeepsOnlyFragmentsAboveRawTerrainAndRecordsProvenance()
    {
        TerrainVisibleObjectMaskRasterizer rasterizer = new(BuildFlatTerrain(10f));
        float[,] mask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] topElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] terrainElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        byte[,] source = new byte[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];

        ObjectTriangleRasterResult result = rasterizer.PaintTriangle(
            mask,
            topElevation,
            terrainElevation,
            source,
            new Vector2(0f, 0f),
            new Vector2(8f, 0f),
            new Vector2(0f, 8f),
            elevation0: 9f,
            elevation1: 11f,
            elevation2: 11f,
            ObjectGeometryPixelSource.M2Triangle);

        Assert.True(result.VisiblePixels > 0);
        Assert.True(result.OccludedPixels > 0);

        // This fragment is below the 10.0 MCVT surface plus the 0.25 clearance.
        Assert.Equal(0f, mask[0, 0]);

        // At pixel centre (4.5, 1.5), the transformed triangle is z=10.5,
        // so it is valid foreground geometry rather than a whole-placement mask.
        Assert.Equal(1f, mask[1, 4]);
        Assert.Equal(10.5f, topElevation[1, 4], precision: 3);
        Assert.Equal(10f, terrainElevation[1, 4], precision: 3);
        Assert.Equal((byte)ObjectGeometryPixelSource.M2Triangle, source[1, 4]);
    }

    [Fact]
    public void PaintTriangle_MissingRawMcvtVertexIsUnknownAndNeverBecomesVisible()
    {
        TerrainVertexLattice terrain = BuildFlatTerrain(10f);
        terrain.Present[0, 0, TerrainVertexLattice.ResolveSampleIndex(1, 1)] = false;
        TerrainVisibleObjectMaskRasterizer rasterizer = new(terrain);
        float[,] mask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] topElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] terrainElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        byte[,] source = new byte[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];

        ObjectTriangleRasterResult result = rasterizer.PaintTriangle(
            mask,
            topElevation,
            terrainElevation,
            source,
            new Vector2(0.25f, 0.25f),
            new Vector2(1.25f, 0.25f),
            new Vector2(0.25f, 1.25f),
            elevation0: 50f,
            elevation1: 50f,
            elevation2: 50f,
            ObjectGeometryPixelSource.WmoTriangle);

        Assert.True(result.TerrainUnknownPixels > 0);
        Assert.Equal(0, result.VisiblePixels);
        Assert.Equal(0f, mask[0, 0]);
        Assert.Equal((byte)ObjectGeometryPixelSource.None, source[0, 0]);
    }

    [Fact]
    public void PaintTriangle_LiquidHidesOnlyTheSubmergedFragment()
    {
        float[,] liquidMask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] liquidHeight = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        liquidMask[1, 4] = 1f;
        liquidHeight[1, 4] = 20f;
        TerrainVisibleObjectMaskRasterizer rasterizer = new(
            BuildFlatTerrain(10f),
            liquidMask: liquidMask,
            liquidHeight: liquidHeight);
        float[,] mask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] topElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] terrainElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        byte[,] source = new byte[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];

        ObjectTriangleRasterResult result = rasterizer.PaintTriangle(
            mask,
            topElevation,
            terrainElevation,
            source,
            new Vector2(0f, 0f),
            new Vector2(8f, 0f),
            new Vector2(0f, 8f),
            elevation0: 15f,
            elevation1: 15f,
            elevation2: 15f,
            ObjectGeometryPixelSource.M2Triangle);

        Assert.True(result.LiquidCoveredPixels > 0);
        Assert.True(result.LiquidHiddenPixels > 0);
        Assert.True(result.VisiblePixels > 0);
        Assert.Equal(0f, mask[1, 4]);
        Assert.Equal(1f, mask[0, 1]);
    }

    [Fact]
    public void PaintTriangle_RetainsObjectFragmentAboveKnownLiquidSurface()
    {
        float[,] liquidMask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] liquidHeight = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        liquidMask[1, 4] = 1f;
        liquidHeight[1, 4] = 12f;
        TerrainVisibleObjectMaskRasterizer rasterizer = new(
            BuildFlatTerrain(10f),
            liquidMask: liquidMask,
            liquidHeight: liquidHeight);
        float[,] mask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] topElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] terrainElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        byte[,] source = new byte[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];

        ObjectTriangleRasterResult result = rasterizer.PaintTriangle(
            mask,
            topElevation,
            terrainElevation,
            source,
            new Vector2(0f, 0f),
            new Vector2(8f, 0f),
            new Vector2(0f, 8f),
            elevation0: 15f,
            elevation1: 15f,
            elevation2: 15f,
            ObjectGeometryPixelSource.WmoTriangle);

        Assert.True(result.LiquidCoveredPixels > 0);
        Assert.True(result.LiquidAboveSurfacePixels > 0);
        Assert.Equal(0, result.LiquidHiddenPixels);
        Assert.Equal(1f, mask[1, 4]);
        Assert.Equal((byte)ObjectGeometryPixelSource.WmoTriangle, source[1, 4]);
    }

    [Fact]
    public void PaintTriangleWithTrace_PreservesUncollapsedWorldMeshAndPlacementEvidence()
    {
        TerrainVisibleObjectMaskRasterizer rasterizer = new(BuildFlatTerrain(10f));
        float[,] mask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] topElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] terrainElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        byte[,] source = new byte[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        List<ObjectGeometryFragmentRecord> trace = [];

        rasterizer.PaintTriangleWithTrace(
            mask,
            topElevation,
            terrainElevation,
            source,
            new Vector2(0f, 0f),
            new Vector2(8f, 0f),
            new Vector2(0f, 8f),
            new Vector3(100f, 200f, 15f),
            new Vector3(108f, 200f, 15f),
            new Vector3(100f, 208f, 15f),
            elevation0: 15f,
            elevation1: 15f,
            elevation2: 15f,
            ObjectGeometryPixelSource.M2Triangle,
            placementUniqueId: 771,
            assetIndex: 4,
            sourceTriangleIndex: 19,
            trace);

        ObjectGeometryFragmentRecord fragment = Assert.Single(
            trace,
            static item => item.RasterX == 1 && item.RasterY == 1);
        Assert.Equal(ObjectGeometryFragmentClassification.TerrainVisible, fragment.Classification);
        Assert.Equal(ObjectGeometryPixelSource.M2Triangle, fragment.Source);
        Assert.Equal(771, fragment.PlacementUniqueId);
        Assert.Equal(4, fragment.AssetIndex);
        Assert.Equal(19, fragment.SourceTriangleIndex);
        Assert.Equal(101.5f, fragment.ObjectWorldX, precision: 3);
        Assert.Equal(201.5f, fragment.ObjectWorldY, precision: 3);
        Assert.Equal(15f, fragment.ObjectWorldZ, precision: 3);
        Assert.Equal(15f, fragment.ObjectElevation, precision: 3);
        Assert.True(fragment.TerrainVertex0Present);
        Assert.True(fragment.TerrainVertex1Present);
        Assert.True(fragment.TerrainVertex2Present);
        Assert.Equal(10f, fragment.TerrainElevation, precision: 3);
        Assert.True(float.IsNaN(fragment.LiquidSurfaceElevation));

        // The final union retains one positive pixel, while the trace keeps
        // both source fragments that overlap it.
        rasterizer.PaintTriangleWithTrace(
            mask,
            topElevation,
            terrainElevation,
            source,
            new Vector2(0f, 0f),
            new Vector2(8f, 0f),
            new Vector2(0f, 8f),
            new Vector3(100f, 200f, 20f),
            new Vector3(108f, 200f, 20f),
            new Vector3(100f, 208f, 20f),
            elevation0: 20f,
            elevation1: 20f,
            elevation2: 20f,
            ObjectGeometryPixelSource.WmoTriangle,
            placementUniqueId: 772,
            assetIndex: 5,
            sourceTriangleIndex: 23,
            trace);

        Assert.Equal(2, trace.Count(static item => item.RasterX == 1 && item.RasterY == 1));
        Assert.Equal(1f, mask[1, 1]);
        Assert.Equal((byte)ObjectGeometryPixelSource.WmoTriangle, source[1, 1]);
    }

    [Fact]
    public void PaintTriangleWithTrace_RecordsWaterAndUnknownTerrainAsRejectedFragments()
    {
        float[,] liquidMask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] liquidHeight = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        liquidMask[1, 1] = 1f;
        liquidHeight[1, 1] = 20f;
        TerrainVertexLattice terrain = BuildFlatTerrain(10f);
        terrain.Present[0, 0, TerrainVertexLattice.ResolveSampleIndex(1, 1)] = false;
        TerrainVisibleObjectMaskRasterizer rasterizer = new(
            terrain,
            liquidMask: liquidMask,
            liquidHeight: liquidHeight);
        float[,] mask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] topElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] terrainElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        byte[,] source = new byte[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        List<ObjectGeometryFragmentRecord> trace = [];

        rasterizer.PaintTriangleWithTrace(
            mask,
            topElevation,
            terrainElevation,
            source,
            new Vector2(0f, 0f),
            new Vector2(8f, 0f),
            new Vector2(0f, 8f),
            new Vector3(0f, 0f, 15f),
            new Vector3(8f, 0f, 15f),
            new Vector3(0f, 8f, 15f),
            elevation0: 15f,
            elevation1: 15f,
            elevation2: 15f,
            ObjectGeometryPixelSource.WmoTriangle,
            placementUniqueId: 88,
            assetIndex: 7,
            sourceTriangleIndex: 3,
            trace);

        ObjectGeometryFragmentRecord fragment = Assert.Single(
            trace,
            static item => item.RasterX == 1 && item.RasterY == 1);
        Assert.Equal(ObjectGeometryFragmentClassification.TerrainUnknown, fragment.Classification);
        Assert.False(fragment.TerrainVertex0Present && fragment.TerrainVertex1Present && fragment.TerrainVertex2Present);
        Assert.Equal(20f, fragment.LiquidSurfaceElevation, precision: 3);
        Assert.Equal(0f, mask[1, 1]);
    }

    [Fact]
    public void PaintTriangle_ClipsGeometryAtTileEdgeWithoutCentroidOrBoundsErasure()
    {
        TerrainVisibleObjectMaskRasterizer rasterizer = new(BuildFlatTerrain(10f));
        float[,] mask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] topElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] terrainElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        byte[,] source = new byte[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];

        ObjectTriangleRasterResult result = rasterizer.PaintTriangle(
            mask,
            topElevation,
            terrainElevation,
            source,
            new Vector2(-4f, 0f),
            new Vector2(4f, 0f),
            new Vector2(-4f, 8f),
            elevation0: 20f,
            elevation1: 20f,
            elevation2: 20f,
            ObjectGeometryPixelSource.WmoTriangle);

        Assert.True(result.VisiblePixels > 0);
        Assert.Equal(1f, mask[0, 0]);
        Assert.Equal((byte)ObjectGeometryPixelSource.WmoTriangle, source[0, 0]);
    }

    [Fact]
    public void PaintTriangleWithTrace_PaintsInstanceIdUnderTheFrontMostRule()
    {
        TerrainVisibleObjectMaskRasterizer rasterizer = new(BuildFlatTerrain(10f));
        float[,] mask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] topElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] terrainElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        byte[,] source = new byte[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        int[,] instance = new int[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];

        // Instance 1: the large low triangle (elevation 12).
        rasterizer.PaintTriangleWithTrace(
            mask, topElevation, terrainElevation, source,
            new Vector2(0f, 0f), new Vector2(8f, 0f), new Vector2(0f, 8f),
            new Vector3(0f, 0f, 12f), new Vector3(8f, 0f, 12f), new Vector3(0f, 8f, 12f),
            12f, 12f, 12f,
            ObjectGeometryPixelSource.M2Triangle,
            placementUniqueId: 101,
            assetIndex: 0,
            sourceTriangleIndex: 0,
            fragmentTrace: null,
            visibleInstance: instance,
            instanceId: 1);

        // Instance 2: the smaller overlapping but HIGHER triangle (elevation 14).
        rasterizer.PaintTriangleWithTrace(
            mask, topElevation, terrainElevation, source,
            new Vector2(0f, 0f), new Vector2(4f, 0f), new Vector2(0f, 4f),
            new Vector3(0f, 0f, 14f), new Vector3(4f, 0f, 14f), new Vector3(0f, 4f, 14f),
            14f, 14f, 14f,
            ObjectGeometryPixelSource.WmoTriangle,
            placementUniqueId: 202,
            assetIndex: 1,
            sourceTriangleIndex: 0,
            fragmentTrace: null,
            visibleInstance: instance,
            instanceId: 2);

        // Overlap pixel: the higher (front-most) fragment owns the identity.
        Assert.Equal(1f, mask[0, 0]);
        Assert.Equal(2, instance[0, 0]);
        Assert.Equal((byte)ObjectGeometryPixelSource.WmoTriangle, source[0, 0]);

        // Pixel covered only by instance 1 keeps instance 1's identity.
        Assert.Equal(1f, mask[0, 5]);
        Assert.Equal(1, instance[0, 5]);
        Assert.Equal((byte)ObjectGeometryPixelSource.M2Triangle, source[0, 5]);
    }

    [Fact]
    public void PaintTriangleWithTrace_OccludedTrianglePaintsNoInstanceId()
    {
        TerrainVisibleObjectMaskRasterizer rasterizer = new(BuildFlatTerrain(10f));
        float[,] mask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] topElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] terrainElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        byte[,] source = new byte[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        int[,] instance = new int[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];

        // Elevation 9 is below the 10.0 surface plus the 0.25 clearance everywhere.
        ObjectTriangleRasterResult result = rasterizer.PaintTriangleWithTrace(
            mask, topElevation, terrainElevation, source,
            new Vector2(0f, 0f), new Vector2(8f, 0f), new Vector2(0f, 8f),
            new Vector3(0f, 0f, 9f), new Vector3(8f, 0f, 9f), new Vector3(0f, 8f, 9f),
            9f, 9f, 9f,
            ObjectGeometryPixelSource.M2Triangle,
            placementUniqueId: 303,
            assetIndex: 0,
            sourceTriangleIndex: 0,
            fragmentTrace: null,
            visibleInstance: instance,
            instanceId: 5);

        Assert.Equal(0, result.VisiblePixels);
        Assert.True(result.OccludedPixels > 0);
        Assert.Equal(0, instance[0, 0]);
        Assert.Equal(0, instance[4, 4]);
    }

    [Fact]
    public void PaintTriangleWithTrace_InstanceIdIsPositiveExactlyWhereTheMaskIsPositive()
    {
        TerrainVisibleObjectMaskRasterizer rasterizer = new(BuildFlatTerrain(10f));
        float[,] mask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] topElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] terrainElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        byte[,] source = new byte[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        int[,] instance = new int[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];

        rasterizer.PaintTriangleWithTrace(
            mask, topElevation, terrainElevation, source,
            new Vector2(0f, 0f), new Vector2(16f, 0f), new Vector2(0f, 16f),
            new Vector3(0f, 0f, 12f), new Vector3(16f, 0f, 12f), new Vector3(0f, 16f, 12f),
            12f, 12f, 12f,
            ObjectGeometryPixelSource.M2Triangle,
            placementUniqueId: 404,
            assetIndex: 0,
            sourceTriangleIndex: 0,
            fragmentTrace: null,
            visibleInstance: instance,
            instanceId: 3);

        for (int y = 0; y < 24; y++)
        {
            for (int x = 0; x < 24; x++)
            {
                Assert.Equal(mask[y, x] > 0f, instance[y, x] > 0);
            }
        }
    }

    [Fact]
    public void PaintTriangleWithTrace_InstancePaintRequiresAPositiveCompactId()
    {
        TerrainVisibleObjectMaskRasterizer rasterizer = new(BuildFlatTerrain(10f));
        float[,] mask = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] topElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        float[,] terrainElevation = new float[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        byte[,] source = new byte[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];
        int[,] instance = new int[TerrainVisibleObjectMaskRasterizer.Size, TerrainVisibleObjectMaskRasterizer.Size];

        Assert.Throws<ArgumentOutOfRangeException>(() => rasterizer.PaintTriangleWithTrace(
            mask, topElevation, terrainElevation, source,
            new Vector2(0f, 0f), new Vector2(8f, 0f), new Vector2(0f, 8f),
            new Vector3(0f, 0f, 12f), new Vector3(8f, 0f, 12f), new Vector3(0f, 8f, 12f),
            12f, 12f, 12f,
            ObjectGeometryPixelSource.M2Triangle,
            placementUniqueId: 505,
            assetIndex: 0,
            sourceTriangleIndex: 0,
            fragmentTrace: null,
            visibleInstance: instance,
            instanceId: 0));
    }

    private static TerrainVertexLattice BuildFlatTerrain(float elevation)
    {
        float[,,] z = new float[TerrainVertexLattice.ChunksPerAxis, TerrainVertexLattice.ChunksPerAxis, TerrainVertexLattice.SamplesPerChunk];
        float[,,] worldX = new float[TerrainVertexLattice.ChunksPerAxis, TerrainVertexLattice.ChunksPerAxis, TerrainVertexLattice.SamplesPerChunk];
        float[,,] worldY = new float[TerrainVertexLattice.ChunksPerAxis, TerrainVertexLattice.ChunksPerAxis, TerrainVertexLattice.SamplesPerChunk];
        bool[,,] present = new bool[TerrainVertexLattice.ChunksPerAxis, TerrainVertexLattice.ChunksPerAxis, TerrainVertexLattice.SamplesPerChunk];
        bool[,] dense = new bool[TerrainVertexLattice.DenseGridSize, TerrainVertexLattice.DenseGridSize];

        for (int chunkY = 0; chunkY < TerrainVertexLattice.ChunksPerAxis; chunkY++)
        {
            for (int chunkX = 0; chunkX < TerrainVertexLattice.ChunksPerAxis; chunkX++)
            {
                for (int sample = 0; sample < TerrainVertexLattice.SamplesPerChunk; sample++)
                {
                    TerrainVertexLattice.ResolveDenseCoordinates(chunkX, chunkY, sample, out int x, out int y);
                    z[chunkY, chunkX, sample] = elevation;
                    worldX[chunkY, chunkX, sample] = x;
                    worldY[chunkY, chunkX, sample] = y;
                    present[chunkY, chunkX, sample] = true;
                    dense[y, x] = true;
                }
            }
        }

        return new TerrainVertexLattice(z, worldX, worldY, present, dense);
    }
}
