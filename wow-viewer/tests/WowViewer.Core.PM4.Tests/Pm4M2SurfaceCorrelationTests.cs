using System.Numerics;
using WowViewer.Core.M2;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4M2SurfaceCorrelationTests
{
    private static M2ModelDocument CreateTestM2(
        string path,
        IReadOnlyList<Vector3> collisionVertices,
        IReadOnlyList<ushort> collisionTriangles,
        Vector3 min,
        Vector3 max)
    {
        M2ModelIdentity identity = M2ModelIdentity.FromPath(path);
        return new M2ModelDocument(
            identity: identity,
            rawBytes: [0],
            signature: "MD20",
            version: 264,
            flags: 0,
            viewCount: 1,
            modelName: path,
            globalLoops: [],
            sequences: [],
            sequenceLookup: [],
            colors: [],
            textureWeights: [],
            textureTransforms: [],
            lights: [],
            cameras: null,
            boundsMin: min,
            boundsMax: max,
            boundsRadius: 10f,
            embeddedSkinProfileCount: 0,
            embeddedSkinProfileOffset: 0,
            collisionBoundsMin: min,
            collisionBoundsMax: max,
            collisionRadius: 10f,
            collisionVertices: collisionVertices,
            collisionTriangles: collisionTriangles);
    }

    [Fact]
    public void ExtractFromM2Collision_FiltersOutSideAndBottomSurfaces_RetainsWalkableTopSurfaces()
    {
        // Construct a box model with:
        // Top surface (Z = +10, Normal Z = +1.0) -> 2 triangles
        // Bottom surface (Z = 0, Normal Z = -1.0) -> 2 triangles
        // 4 Side walls (Normal Z = 0.0) -> 8 triangles
        Vector3[] vertices =
        [
            // Bottom 4 vertices (0..3)
            new Vector3(0, 0, 0),
            new Vector3(10, 0, 0),
            new Vector3(10, 10, 0),
            new Vector3(0, 10, 0),
            // Top 4 vertices (4..7)
            new Vector3(0, 0, 10),
            new Vector3(10, 0, 10),
            new Vector3(10, 10, 10),
            new Vector3(0, 10, 10),
        ];

        ushort[] indices =
        [
            // Top quad (walkable, normal +Z): (4, 5, 6), (4, 6, 7)
            4, 5, 6,
            4, 6, 7,

            // Bottom quad (normal -Z): (0, 2, 1), (0, 3, 2)
            0, 2, 1,
            0, 3, 2,

            // Side wall 1 (+X): (1, 2, 6), (1, 6, 5)
            1, 2, 6,
            1, 6, 5,

            // Side wall 2 (-X): (3, 0, 4), (3, 4, 7)
            3, 0, 4,
            3, 4, 7,

            // Side wall 3 (+Y): (2, 3, 7), (2, 7, 6)
            2, 3, 7,
            2, 7, 6,

            // Side wall 4 (-Y): (0, 1, 5), (0, 5, 4)
            0, 1, 5,
            0, 5, 4,
        ];

        string m2Path = @"World\Generic\Human\PassiveDoodads\Crates\Crate01.m2";
        M2ModelDocument m2Doc = CreateTestM2(
            m2Path,
            vertices,
            indices,
            Vector3.Zero,
            new Vector3(10, 10, 10));

        Assert.True(m2Doc.HasCollisionMesh);
        Assert.Equal(12, m2Doc.CollisionTriangles.Count / 3);

        SurfaceCorrelationFingerprint? fp = Pm4SurfaceCorrelationExtractor.ExtractFromM2Model(
            m2Doc,
            m2Path,
            minNormalZ: 0.5f);

        Assert.NotNull(fp);
        Assert.Equal("m2", fp.AssetKind);
        Assert.Equal(0x40, fp.Ck24Type);
        // Only the 2 top triangles should be retained; all 8 side triangles and 2 bottom triangles are excluded
        Assert.Equal(2, fp.TriangleCount);
        Assert.Equal(4, fp.VertexCount);
        Assert.True(fp.TotalTriangleArea > 0f);
    }

    [Fact]
    public void MatchOne_M2TopSurfacesMatchIdenticalGeometryWithFullScore()
    {
        // Top surface of a crate (1 quad = 2 triangles)
        Vector3[] topVerts =
        [
            new Vector3(0, 0, 5),
            new Vector3(4, 0, 5),
            new Vector3(4, 4, 5),
            new Vector3(0, 4, 5),
        ];

        ushort[] topIndices =
        [
            0, 1, 2,
            0, 2, 3,
        ];

        string m2Path = @"World\Generic\Human\PassiveDoodads\Crates\CrateSmall.m2";
        M2ModelDocument crateDoc = CreateTestM2(
            m2Path,
            topVerts,
            topIndices,
            Vector3.Zero,
            new Vector3(4, 4, 5));

        SurfaceCorrelationFingerprint? m2Fp = Pm4SurfaceCorrelationExtractor.ExtractFromM2Model(
            crateDoc,
            m2Path,
            minNormalZ: 0.5f);

        Assert.NotNull(m2Fp);

        // Build a database containing this M2
        SurfaceCorrelationDatabase db = new(
            ArchiveRoot: "test-data",
            BuildDate: "2026-10-20",
            WmoCount: 0,
            EdgeBinSize: 1.0f,
            AreaBinSize: 1.0f,
            NormalAlignmentBinSize: 0.0f,
            PlanarOffsetBinSize: 0.0f,
            Records: [m2Fp])
        {
            M2Count = 1,
        };

        Assert.Equal(1, db.TotalAssetCount);

        // PM4 surfaces for the same small crate (single quad polygon with 4 indices)
        List<Pm4MsurEntry> pm4Surfaces =
        [
            new Pm4MsurEntry(
                GroupKey: 0,
                IndexCount: 4,
                AttributeMask: 0,
                Padding: 0,
                Normal: new Vector3(0, 0, 1),
                Height: -5f,
                MsviFirstIndex: 0,
                _0x18: 0,
                PackedParams: 0x40001000u) // CK24 type 0x40
        ];

        List<uint> msvi = [0, 1, 2, 3];

        SurfaceCorrelationFingerprint? pm4Fp = Pm4SurfaceCorrelationExtractor.ExtractFromPm4Group(
            topVerts,
            msvi,
            pm4Surfaces,
            ck24: 0x400010,
            ck24Type: 0x40,
            assetId: "pm4_crate",
            assetPath: "pm4_crate",
            assetKind: "m2");

        Assert.NotNull(pm4Fp);

        SurfaceMatchResult match = Pm4SurfaceCorrelationMatcher.MatchOne(pm4Fp, db);

        Assert.Equal("Matched", match.Status);
        Assert.NotNull(match.TopCandidate);
        Assert.Equal(m2Path, match.TopCandidate.AssetPath);
        Assert.InRange(match.TopScore, 0.99, 1.01);
    }

    [Fact]
    public void MatchOne_EnforcesAssetKindIsolation_M2DoesNotMatchWmo()
    {
        Vector3[] verts =
        [
            new Vector3(0, 0, 0),
            new Vector3(10, 0, 0),
            new Vector3(10, 10, 0),
            new Vector3(0, 10, 0),
        ];

        ushort[] indices = [0, 1, 2, 0, 2, 3];

        string m2Path = @"World\Generic\Human\PassiveDoodads\Crates\CrateSmall.m2";
        M2ModelDocument m2Doc = CreateTestM2(
            m2Path,
            verts,
            indices,
            Vector3.Zero,
            new Vector3(10, 10, 5));

        SurfaceCorrelationFingerprint? m2Fp = Pm4SurfaceCorrelationExtractor.ExtractFromM2Model(m2Doc, m2Path);
        Assert.NotNull(m2Fp);

        // Put only M2 in the database
        SurfaceCorrelationDatabase db = new(
            ArchiveRoot: "test",
            BuildDate: "2026-10-20",
            WmoCount: 0,
            EdgeBinSize: 1.0f,
            AreaBinSize: 1.0f,
            NormalAlignmentBinSize: 0.0f,
            PlanarOffsetBinSize: 0.0f,
            Records: [m2Fp])
        {
            M2Count = 1,
        };

        // Query with WMO CK24 type 0x42
        List<Pm4MsurEntry> wmoSurfaces =
        [
            new Pm4MsurEntry(0, 4, 0, 0, new Vector3(0, 0, 1), 0f, 0, 0, 0x42000100u)
        ];
        List<uint> msvi = [0, 1, 2, 3];

        SurfaceCorrelationFingerprint? wmoQuery = Pm4SurfaceCorrelationExtractor.ExtractFromPm4Group(
            verts,
            msvi,
            wmoSurfaces,
            ck24: 0x420001,
            ck24Type: 0x42,
            assetId: "wmo_query",
            assetPath: "wmo_query",
            assetKind: "wmo");

        Assert.NotNull(wmoQuery);

        SurfaceMatchResult result = Pm4SurfaceCorrelationMatcher.MatchOne(wmoQuery, db);

        // Since the DB has no WMO records, WMO query must return Ineligible and not match M2
        Assert.Equal("Ineligible", result.Status);
        Assert.Empty(result.Candidates);
    }

    [Fact]
    public void ResolveTile_Development5038_PartitionsM2SurfacesIntoDoodadsNotWmo()
    {
        string mapDir = @"I:\parp\parp-tools\wow-viewer\test_data\original_development\World\Maps\development";
        string pm4Path = Path.Combine(mapDir, "development_50_38.pm4");

        if (!File.Exists(pm4Path))
            return; // Skip if test data not present

        Pm4TileResolutionResult result = Pm4UnauthoredPlacementResolver.ResolveTile(
            pm4Path,
            mapDir);

        // development_50_38 has zero WMOs; its 4 M2 surfaces (CK24 type 0x41) must NOT be misclassified as WMOs!
        Assert.Empty(result.ResolvedWmos);
    }
}
