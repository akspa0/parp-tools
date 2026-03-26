using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4ResearchIntegrationTests
{
    [Fact]
    public void ReadFile_DevelopmentTile_ReturnsExpectedCounts()
    {
        Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(Pm4TestPaths.DevelopmentTilePath);

        Assert.Equal((uint)12304, document.Version);
        Assert.Equal(54, document.Chunks.Count);
        Assert.Equal(6318, document.KnownChunks.Msvt.Count);
        Assert.Equal(9990, document.KnownChunks.Mscn.Count);
        Assert.Equal(2493, document.KnownChunks.Mprl.Count);
        Assert.Equal(4110, document.KnownChunks.Msur.Count);
        Assert.Empty(document.Diagnostics);
    }

    [Fact]
    public void Analyze_DevelopmentTile_PreservesCurrentSummaryAndUniqueIdHypothesis()
    {
        Pm4AnalysisReport report = Pm4ResearchAnalyzer.Analyze(Pm4ResearchReader.ReadFile(Pm4TestPaths.DevelopmentTilePath));

        Assert.Empty(report.UnknownChunks);
        Assert.NotEmpty(report.TopCk24Groups);
        Assert.Equal((uint)4237834, report.TopCk24Groups[0].Ck24);
        Assert.Equal((ushort)43530, report.TopCk24Groups[0].Ck24ObjectId);
        Assert.Equal(896, report.TopCk24Groups[0].SurfaceCount);
        Assert.Contains(report.ResearchNotes, note => note.Contains("UniqueID", StringComparison.Ordinal));
    }

    [Fact]
    public void Audit_DevelopmentTile_ReportsCurrentReferenceStatus()
    {
        Pm4DecodeAuditReport report = Pm4ResearchAuditAnalyzer.Analyze(Pm4ResearchReader.ReadFile(Pm4TestPaths.DevelopmentTilePath));

        Assert.Equal(54, report.ChunkCount);
        Assert.Equal(54, report.RecognizedChunkCount);
        Assert.Equal(0, report.UnknownChunkCount);
        Assert.False(report.HasTrailingBytesDiagnostic);
        Assert.False(report.HasOverrunDiagnostic);

        Pm4ReferenceAudit mslkToMsur = Assert.Single(report.ReferenceAudits, static audit => audit.Name == "MSLK.RefIndex->MSUR");
        Pm4ReferenceAudit mdosToMdbh = Assert.Single(report.ReferenceAudits, static audit => audit.Name == "MDOS.buildingIndex->MDBH");

        Assert.Equal(0, mslkToMsur.InvalidCount);
        Assert.Equal(24, mdosToMdbh.InvalidCount);
    }

    [Fact]
    public void AuditDirectory_DevelopmentCorpus_ReportsCurrentCorpusShape()
    {
        Pm4CorpusAuditReport report = Pm4ResearchAuditAnalyzer.AnalyzeDirectory(Pm4TestPaths.DevelopmentDirectoryPath);

        Assert.Equal(616, report.FileCount);
        Assert.Equal(0, report.FilesWithDiagnostics);
        Assert.Equal(0, report.FilesWithUnknownChunks);

        Pm4CorpusReferenceAudit mslkToMsur = Assert.Single(report.ReferenceAudits, static audit => audit.Name == "MSLK.RefIndex->MSUR");
        Pm4CorpusReferenceAudit mdosToMdbh = Assert.Single(report.ReferenceAudits, static audit => audit.Name == "MDOS.buildingIndex->MDBH");

        Assert.Equal(4553, mslkToMsur.InvalidCount);
        Assert.Equal(24, mdosToMdbh.InvalidCount);
    }

    [Fact]
    public void LinkageDirectory_DevelopmentCorpus_PreservesCurrentMismatchAndReuseSignals()
    {
        Pm4LinkageReport report = Pm4ResearchLinkageAnalyzer.AnalyzeDirectory(Pm4TestPaths.DevelopmentDirectoryPath);

        Assert.Equal(616, report.FileCount);
        Assert.Equal(150, report.FilesWithRefIndexMismatches);
        Assert.Equal(58, report.FilesWithBadMdos);
        Assert.Equal(4553, report.TotalRefIndexMismatchCount);
        Assert.True(report.IdentitySummary.ReusedObjectIdGroupCount > 0);
        Assert.Contains(report.Notes, note => note.Contains("UniqueID", StringComparison.Ordinal));
    }

    [Fact]
    public void PlacementMath_DevelopmentTile_PreservesCurrentRangeAndTileLocalHeuristics()
    {
        Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(Pm4TestPaths.DevelopmentTilePath);

        Assert.Equal(Pm4AxisConvention.XYPlaneZUp, Pm4PlacementMath.DetectAxisConventionByRanges(document.KnownChunks.Msvt));
        Assert.Equal(Pm4AxisConvention.XYPlaneZUp, Pm4PlacementMath.DetectAxisConventionByTriangleNormals(document.KnownChunks.Msvt, document.KnownChunks.Msvi));
        Assert.Equal(Pm4AxisConvention.XYPlaneZUp, Pm4PlacementMath.DetectAxisConventionBySurfaceNormals(document.KnownChunks.Msvt, document.KnownChunks.Msvi, document.KnownChunks.Msur));
        Assert.True(Pm4PlacementMath.IsLikelyTileLocal(document.KnownChunks.Msvt));
        Assert.True(Pm4PlacementMath.ScoreAxisConventionByTriangleNormals(document.KnownChunks.Msvt, document.KnownChunks.Msvi, Pm4AxisConvention.XYPlaneZUp) > 0f);

        Pm4PlanarTransform transform = Pm4PlacementContract.GetDefaultPlanarTransform(Pm4CoordinateMode.TileLocal);
        Vector3 world = Pm4PlacementMath.ConvertPm4VertexToWorld(document.KnownChunks.Msvt[0], 0, 0, Pm4CoordinateMode.TileLocal, Pm4AxisConvention.XYPlaneZUp, transform);

        Assert.InRange(world.X, 0f, Pm4CoordinateService.TileSize);
        Assert.InRange(world.Y, 0f, Pm4CoordinateService.TileSize);
        Assert.Equal(document.KnownChunks.Msvt[0].Z, world.Z);

        Pm4PlanarTransform resolved = Pm4PlacementMath.ResolvePlanarTransform(
            document.KnownChunks.Msvt,
            document.KnownChunks.Msvi,
            document.KnownChunks.Msur,
            document.KnownChunks.Mprl,
            anchorPositionRefs: null,
            tileX: 0,
            tileY: 0,
            Pm4CoordinateMode.TileLocal,
            Pm4AxisConvention.XYPlaneZUp);

        Assert.Equal(new Pm4PlanarTransform(false, false, false), resolved);
    }

    [Fact]
    public void PlacementMath_DevelopmentTile_PreservesFallbackDrivenCoordinateModeContract()
    {
        Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(Pm4TestPaths.DevelopmentTilePath);

        Pm4CoordinateModeResolution tileLocalResolution = Pm4PlacementMath.ResolveCoordinateMode(
            document.KnownChunks.Msvt,
            document.KnownChunks.Msvi,
            document.KnownChunks.Msur,
            document.KnownChunks.Mprl,
            anchorPositionRefs: null,
            tileX: 0,
            tileY: 0,
            Pm4AxisConvention.XYPlaneZUp,
            Pm4CoordinateMode.TileLocal);

        Pm4CoordinateModeResolution worldSpaceResolution = Pm4PlacementMath.ResolveCoordinateMode(
            document.KnownChunks.Msvt,
            document.KnownChunks.Msvi,
            document.KnownChunks.Msur,
            document.KnownChunks.Mprl,
            anchorPositionRefs: null,
            tileX: 0,
            tileY: 0,
            Pm4AxisConvention.XYPlaneZUp,
            Pm4CoordinateMode.WorldSpace);

        Assert.Equal(Pm4CoordinateMode.TileLocal, tileLocalResolution.CoordinateMode);
        Assert.Equal(Pm4CoordinateMode.WorldSpace, worldSpaceResolution.CoordinateMode);
        Assert.Equal(new Pm4PlanarTransform(false, false, false), tileLocalResolution.PlanarTransform);
        Assert.Equal(new Pm4PlanarTransform(false, false, false), worldSpaceResolution.PlanarTransform);
        Assert.True(tileLocalResolution.HasTileLocalScore);
        Assert.True(tileLocalResolution.HasWorldSpaceScore);
        Assert.True(tileLocalResolution.UsedFallback);
        Assert.True(worldSpaceResolution.UsedFallback);
        Assert.Equal(tileLocalResolution.TileLocalScore, worldSpaceResolution.TileLocalScore, 3);
        Assert.Equal(tileLocalResolution.WorldSpaceScore, worldSpaceResolution.WorldSpaceScore, 3);
    }

    [Fact]
    public void PlacementMath_SyntheticWorldSpace_ResolvesQuarterTurnPlanarTransform()
    {
        List<Vector3> meshVertices =
        [
            new(0f, 10f, 0f),
            new(0f, 30f, 0f),
            new(20f, 30f, 0f),
            new(20f, 10f, 0f)
        ];
        List<uint> meshIndices = [0u, 1u, 2u, 3u];
        List<Pm4MsurEntry> surfaces =
        [
            new Pm4MsurEntry(0, 4, 0, 0, Vector3.UnitZ, 0f, 0u, 0u, 0u)
        ];
        List<Pm4MprlEntry> positionRefs =
        [
            new Pm4MprlEntry(0, 0, 0, 0, new Vector3(-10f, 0f, 20f), 0, 0)
        ];

        Pm4PlanarTransform resolved = Pm4PlacementMath.ResolvePlanarTransform(
            meshVertices,
            meshIndices,
            surfaces,
            positionRefs,
            anchorPositionRefs: null,
            tileX: 0,
            tileY: 0,
            Pm4CoordinateMode.WorldSpace,
            Pm4AxisConvention.XYPlaneZUp);

        Assert.Equal(new Pm4PlanarTransform(true, true, false), resolved);
    }

    [Fact]
    public void PlacementMath_SyntheticWorldSpace_ResolvesWorldSpaceCoordinateMode()
    {
        List<Vector3> meshVertices =
        [
            new(0f, 10f, 0f),
            new(0f, 30f, 0f),
            new(20f, 30f, 0f),
            new(20f, 10f, 0f)
        ];
        List<uint> meshIndices = [0u, 1u, 2u, 3u];
        List<Pm4MsurEntry> surfaces =
        [
            new Pm4MsurEntry(0, 4, 0, 0, Vector3.UnitZ, 0f, 0u, 0u, 0u)
        ];
        List<Pm4MprlEntry> positionRefs =
        [
            new Pm4MprlEntry(0, 0, 0, 0, new Vector3(-10f, 0f, 20f), 0, 0)
        ];

        Pm4CoordinateModeResolution resolution = Pm4PlacementMath.ResolveCoordinateMode(
            meshVertices,
            meshIndices,
            surfaces,
            positionRefs,
            anchorPositionRefs: null,
            tileX: 0,
            tileY: 0,
            Pm4AxisConvention.XYPlaneZUp,
            Pm4CoordinateMode.TileLocal);

        Assert.Equal(Pm4CoordinateMode.WorldSpace, resolution.CoordinateMode);
        Assert.Equal(new Pm4PlanarTransform(true, true, false), resolution.PlanarTransform);
        Assert.True(resolution.HasTileLocalScore);
        Assert.True(resolution.HasWorldSpaceScore);
        Assert.False(resolution.UsedFallback);
        Assert.True(resolution.WorldSpaceScore < resolution.TileLocalScore);
    }

    [Fact]
    public void PlacementMath_SyntheticCoordinateModeMissingEvidence_UsesFallback()
    {
        List<Vector3> meshVertices =
        [
            new(10f, 20f, 5f),
            new(30f, 20f, 5f),
            new(30f, 40f, 5f),
            new(10f, 40f, 5f)
        ];
        List<uint> meshIndices = [0u, 1u, 2u, 3u];
        List<Pm4MsurEntry> surfaces =
        [
            new Pm4MsurEntry(0, 4, 0, 0, Vector3.UnitZ, 0f, 0u, 0u, 0u)
        ];

        Pm4CoordinateModeResolution resolution = Pm4PlacementMath.ResolveCoordinateMode(
            meshVertices,
            meshIndices,
            surfaces,
            positionRefs: [],
            anchorPositionRefs: null,
            tileX: 3,
            tileY: 4,
            Pm4AxisConvention.XYPlaneZUp,
            Pm4CoordinateMode.WorldSpace);

        Assert.Equal(Pm4CoordinateMode.WorldSpace, resolution.CoordinateMode);
        Assert.Equal(new Pm4PlanarTransform(false, false, false), resolution.PlanarTransform);
        Assert.True(resolution.UsedFallback);
        Assert.False(resolution.HasTileLocalScore);
        Assert.False(resolution.HasWorldSpaceScore);
    }

    [Fact]
    public void PlacementMath_SyntheticWorldSpace_ComputesMeaningfulYawCorrection()
    {
        List<Vector3> meshVertices =
        [
            new(0f, 0f, 0f),
            new(0f, 40f, 0f),
            new(10f, 40f, 0f),
            new(10f, 0f, 0f)
        ];
        List<uint> meshIndices = [0u, 1u, 2u, 3u];
        List<Pm4MsurEntry> surfaces =
        [
            new Pm4MsurEntry(0, 4, 0, 0, Vector3.UnitZ, 0f, 0u, 0u, 0u)
        ];
        ushort packed45Degrees = 8192;
        List<Pm4MprlEntry> positionRefs =
        [
            new Pm4MprlEntry(0, 0, packed45Degrees, 0, new Vector3(0f, 0f, 0f), 0, 0),
            new Pm4MprlEntry(0, 0, packed45Degrees, 0, new Vector3(10f, 0f, 0f), 0, 0)
        ];

        bool resolved = Pm4PlacementMath.TryComputeWorldYawCorrectionRadians(
            meshVertices,
            meshIndices,
            surfaces,
            positionRefs,
            tileX: 0,
            tileY: 0,
            Pm4CoordinateMode.WorldSpace,
            Pm4AxisConvention.XYPlaneZUp,
            new Pm4PlanarTransform(false, false, false),
            out float yawCorrectionRadians);

        Assert.True(resolved);
        Assert.InRange(MathF.Abs(yawCorrectionRadians), 0.70f, 0.87f);
    }

    [Fact]
    public void PlacementMath_SyntheticTileLocal_ComputesExpectedSurfaceWorldCentroid()
    {
        List<Vector3> meshVertices =
        [
            new(10f, 20f, 5f),
            new(10f, 40f, 5f),
            new(30f, 40f, 5f),
            new(30f, 20f, 5f)
        ];
        List<uint> meshIndices = [0u, 1u, 2u, 3u];
        List<Pm4MsurEntry> surfaces =
        [
            new Pm4MsurEntry(0, 4, 0, 0, Vector3.UnitZ, 0f, 0u, 0u, 0u)
        ];

        Vector3 centroid = Pm4PlacementMath.ComputeSurfaceWorldCentroid(
            meshVertices,
            meshIndices,
            surfaces,
            tileX: 2,
            tileY: 1,
            Pm4CoordinateMode.TileLocal,
            Pm4AxisConvention.XYPlaneZUp,
            new Pm4PlanarTransform(false, false, false));

        Assert.Equal(563.3333f, centroid.X, 3);
        Assert.Equal(1086.6666f, centroid.Y, 3);
        Assert.Equal(5f, centroid.Z);
    }

    [Fact]
    public void PlacementMath_SyntheticWorldSpace_RotatesWorldAroundPivot()
    {
        Vector3 rotated = Pm4PlacementMath.RotateWorldAroundPivot(
            new Vector3(10f, 0f, 7f),
            Vector3.Zero,
            MathF.PI * 0.5f);

        Assert.Equal(0f, rotated.X, 3);
        Assert.Equal(10f, rotated.Y, 3);
        Assert.Equal(7f, rotated.Z, 3);
    }

    [Fact]
    public void PlacementMath_SyntheticTileLocal_ConvertsExpectedWorldPositionWithYawCorrection()
    {
        Vector3 correctedWorld = Pm4PlacementMath.ConvertPm4VertexToWorld(
            new Vector3(10f, 20f, 5f),
            tileX: 2,
            tileY: 1,
            Pm4CoordinateMode.TileLocal,
            Pm4AxisConvention.XYPlaneZUp,
            new Pm4PlanarTransform(false, false, false),
            worldPivot: new Vector3(563.3333f, 1086.6666f, 5f),
            worldYawCorrectionRadians: MathF.PI * 0.5f);

        Assert.Equal(573.3333f, correctedWorld.X, 3);
        Assert.Equal(1076.6666f, correctedWorld.Y, 3);
        Assert.Equal(5f, correctedWorld.Z, 3);
    }

    [Fact]
    public void PlacementMath_SyntheticWorldSpace_ResolvesExpectedPlacementSolutionWithoutYawCorrection()
    {
        List<Vector3> meshVertices =
        [
            new(0f, 10f, 0f),
            new(0f, 30f, 0f),
            new(20f, 30f, 0f),
            new(20f, 10f, 0f)
        ];
        List<uint> meshIndices = [0u, 1u, 2u, 3u];
        List<Pm4MsurEntry> surfaces =
        [
            new Pm4MsurEntry(0, 4, 0, 0, Vector3.UnitZ, 0f, 0u, 0u, 0u)
        ];
        List<Pm4MprlEntry> positionRefs =
        [
            new Pm4MprlEntry(0, 0, 0, 0, new Vector3(-10f, 0f, 20f), 0, 0)
        ];

        Pm4PlacementSolution solution = Pm4PlacementMath.ResolvePlacementSolution(
            meshVertices,
            meshIndices,
            surfaces,
            positionRefs,
            anchorPositionRefs: null,
            tileX: 0,
            tileY: 0,
            Pm4CoordinateMode.WorldSpace,
            Pm4AxisConvention.XYPlaneZUp);

        Assert.Equal(new Pm4PlanarTransform(true, true, false), solution.PlanarTransform);
        Assert.Equal(-10f, solution.WorldPivot.X, 3);
        Assert.Equal(20f, solution.WorldPivot.Y, 3);
        Assert.Equal(0f, solution.WorldPivot.Z, 3);
        Assert.False(solution.HasWorldYawCorrection);

        Vector3 world = Pm4PlacementMath.ConvertPm4VertexToWorld(meshVertices[0], solution);
        Assert.Equal(0f, world.X, 3);
        Assert.Equal(10f, world.Y, 3);
        Assert.Equal(0f, world.Z, 3);
    }

    [Fact]
    public void PlacementMath_SyntheticWorldSpace_ResolvesExpectedPlacementSolutionWithYawCorrection()
    {
        List<Vector3> meshVertices =
        [
            new(0f, 0f, 0f),
            new(0f, 40f, 0f),
            new(10f, 40f, 0f),
            new(10f, 0f, 0f)
        ];
        List<uint> meshIndices = [0u, 1u, 2u, 3u];
        List<Pm4MsurEntry> surfaces =
        [
            new Pm4MsurEntry(0, 4, 0, 0, Vector3.UnitZ, 0f, 0u, 0u, 0u)
        ];
        ushort packed45Degrees = 8192;
        List<Pm4MprlEntry> positionRefs =
        [
            new Pm4MprlEntry(0, 0, packed45Degrees, 0, new Vector3(0f, 0f, 0f), 0, 0),
            new Pm4MprlEntry(0, 0, packed45Degrees, 0, new Vector3(10f, 0f, 0f), 0, 0)
        ];

        Pm4PlacementSolution solution = Pm4PlacementMath.ResolvePlacementSolution(
            meshVertices,
            meshIndices,
            surfaces,
            positionRefs,
            anchorPositionRefs: null,
            tileX: 0,
            tileY: 0,
            Pm4CoordinateMode.WorldSpace,
            Pm4AxisConvention.XYPlaneZUp);

        Assert.Equal(new Pm4PlanarTransform(false, false, false), solution.PlanarTransform);
        Assert.Equal(20f, solution.WorldPivot.X, 3);
        Assert.Equal(5f, solution.WorldPivot.Y, 3);
        Assert.Equal(0f, solution.WorldPivot.Z, 3);
        Assert.True(solution.HasWorldYawCorrection);
        Assert.InRange(MathF.Abs(solution.WorldYawCorrectionRadians), 0.70f, 0.87f);

        Vector3 uncorrectedWorld = Pm4PlacementMath.ConvertPm4VertexToWorld(
            meshVertices[0],
            0,
            0,
            Pm4CoordinateMode.WorldSpace,
            Pm4AxisConvention.XYPlaneZUp,
            solution.PlanarTransform);
        Vector3 correctedWorld = Pm4PlacementMath.ConvertPm4VertexToWorld(meshVertices[0], solution);

        Assert.NotEqual(uncorrectedWorld.X, correctedWorld.X);
        Assert.NotEqual(uncorrectedWorld.Y, correctedWorld.Y);
        Assert.Equal(
            Vector2.Distance(new Vector2(uncorrectedWorld.X, uncorrectedWorld.Y), new Vector2(solution.WorldPivot.X, solution.WorldPivot.Y)),
            Vector2.Distance(new Vector2(correctedWorld.X, correctedWorld.Y), new Vector2(solution.WorldPivot.X, solution.WorldPivot.Y)),
            3);
        Assert.Equal(0f, correctedWorld.Z, 3);
    }

    [Fact]
    public void MscnDirectory_DevelopmentCorpus_ProducesExpectedHighLevelSignals()
    {
        Pm4MscnRelationshipReport report = Pm4ResearchMscnAnalyzer.AnalyzeDirectory(Pm4TestPaths.DevelopmentDirectoryPath);

        Assert.Equal(616, report.FileCount);
        Assert.Equal(309, report.FilesWithMscn);
        Assert.Equal(616, report.FilesWithTileCoordinates);
        Assert.Equal(1342410, report.TotalMscnPointCount);
        Assert.Equal(511891, report.Relationships[0].Fits);
        Assert.Equal(6201, report.Relationships[0].Misses);
        Assert.Equal(1162, report.Relationships[3].Fits);
        Assert.Equal(10, report.Relationships[4].Fits);
        Assert.Equal(1, report.CoordinateSpace.FilesTileLocalDominant);
        Assert.Equal(615, report.CoordinateSpace.FilesNoDominant);
        Assert.Equal("raw-only", report.TopInvalidMdosClusters[0].AlignmentMode);
        Assert.Contains(report.Notes, note => note.Contains("MSUR.MdosIndex", StringComparison.Ordinal));
    }

    [Fact]
    public void UnknownsDirectory_DevelopmentCorpus_ProducesExpectedHighLevelSignals()
    {
        Pm4UnknownsReport report = Pm4ResearchUnknownsAnalyzer.AnalyzeDirectory(Pm4TestPaths.DevelopmentDirectoryPath);

        Assert.Equal(616, report.FileCount);
        Assert.Equal(309, report.NonEmptyFileCount);
        Assert.Equal(598882, report.MspiInterpretation.ActiveLinkCount);
        Assert.Equal(399183, report.MspiInterpretation.IndicesModeOnlyCount);
        Assert.Equal(199699, report.MspiInterpretation.BothModesCount);
        Assert.Equal(1273335, report.LinkIdPatterns.TotalCount);
        Assert.Equal(1273335, report.LinkIdPatterns.SentinelTileLinkCount);
        Assert.Equal(0, report.LinkIdPatterns.ZeroCount);
        Assert.Equal(0, report.LinkIdPatterns.OtherCount);
        Assert.Contains(report.Unknowns, finding => finding.Name == "MSLK.RefIndex semantics");
    }
}

internal static class Pm4TestPaths
{
    public static string DevelopmentDirectoryPath => Path.Combine(GetWowViewerRoot(), "..", "gillijimproject_refactor", "test_data", "development", "World", "Maps", "development");

    public static string DevelopmentTilePath => Path.Combine(DevelopmentDirectoryPath, "development_00_00.pm4");

    private static string GetWowViewerRoot()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
                return current.FullName;

            current = current.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate the wow-viewer repository root from the test output directory.");
    }
}