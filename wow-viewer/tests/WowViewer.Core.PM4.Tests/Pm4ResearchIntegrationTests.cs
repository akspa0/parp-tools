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
    public void PlacementMath_SyntheticWorldSpace_BuildsDistinctSortedConnectorKeys()
    {
        List<Vector3> exteriorVertices =
        [
            new(0f, 8f, 4f),
            new(8f, 8f, 4f),
            new(8f, 0f, 4f)
        ];
        List<Pm4MsurEntry> surfaces =
        [
            new Pm4MsurEntry(0, 0, 0, 0, Vector3.UnitZ, 0f, 0u, 1u, 0u),
            new Pm4MsurEntry(0, 0, 0, 0, Vector3.UnitZ, 0f, 0u, 0u, 0u),
            new Pm4MsurEntry(0, 0, 0, 0, Vector3.UnitZ, 0f, 0u, 1u, 0u),
            new Pm4MsurEntry(0, 0, 0, 0, Vector3.UnitZ, 0f, 0u, 2u, 0u),
            new Pm4MsurEntry(0, 0, 0, 0, Vector3.UnitZ, 0f, 0u, 9u, 0u)
        ];
        Pm4PlacementSolution placement = new(
            TileX: 0,
            TileY: 0,
            CoordinateMode: Pm4CoordinateMode.WorldSpace,
            AxisConvention: Pm4AxisConvention.XYPlaneZUp,
            PlanarTransform: new Pm4PlanarTransform(false, false, false),
            WorldPivot: Vector3.Zero,
            WorldYawCorrectionRadians: 0f);

        IReadOnlyList<Pm4ConnectorKey> connectorKeys = Pm4PlacementMath.BuildConnectorKeys(exteriorVertices, surfaces, placement);

        Assert.Equal(
            [
                new Pm4ConnectorKey(0, 4, 2),
                new Pm4ConnectorKey(4, 0, 2),
                new Pm4ConnectorKey(4, 4, 2)
            ],
            connectorKeys);
    }

    [Fact]
    public void PlacementMath_SyntheticWorldSpace_AppliesYawCorrectionWhenBuildingConnectorKeys()
    {
        List<Vector3> exteriorVertices =
        [
            new(0f, 8f, 4f)
        ];
        List<Pm4MsurEntry> surfaces =
        [
            new Pm4MsurEntry(0, 0, 0, 0, Vector3.UnitZ, 0f, 0u, 0u, 0u)
        ];
        Pm4PlacementSolution placement = new(
            TileX: 0,
            TileY: 0,
            CoordinateMode: Pm4CoordinateMode.WorldSpace,
            AxisConvention: Pm4AxisConvention.XYPlaneZUp,
            PlanarTransform: new Pm4PlanarTransform(false, false, false),
            WorldPivot: Vector3.Zero,
            WorldYawCorrectionRadians: MathF.PI * 0.5f);

        IReadOnlyList<Pm4ConnectorKey> connectorKeys = Pm4PlacementMath.BuildConnectorKeys(exteriorVertices, surfaces, placement);

        Pm4ConnectorKey connectorKey = Assert.Single(connectorKeys);
        Assert.Equal(new Pm4ConnectorKey(0, 4, 2), connectorKey);
    }

    [Fact]
    public void PlacementMath_SyntheticNeighborGroupsWithSharedConnectors_BuildsMergedGroupMap()
    {
        Pm4ObjectGroupKey firstKey = new(0, 0, 9u);
        Pm4ObjectGroupKey secondKey = new(1, 0, 7u);
        Pm4ObjectGroupKey thirdKey = new(2, 0, 5u);

        List<Pm4ConnectorMergeCandidate> groups =
        [
            new(
                firstKey,
                new Vector3(0f, 0f, 0f),
                new Vector3(32f, 32f, 16f),
                new Vector3(16f, 16f, 8f),
                new HashSet<Pm4ConnectorKey>
                {
                    new(10, 10, 2),
                    new(12, 10, 2),
                    new(14, 10, 2)
                }),
            new(
                secondKey,
                new Vector3(24f, 0f, 0f),
                new Vector3(56f, 32f, 16f),
                new Vector3(40f, 16f, 8f),
                new HashSet<Pm4ConnectorKey>
                {
                    new(10, 10, 2),
                    new(12, 10, 2),
                    new(18, 10, 2)
                }),
            new(
                thirdKey,
                new Vector3(900f, 900f, 0f),
                new Vector3(932f, 932f, 16f),
                new Vector3(916f, 916f, 8f),
                new HashSet<Pm4ConnectorKey>
                {
                    new(40, 40, 2),
                    new(42, 40, 2)
                })
        ];

        IReadOnlyDictionary<Pm4ObjectGroupKey, Pm4ObjectGroupKey> mergedGroupMap = Pm4PlacementMath.BuildMergedGroupMap(groups);

        Assert.Equal(firstKey, mergedGroupMap[firstKey]);
        Assert.Equal(firstKey, mergedGroupMap[secondKey]);
        Assert.Equal(thirdKey, mergedGroupMap[thirdKey]);
    }

    [Fact]
    public void PlacementMath_SyntheticSameTileGroups_DoNotMergeEvenWithSharedConnectors()
    {
        Pm4ObjectGroupKey firstKey = new(4, 4, 100u);
        Pm4ObjectGroupKey secondKey = new(4, 4, 200u);

        List<Pm4ConnectorMergeCandidate> groups =
        [
            new(
                firstKey,
                new Vector3(0f, 0f, 0f),
                new Vector3(32f, 32f, 16f),
                new Vector3(16f, 16f, 8f),
                new HashSet<Pm4ConnectorKey>
                {
                    new(10, 10, 2),
                    new(12, 10, 2)
                }),
            new(
                secondKey,
                new Vector3(8f, 8f, 0f),
                new Vector3(40f, 40f, 16f),
                new Vector3(24f, 24f, 8f),
                new HashSet<Pm4ConnectorKey>
                {
                    new(10, 10, 2),
                    new(12, 10, 2)
                })
        ];

        IReadOnlyDictionary<Pm4ObjectGroupKey, Pm4ObjectGroupKey> mergedGroupMap = Pm4PlacementMath.BuildMergedGroupMap(groups);

        Assert.Equal(firstKey, mergedGroupMap[firstKey]);
        Assert.Equal(secondKey, mergedGroupMap[secondKey]);
    }

    [Fact]
    public void CorrelationMath_SyntheticMetrics_ComputesExpectedOverlapAndDistanceSignals()
    {
        Vector2[] referenceHull =
        [
            new(0f, 0f),
            new(10f, 0f),
            new(10f, 10f),
            new(0f, 10f)
        ];
        Vector2[] candidateHull =
        [
            new(5f, 0f),
            new(15f, 0f),
            new(15f, 10f),
            new(5f, 10f)
        ];

        Pm4CorrelationMetrics metrics = Pm4CorrelationMath.EvaluateMetrics(
            referenceBoundsMin: new Vector3(0f, 0f, 0f),
            referenceBoundsMax: new Vector3(10f, 10f, 10f),
            referenceCenter: new Vector3(5f, 5f, 5f),
            referenceFootprintHull: referenceHull,
            referenceFootprintArea: 100f,
            candidateBoundsMin: new Vector3(5f, 0f, 2f),
            candidateBoundsMax: new Vector3(15f, 10f, 12f),
            candidateCenter: new Vector3(10f, 5f, 7f),
            candidateFootprintHull: candidateHull,
            candidateFootprintArea: 100f);

        Assert.Equal(0f, metrics.PlanarGap, 3);
        Assert.Equal(0f, metrics.VerticalGap, 3);
        Assert.Equal(5.385f, metrics.CenterDistance, 3);
        Assert.Equal(0.5f, metrics.PlanarOverlapRatio, 3);
        Assert.Equal(0.4f, metrics.VolumeOverlapRatio, 3);
        Assert.Equal(0.5f, metrics.FootprintOverlapRatio, 3);
        Assert.Equal(1f, metrics.FootprintAreaRatio, 3);
        Assert.Equal(5f, metrics.FootprintDistance, 3);
    }

    [Fact]
    public void CorrelationMath_SameTileCandidateOutranksCrossTileCandidate()
    {
        Pm4CorrelationCandidateScore sameTileScore = new(
            SameTile: true,
            Metrics: new Pm4CorrelationMetrics(20f, 20f, 50f, 0.05f, 0.01f, 0.05f, 0.1f, 40f),
            BoundsMin: Vector3.Zero,
            BoundsMax: Vector3.One,
            Center: Vector3.Zero);
        Pm4CorrelationCandidateScore crossTileScore = new(
            SameTile: false,
            Metrics: new Pm4CorrelationMetrics(0f, 0f, 1f, 0.9f, 0.9f, 0.9f, 1f, 0f),
            BoundsMin: Vector3.Zero,
            BoundsMax: Vector3.One,
            Center: Vector3.One);

        int comparison = Pm4CorrelationMath.CompareCandidateScores(sameTileScore, crossTileScore);

        Assert.True(comparison < 0);
    }

    [Fact]
    public void CorrelationMath_WhenTileParityMatches_BetterFootprintScoreWins()
    {
        Pm4CorrelationCandidateScore strongerScore = new(
            SameTile: false,
            Metrics: new Pm4CorrelationMetrics(5f, 2f, 10f, 0.6f, 0.4f, 0.8f, 0.9f, 3f),
            BoundsMin: Vector3.Zero,
            BoundsMax: Vector3.One,
            Center: Vector3.Zero);
        Pm4CorrelationCandidateScore weakerScore = new(
            SameTile: false,
            Metrics: new Pm4CorrelationMetrics(0f, 0f, 1f, 0.9f, 0.9f, 0.2f, 1f, 0f),
            BoundsMin: Vector3.Zero,
            BoundsMax: Vector3.One,
            Center: Vector3.One);

        int comparison = Pm4CorrelationMath.CompareCandidateScores(strongerScore, weakerScore);

        Assert.True(comparison < 0);
    }

    [Fact]
    public void CorrelationMath_SyntheticObjectStates_ComputesBoundsHullAndArea()
    {
        Pm4CorrelationObjectDescriptor descriptor = new(
            Ck24: 0x12345678u,
            Ck24Type: 0x12,
            ObjectPartId: 4,
            LinkGroupObjectId: 9u,
            SurfaceCount: 6,
            LinkedPositionRefCount: 2,
            DominantGroupKey: 3,
            DominantAttributeMask: 7,
            DominantMdosIndex: 11u,
            AverageSurfaceHeight: 5f);
        Pm4CorrelationObjectInput input = new(
            TileX: 1,
            TileY: 2,
            GroupKey: new Pm4ObjectGroupKey(1, 2, 0x12345678u),
            Object: descriptor,
            WorldGeometryPoints:
            [
                new Vector3(0f, 0f, 0f),
                new Vector3(10f, 0f, 0f),
                new Vector3(10f, 10f, 0f),
                new Vector3(0f, 10f, 0f),
                new Vector3(3f, 3f, 5f)
            ],
            EmptyGeometryCenter: new Vector3(-1f, -1f, -1f));

        Pm4CorrelationObjectState state = Assert.Single(Pm4CorrelationMath.BuildObjectStates([input]));

        Assert.Equal(new Pm4ObjectGroupKey(1, 2, 0x12345678u), state.GroupKey);
        Assert.Equal(descriptor, state.Object);
        Assert.Equal(new Vector3(0f, 0f, 0f), state.BoundsMin);
        Assert.Equal(new Vector3(10f, 10f, 5f), state.BoundsMax);
        Assert.Equal(new Vector3(5f, 5f, 2.5f), state.Center);
        Assert.Equal(100f, state.FootprintArea, 3);
        Assert.Equal(
            [
                new Vector2(0f, 0f),
                new Vector2(10f, 0f),
                new Vector2(10f, 10f),
                new Vector2(0f, 10f)
            ],
            state.FootprintHull);
    }

    [Fact]
    public void CorrelationMath_SyntheticObjectStates_UsesFallbackCenterForEmptyGeometry()
    {
        Pm4CorrelationObjectInput input = new(
            TileX: 0,
            TileY: 0,
            GroupKey: new Pm4ObjectGroupKey(0, 0, 1u),
            Object: new Pm4CorrelationObjectDescriptor(1u, 0, 0, 0u, 0, 0, 0, 0, 0u, 0f),
            WorldGeometryPoints: Array.Empty<Vector3>(),
            EmptyGeometryCenter: new Vector3(9f, 8f, 7f));

        Pm4CorrelationObjectState state = Assert.Single(Pm4CorrelationMath.BuildObjectStates([input]));

        Assert.Equal(new Vector3(9f, 8f, 7f), state.BoundsMin);
        Assert.Equal(new Vector3(9f, 8f, 7f), state.BoundsMax);
        Assert.Equal(new Vector3(9f, 8f, 7f), state.Center);
        Assert.Empty(state.FootprintHull);
        Assert.Equal(0f, state.FootprintArea, 3);
    }

    [Fact]
    public void CorrelationMath_TransformedFootprintHull_AppliesWorldTransform()
    {
        Vector3[] sourcePoints =
        [
            new(0f, 0f, 0f),
            new(2f, 0f, 0f),
            new(2f, 1f, 0f),
            new(0f, 1f, 0f)
        ];

        Vector2[] hull = Pm4CorrelationMath.BuildTransformedFootprintHull(
            sourcePoints,
            Matrix4x4.CreateTranslation(5f, 7f, 0f));

        Assert.Equal(
            [
                new Vector2(5f, 7f),
                new Vector2(7f, 7f),
                new Vector2(7f, 8f),
                new Vector2(5f, 8f)
            ],
            hull);
        Assert.Equal(2f, Pm4CorrelationMath.ComputeFootprintArea(hull), 3);
    }

    [Fact]
    public void CorrelationMath_GeometryInputs_BuildObjectStatesWithoutViewerSpecificWorldPointAssembly()
    {
        Pm4CorrelationGeometryInput input = new(
            TileX: 3,
            TileY: 4,
            GroupKey: new Pm4ObjectGroupKey(3, 4, 0x55u),
            Object: new Pm4CorrelationObjectDescriptor(0x55u, 1, 2, 3u, 4, 5, 6, 7, 8u, 9f),
            Lines:
            [
                new Pm4GeometryLineSegment(new Vector3(0f, 0f, 0f), new Vector3(2f, 0f, 0f))
            ],
            Triangles:
            [
                new Pm4GeometryTriangle(new Vector3(0f, 0f, 0f), new Vector3(0f, 1f, 0f), new Vector3(2f, 1f, 0f))
            ],
            GeometryTransform: Matrix4x4.CreateTranslation(10f, 20f, 5f));

        Pm4CorrelationObjectState state = Assert.Single(Pm4CorrelationMath.BuildObjectStatesFromGeometry([input]));

        Assert.Equal(new Vector3(10f, 20f, 5f), state.BoundsMin);
        Assert.Equal(new Vector3(12f, 21f, 5f), state.BoundsMax);
        Assert.Equal(new Vector3(11f, 20.5f, 5f), state.Center);
        Assert.Equal(2f, state.FootprintArea, 3);
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
    public void PlacementMath_SyntheticLinkedPositionRefs_SummarizesHeadingAndFloorSignals()
    {
        List<Pm4MprlEntry> positionRefs =
        [
            new(0, 0, 0, 0, Vector3.Zero, -1, 0),
            new(0, 0, 16384, 0, Vector3.Zero, 5, 0),
            new(0, 0, 0, 0, Vector3.Zero, 0, 1)
        ];

        Pm4LinkedPositionRefSummary summary = Pm4PlacementMath.SummarizeLinkedPositionRefs(positionRefs);

        Assert.Equal(3, summary.TotalCount);
        Assert.Equal(2, summary.NormalCount);
        Assert.Equal(1, summary.TerminatorCount);
        Assert.Equal(-1, summary.FloorMin);
        Assert.Equal(5, summary.FloorMax);
        Assert.Equal(0f, summary.HeadingMinDegrees, 3);
        Assert.Equal(90f, summary.HeadingMaxDegrees, 3);
        Assert.Equal(45f, summary.HeadingMeanDegrees, 3);
        Assert.True(summary.HasNormalHeadings);
    }

    [Fact]
    public void PlacementMath_SyntheticLinkedPositionRefs_UsesNaNHeadingsWhenOnlyTerminatorsExist()
    {
        List<Pm4MprlEntry> positionRefs =
        [
            new(0, 0, 0, 0, Vector3.Zero, 7, 1),
            new(0, 0, 8192, 0, Vector3.Zero, 3, 2)
        ];

        Pm4LinkedPositionRefSummary summary = Pm4PlacementMath.SummarizeLinkedPositionRefs(positionRefs);

        Assert.Equal(2, summary.TotalCount);
        Assert.Equal(0, summary.NormalCount);
        Assert.Equal(2, summary.TerminatorCount);
        Assert.Equal(0, summary.FloorMin);
        Assert.Equal(0, summary.FloorMax);
        Assert.True(float.IsNaN(summary.HeadingMinDegrees));
        Assert.True(float.IsNaN(summary.HeadingMaxDegrees));
        Assert.True(float.IsNaN(summary.HeadingMeanDegrees));
        Assert.False(summary.HasNormalHeadings);
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

    [Fact]
    public void Ck24Forensics_DevelopmentTile_412CDC_PreservesComponentGroupingAndHeadingEvidence()
    {
        Pm4Ck24ForensicsReport report = Pm4Ck24ForensicsAnalyzer.Analyze(
            Pm4ResearchReader.ReadFile(Pm4TestPaths.DevelopmentTilePath),
            0x412CDCu);

        Assert.Equal((ushort)11484, report.Ck24ObjectId);
        Assert.Equal(72, report.SurfaceCount);
        Assert.Equal(12, report.DistinctLinkGroupCount);
        Assert.Equal(72, report.LinkGroups.Sum(static group => group.SurfaceCount));
        Assert.Contains(report.LinkGroups, static group => group.LinkGroupObjectId == 0 && group.SurfaceCount == 9);
        Assert.Contains(report.LinkGroups, static group => group.LinkGroupObjectId == 1779 && group.SurfaceCount == 34);
        Assert.Equal(3.078537f, Assert.IsType<float>(report.PlacementComparison.MprlHeadingMeanDegrees), 3);
        Assert.Contains(report.Notes, static note => note.Contains("terrain/object seam evidence", StringComparison.Ordinal));
    }

    [Fact]
    public void Ck24Forensics_DevelopmentTile_41C0F5_PreservesSparseLinkageCase()
    {
        Pm4Ck24ForensicsReport report = Pm4Ck24ForensicsAnalyzer.Analyze(
            Pm4ResearchReader.ReadFile(Pm4TestPaths.DevelopmentTilePath),
            0x41C0F5u);

        Assert.Equal((ushort)49397, report.Ck24ObjectId);
        Assert.Equal(7, report.SurfaceCount);
        Assert.Equal(1, report.DistinctLinkGroupCount);
        Pm4ForensicsLinkGroupReport linkGroup = Assert.Single(report.LinkGroups);
        Assert.Equal((uint)4704, linkGroup.LinkGroupObjectId);
        Assert.Empty(linkGroup.ReferencedPositionRefIndices);
        Assert.Empty(linkGroup.MprlRows);
        Assert.Null(report.PlacementComparison.MprlHeadingMeanDegrees);
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