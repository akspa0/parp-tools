using System.Linq;
using System.Numerics;
using WowViewer.Core.PM4.Footprint;
using WowViewer.Core.PM4.Matching;
using Xunit;

namespace WowViewer.Core.Tests.PM4;

public sealed class Pm4ShapeMatchAnalysisServiceTests
{
    private static Pm4FootprintPrism Prism(float x = 100f, float y = 200f)
        => new(
            ConvexHull2D: [],
            TopZ: 14f,
            BaseZ: 10f,
            Height: 4f,
            Centroid: new Vector3(x, y, 12f),
            YawDegrees: 0f,
            UnrotatedLength: 10f,
            UnrotatedWidth: 5f,
            Walls: [],
            WallTriangles: []);

    private static Pm4ModelCandidate Candidate(string path, float length, float width, float height)
        => Pm4ModelCandidate.Create(
            path,
            "WMO",
            new Vector3(-length * 0.5f, -width * 0.5f, 0f),
            new Vector3(length * 0.5f, width * 0.5f, height));

    [Fact]
    public void Calibration_AppliesOnceAndResets()
    {
        var calibration = new Pm4CentroidCalibration(new Vector3(-0.75f, 0f, 0f), "validation tile");
        Vector3 solved = new(100f, 200f, 10f);

        Assert.Equal(new Vector3(99.25f, 200f, 10f), calibration.Apply(solved));
        Assert.Equal(solved, Pm4CentroidCalibration.None.Apply(solved));
        Assert.True(Pm4CentroidCalibration.None.IsIdentity);
        Assert.False(calibration.IsIdentity);
        Assert.Equal(new Vector3(0.75f, 0f, 0f), calibration.Inverted().Offset);
    }

    [Fact]
    public void MeasureCalibration_AggregatesSignedOffsets()
    {
        var samples = new[]
        {
            (Solved: new Vector3(100f, 200f, 10f), Authored: new Vector3(99.5f, 200f, 10f)),
            (Solved: new Vector3(110f, 210f, 10f), Authored: new Vector3(109.5f, 210f, 10f)),
        };

        Pm4CentroidMeasurement measurement = Pm4ShapeMatchAnalysisService.MeasureCalibration(samples, "test tile");

        Assert.Equal(2, measurement.SampleCount);
        Assert.Equal(-0.5f, measurement.MeanOffset.X, 3);
        Assert.Equal(0f, measurement.MeanOffset.Y, 3);
        Assert.Equal(-0.5f, measurement.MinOffset.X, 3);
        Assert.Equal(-0.5f, measurement.MaxOffset.X, 3);
    }

    [Fact]
    public void Analyze_RanksMatchingCandidateFirstAndAppliesCalibration()
    {
        Pm4FootprintPrism prism = Prism();
        var candidates = new[]
        {
            Candidate("world\\wrong.wmo", 4f, 4f, 4f),
            Candidate("world\\right.wmo", 10f, 5f, 4f),
        };
        var calibration = new Pm4CentroidCalibration(new Vector3(-0.5f, 0f, 0f), "test");

        Pm4ShapeMatchProposal proposal = Pm4ShapeMatchAnalysisService.Analyze(7, prism, candidates, calibration);

        Assert.False(proposal.IsUnmatched);
        Assert.NotNull(proposal.TopCandidate);
        Assert.Equal("world\\right.wmo", proposal.TopCandidate!.ModelPath);
        Assert.Equal(prism.Centroid.X - 0.5f, proposal.TopCandidate.SolvedPosition.X, 3);
    }

    [Fact]
    public void Analyze_ReportsUnmatchedWhenNoCandidateClearsThreshold()
    {
        Pm4FootprintPrism prism = Prism();
        var candidates = new[] { Candidate("world\\tiny.wmo", 1f, 1f, 1f) };

        Pm4ShapeMatchProposal proposal = Pm4ShapeMatchAnalysisService.Analyze(8, prism, candidates, Pm4CentroidCalibration.None);

        Assert.True(proposal.IsUnmatched);
        Assert.Empty(proposal.Candidates);
        Assert.Equal(0.0, proposal.Confidence);
    }

    [Fact]
    public void AnalyzeBatch_FlagsDuplicateClaims()
    {
        var candidates = new[] { Candidate("world\\right.wmo", 10f, 5f, 4f) };
        var hulls = new[]
        {
            (HullKey: 1u, Prism: Prism(100f, 200f)),
            (HullKey: 2u, Prism: Prism(100.5f, 200f)),
        };

        IReadOnlyList<Pm4ShapeMatchProposal> proposals = Pm4ShapeMatchAnalysisService.AnalyzeBatch(
            hulls, candidates, Pm4CentroidCalibration.None);

        Assert.All(proposals, proposal => Assert.True(proposal.HasConflict));
        Assert.Single(Pm4ShapeMatchAnalysisService.FindConflicts(proposals));
    }

    [Fact]
    public void HealthCheck_MarksMissingAssetsAndFiltersThem()
    {
        var candidates = new[]
        {
            Candidate("world\\present.wmo", 10f, 5f, 4f),
            Candidate("world\\missing.wmo", 10f, 5f, 4f),
        };

        IReadOnlyList<Pm4ShapeLibraryHealthEntry> health = Pm4ShapeLibraryHealth.Check(
            candidates.Select(candidate => candidate.ModelPath),
            path => path.Contains("present", StringComparison.OrdinalIgnoreCase));

        Assert.Contains(health, entry => entry.ModelPath.Contains("missing") && !entry.IsHealthy);
        var healthByPath = health.ToDictionary(entry => entry.ModelPath, entry => entry.IsHealthy);
        IReadOnlyList<Pm4ModelCandidate> healthy = Pm4ShapeLibraryHealth.FilterHealthy(candidates, healthByPath);
        Assert.Single(healthy);
        Assert.Contains("present", healthy[0].ModelPath);
    }
}
