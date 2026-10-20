using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4MapPartitionAuditTests
{
    [Fact]
    public void AddressedGraphResolvesOtherTileRatherThanAcceptingMisleadingLocalIndex()
    {
        var a = File("development_01_02.pm4", [Triangle(Vector3.Zero, 1)], [Surface(0, length: 1)],
            [Link(0) with { LinkId = 0xFFFF0202 }]);
        var b = File("development_02_02.pm4", [Triangle(new Vector3(20, 0, 0), 1)], [Surface(0, length: 1)],
            [Link(0) with { LinkId = 0xFFFF0201 }]);
        var report = Pm4MapPartitionAudit.Analyze([a, b]);
        Assert.Equal(2, report.Hypotheses.Single(h => h.Name == "window-ref-all").Components.Count);
        Assert.Single(report.Hypotheses.Single(h => h.Name == "tile-addressed-window-ref-all").Components);
        Assert.All(report.Links, l => Assert.NotEqual(l.FileId, l.AddressedFileId));
        Assert.Equal(2, report.AddressedWindowReferenceEdges.Count);
    }

    [Fact]
    public void UniformScaledCopiesAcrossFilesShareFamilyButKeepAbsoluteScaleAndProvenance()
    {
        var report = Pm4MapPartitionAudit.Analyze([
            File("a", [Triangle(Vector3.Zero, 1)], [Surface(0)]),
            File("b", [Triangle(new Vector3(100, 0, 0), 2)], [Surface(0)]),
            File("c", [[new Vector3(200, 0, 0), new Vector3(204, 0, 0), new Vector3(200, 1, 0)]], [Surface(0)]),
        ]);
        var hypothesis = report.Hypotheses.Single(h => h.Name == "geometry-map");
        Assert.Equal(3, hypothesis.Components.Count);
        var family = Assert.Single(hypothesis.CandidateScaleFamilies);
        Assert.Equal(2, family.ComponentIds.Count);
        Assert.Equal(2f, family.ScaleRatio, 4);
        Assert.Equal(2, family.Files);
        Assert.Equal(3, report.Surfaces.Select(n => (n.FileId, n.LocalIndex)).Distinct().Count());
    }

    [Fact]
    public void SameLocalIndicesDoNotCreateCrossFileLinksButCoincidentGeometryCan()
    {
        var file = File("a", [Triangle(Vector3.Zero, 1)], [Surface(0, length: 1)], [Link(0, group: 7)]);
        var second = file with { FileId = "b" };
        var report = Pm4MapPartitionAudit.Analyze([file, second]);
        Assert.Equal(2, report.Hypotheses.Single(h => h.Name == "window-ref-all").Components.Count);
        Assert.Single(report.Hypotheses.Single(h => h.Name == "geometry-map").Components);
        Assert.Equal(3, report.CrossFileGeometry.Count);
        Assert.All(report.Links, l => Assert.Equal(l.FileId, report.Surfaces[l.SurfaceReferenceId!.Value].FileId));
    }

    [Fact]
    public void AnchorOnlyLinkCanBridgeDisconnectedGeometryWithoutBeingDropped()
    {
        var report = Pm4MapPartitionAudit.Analyze([File("a",
            [Triangle(Vector3.Zero, 1), Triangle(new Vector3(20, 0, 0), 1)],
            [Surface(0, length: 1), Surface(3, start: 1)], [Link(1)])]);
        Assert.Equal(2, report.Hypotheses.Single(h => h.Name == "geometry-map").Components.Count);
        Assert.Single(report.Hypotheses.Single(h => h.Name == "window-ref-anchor").Components);
        Assert.Equal(2, report.Hypotheses.Single(h => h.Name == "window-ref-path").Components.Count);
        Assert.Equal(1, report.AnchorOnlyLinks);
        Assert.Equal(0, report.UnownedLinks);
    }

    [Fact]
    public void InvalidAndUnownedLinksSurviveAndAllFieldIncidencesRemainAvailable()
    {
        var report = Pm4MapPartitionAudit.Analyze([File("a", [Triangle(Vector3.Zero, 1)],
            [Surface(0, start: 999, length: 3)], [Link(99, group: 12), Link(0, group: 13)])]);
        Assert.Equal(2, report.Links.Count);
        Assert.Equal(2, report.UnownedLinks);
        Assert.Equal(1, report.InvalidSurfaceReferences);
        Assert.Equal(1, report.InvalidSurfaceWindows);
        Assert.Empty(report.WindowReferenceEdges);
        Assert.Equal(new uint[] { 12, 13 }, report.FieldValues.Where(v => v.Field == "_0x04-GroupObjectId").Select(v => v.RawValue));
        Assert.Contains(report.SurfaceFieldValues, v => v.Field == "_0x02-attrMask" && v.RawValue == 3);
        Assert.Equal(3, report.WindowControls.Count);
    }

    [Fact]
    public void ZeroCk24IsNotSilentlyEquatedWithZeroRawWordAndKeyedContextIsRetained()
    {
        var report = Pm4MapPartitionAudit.Analyze([File("a",
            [Triangle(Vector3.Zero, 1), Triangle(new Vector3(10, 0, 0), 1)],
            [Surface(0, raw: 1, length: 1), Surface(3, raw: 0x42123456, start: 1)], [Link(1)])]);
        Assert.Equal(1, report.ZeroCk24SurfaceCount);
        Assert.Equal(0, report.RawWordZeroSurfaceCount);
        var merged = Assert.Single(report.Hypotheses.Single(h => h.Name == "window-ref-all").Components);
        Assert.Equal(2, merged.SurfaceIds.Count);
        Assert.Equal(1, merged.ZeroCk24Surfaces);
    }

    [Fact]
    public void FirstLinkTypeIsNotChosenAndOwnershipOverlapsAreRetained()
    {
        var report = Pm4MapPartitionAudit.Analyze([File("a",
            [Triangle(Vector3.Zero, 1), Triangle(new Vector3(10, 0, 0), 1)],
            [Surface(0, length: 2), Surface(3, length: 1)],
            [Link(1, type: 2), Link(1, type: 4)])]);
        Assert.Equal(2, report.Links[0].WindowOwnerIds.Count);
        Assert.Equal(new uint[] { 2, 4 }, report.FieldValues.Where(v => v.Field == "_0x00-TypeFlags").Select(v => v.RawValue));
        Assert.Equal(1, report.WindowControls.Single(c => c.LengthHypothesis == "attrMask-as-length").MultiplyOwnedLinks);
    }

    [Fact]
    public void AuthoredFamiliesRetainDifferentScalesTransformsAndUnpairedTiles()
    {
        var a = new AdtModelPlacement(0, "World\\same.m2", 1, new Vector3(1, 2, 3), Vector3.Zero, 1);
        var b = a with { UniqueId = 2, Position = new Vector3(11, 12, 13), Scale = 2 };
        var catalog = new AdtPlacementCatalog("authored", MapFileKind.AdtObj, [a.ModelPath], [], [a, b], []);
        var report = Pm4MapPartitionAudit.Analyze([], [new Pm4MapPlacementInput("tile-without-pm4.adt", catalog)]);
        Assert.Equal(2, report.AuthoredPlacements.Count);
        Assert.Equal(new float[] { 1, 2 }, Assert.Single(report.AuthoredM2Families).Scales);
        Assert.All(report.AuthoredPlacements, p => Assert.Equal("tile-without-pm4.adt", p.FileId));
        Assert.Equal(b.Position, report.AuthoredPlacements[1].Position);
    }

    [Fact]
    public void LocalVersusMapScopedGroupReuseIsACompetingHypothesisNotAnImplicitJoin()
    {
        var report = Pm4MapPartitionAudit.Analyze([
            File("a", [Triangle(Vector3.Zero, 1)], [Surface(0, length: 1)], [Link(0, group: 7)]),
            File("b", [Triangle(new Vector3(100, 0, 0), 1)], [Surface(0, length: 1)], [Link(0, group: 7)]),
        ]);
        Assert.Equal(2, report.Hypotheses.Single(h => h.Name == "group-value-owner-file-scope").Components.Count);
        Assert.Single(report.Hypotheses.Single(h => h.Name == "group-value-owner-map-scope").Components);
        Assert.Equal(2, report.Hypotheses.Single(h => h.Name == "geometry-map").Components.Count);
        Assert.Equal(2, report.FieldValues.Single(v => v.Field == "_0x04-GroupObjectId").Files);
    }

    private static Vector3[] Triangle(Vector3 offset, float scale) => [offset, offset + new Vector3(scale, 0, 0), offset + new Vector3(0, scale, 0)];
    private static Pm4MsurEntry Surface(uint first, uint raw = 0, uint start = 0, byte length = 0) =>
        new(0, 3, length, 0, Vector3.UnitZ, 0, first, start, raw);
    private static Pm4MslkEntry Link(ushort target, uint group = 0, byte type = 0) =>
        new(type, 0, 0, group, -1, 0, 0, target, 0);
    private static Pm4MapAuditInput File(string id, Vector3[][] triangles, Pm4MsurEntry[] surfaces, Pm4MslkEntry[]? links = null)
    {
        Vector3[] vertices = triangles.SelectMany(t => t).ToArray();
        var known = new Pm4KnownChunkSet(null, links ?? [], [], [], vertices,
            Enumerable.Range(0, vertices.Length).Select(i => (uint)i).ToArray(), surfaces, [], [], [], null, [], [], [], []);
        return new Pm4MapAuditInput(id, new Pm4ResearchDocument(id, 2, [], known, []));
    }
}
