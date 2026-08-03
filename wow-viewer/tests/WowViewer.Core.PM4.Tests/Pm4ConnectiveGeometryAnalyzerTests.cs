using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

/// <summary>
/// Detector-power tests for <see cref="Pm4ConnectiveGeometryAnalyzer"/>.
/// </summary>
/// <remarks>
/// These exist because the analyzer replaces a measurement that could not measure. The legacy
/// indices-vs-triangles mode counters test <c>first + count &lt;= mspiCount</c> against
/// <c>3*first + 3*count &lt;= mspiCount</c>, and the second implies the first for every non-negative
/// input, so their trianglesOnly bucket is zero by construction. Before trusting a corpus claim from
/// the replacement, the replacement must be shown to give different answers for inputs that differ.
/// </remarks>
public class Pm4ConnectiveGeometryAnalyzerTests
{
    [Fact]
    public void VerticalQuadWindow_IsCoplanarNotCollinearAndPerpendicularToSurfaceAxis()
    {
        // A wall: four corners in a vertical plane, normal has no Z component.
        Vector3[] quad =
        [
            new(0f, 0f, 0f),
            new(10f, 0f, 0f),
            new(10f, 0f, 5f),
            new(0f, 0f, 5f)
        ];

        Pm4ConnectiveGeometryReport report = AnalyzeWindows(quad, [4], floorNormals: true);

        Assert.Equal(1, report.Topology.WindowsMeasured);
        Assert.Equal(1, report.Topology.CoplanarWindows);
        Assert.Equal(0, report.Topology.CollinearWindows);
        Assert.Equal(0, report.Topology.ClosedWindows);
        Assert.Equal(0, report.Topology.MultipleOfThreeWindows);

        // The surface set is Z-dominant, so the quad must register as perpendicular to Z.
        Assert.Equal(0, report.PathWindowOrientation.DominantZ);
        Assert.Equal(1, report.PathWindowOrientation.NearPerpendicularToDominantAxis);
        Assert.Equal(0d, report.PathWindowOrientation.MeanAbsNormalZ, 3);
    }

    [Fact]
    public void CollinearWindow_IsReportedAsCollinearAndNotCoplanar()
    {
        // A polyline degenerate to a straight run — must NOT be mistaken for a face.
        Vector3[] line =
        [
            new(0f, 0f, 0f),
            new(1f, 0f, 0f),
            new(2f, 0f, 0f),
            new(3f, 0f, 0f)
        ];

        Pm4ConnectiveGeometryReport report = AnalyzeWindows(line, [4], floorNormals: true);

        Assert.Equal(1, report.Topology.CollinearWindows);
        Assert.Equal(0, report.Topology.CoplanarWindows);
    }

    [Fact]
    public void ClosedWindow_IsDetectedWhenFirstIndexRepeatsAsLast()
    {
        Vector3[] triangle =
        [
            new(0f, 0f, 0f),
            new(4f, 0f, 0f),
            new(4f, 0f, 3f)
        ];

        // Indices 0,1,2,0 — an explicitly closed loop.
        Pm4ConnectiveGeometryReport report = Analyze(triangle, [0u, 1u, 2u, 0u], [4], floorNormals: true);

        Assert.Equal(1, report.Topology.ClosedWindows);
        Assert.Equal(1, report.Topology.WindowsWithDuplicateVertices);
    }

    [Fact]
    public void TriangleListWindow_ShowsNoDegenerateTriples_WhileAPolylineReadAsTrianglesDoes()
    {
        // Two real triangles as one 6-index window: no degenerate triples.
        Vector3[] realTriangles =
        [
            new(0f, 0f, 0f), new(2f, 0f, 0f), new(0f, 0f, 2f),
            new(5f, 0f, 0f), new(7f, 0f, 0f), new(5f, 0f, 2f)
        ];
        Pm4ConnectiveGeometryReport triangles = AnalyzeWindows(realTriangles, [6], floorNormals: true);

        Assert.Equal(2, triangles.Topology.TriplesTested);
        Assert.Equal(0, triangles.Topology.DegenerateTriples);
        Assert.Equal(1, triangles.Topology.MultipleOfThreeWindows);

        // A straight polyline read as triangles produces degenerate triples for every triple.
        Vector3[] straightRun =
        [
            new(0f, 0f, 0f), new(1f, 0f, 0f), new(2f, 0f, 0f),
            new(3f, 0f, 0f), new(4f, 0f, 0f), new(5f, 0f, 0f)
        ];
        Pm4ConnectiveGeometryReport polyline = AnalyzeWindows(straightRun, [6], floorNormals: true);

        Assert.Equal(2, polyline.Topology.TriplesTested);
        Assert.Equal(2, polyline.Topology.DegenerateTriples);

        // The discriminator separates them. That is the point of this test.
        Assert.NotEqual(triangles.Topology.DegenerateTripleFraction, polyline.Topology.DegenerateTripleFraction);
    }

    [Fact]
    public void NegativeFirstIndexEntries_AreCountedNotDropped()
    {
        Pm4MslkEntry pathEntry = Link(mspiFirstIndex: 0, mspiIndexCount: 4);
        Pm4MslkEntry placementEntry = Link(mspiFirstIndex: -1, mspiIndexCount: 0);

        Pm4ResearchDocument document = Document(
            [new(0f, 0f, 0f), new(1f, 0f, 0f), new(1f, 0f, 1f), new(0f, 0f, 1f)],
            [0u, 1u, 2u, 3u],
            [pathEntry, placementEntry],
            floorNormals: true);

        Pm4ConnectiveGeometryReport report = Pm4ConnectiveGeometryAnalyzer.Analyze("synthetic", [document]);

        Assert.Equal(2, report.WindowPopulation.TotalMslkEntries);
        Assert.Equal(1, report.WindowPopulation.ActiveWindows);
        Assert.Equal(1, report.WindowPopulation.NegativeFirstIndexEntries);
    }

    [Fact]
    public void MscnLinkage_SeparatesReferencedFromUnreferencedPoints()
    {
        Pm4KnownChunkSet chunks = new(
            Mshd: null,
            Mslk: [],
            Mspv: [],
            Mspi: [],
            Msvt: [new(0f, 0f, 0f)],
            Msvi: [],
            // Two surfaces both pointing at MSCN[0]; MSCN[1] and MSCN[2] go unreferenced.
            Msur: [Surface(mscnRef: 0), Surface(mscnRef: 0)],
            Mscn: [new(0f, 0f, 0f), new(1f, 1f, 1f), new(2f, 2f, 2f)],
            Mprl: [],
            Mprr: [],
            Mdbh: null,
            Mdbi: [],
            Mdbf: [],
            Mdos: [],
            Mdsf: []);

        Pm4ConnectiveGeometryReport report = Pm4ConnectiveGeometryAnalyzer.Analyze(
            "synthetic",
            [new Pm4ResearchDocument("synthetic.pm4", 12304, [], chunks, [])]);

        Assert.Equal(2, report.MscnLinkage.MsurToMscnFits);
        Assert.Equal(0, report.MscnLinkage.MsurToMscnMisses);
        Assert.Equal(3, report.MscnLinkage.TotalMscnPoints);
        Assert.Equal(1, report.MscnLinkage.DistinctMscnReferenced);
        Assert.Equal(2, report.MscnLinkage.MscnPointsUnreferenced);
    }

    private static Pm4ConnectiveGeometryReport AnalyzeWindows(Vector3[] vertices, int[] windowSizes, bool floorNormals)
        => Analyze(vertices, Enumerable.Range(0, vertices.Length).Select(static i => (uint)i).ToArray(), windowSizes, floorNormals);

    private static Pm4ConnectiveGeometryReport Analyze(Vector3[] vertices, uint[] indices, int[] windowSizes, bool floorNormals)
    {
        List<Pm4MslkEntry> links = [];
        int cursor = 0;
        foreach (int size in windowSizes)
        {
            links.Add(Link(cursor, (byte)size));
            cursor += size;
        }

        return Pm4ConnectiveGeometryAnalyzer.Analyze("synthetic", [Document(vertices, indices, links, floorNormals)]);
    }

    private static Pm4ResearchDocument Document(
        IReadOnlyList<Vector3> mspv,
        IReadOnlyList<uint> mspi,
        IReadOnlyList<Pm4MslkEntry> mslk,
        bool floorNormals)
    {
        // One Z-up surface so the analyzer has a reference axis to measure the windows against.
        IReadOnlyList<Pm4MsurEntry> msur = floorNormals ? [Surface(mscnRef: 0)] : [];

        Pm4KnownChunkSet chunks = new(
            Mshd: null,
            Mslk: mslk,
            Mspv: mspv,
            Mspi: mspi,
            Msvt: [],
            Msvi: [],
            Msur: msur,
            Mscn: [],
            Mprl: [],
            Mprr: [],
            Mdbh: null,
            Mdbi: [],
            Mdbf: [],
            Mdos: [],
            Mdsf: []);

        return new Pm4ResearchDocument("synthetic.pm4", 12304, [], chunks, []);
    }

    private static Pm4MslkEntry Link(int mspiFirstIndex, byte mspiIndexCount)
        => new(
            TypeFlags: 0x12,
            Subtype: 0,
            Padding: 0,
            GroupObjectId: 1,
            MspiFirstIndex: mspiFirstIndex,
            MspiIndexCount: mspiIndexCount,
            LinkId: 0,
            RefIndex: 0,
            SystemFlag: 0x8000);

    private static Pm4MsurEntry Surface(uint mscnRef)
        => new(
            GroupKey: 0,
            IndexCount: 3,
            AttributeMask: 0,
            Padding: 0,
            Normal: new Vector3(0f, 0f, 1f),
            Height: 0f,
            MsviFirstIndex: 0,
            _0x18: mscnRef,
            PackedParams: 0);
}
