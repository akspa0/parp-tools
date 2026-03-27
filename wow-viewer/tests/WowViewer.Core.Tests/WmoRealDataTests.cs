using WowViewer.Core.Files;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoRealDataTests
{
    [Fact]
    public void Read_Castle01AlphaPerAssetMpq_ProducesExpectedRootSignals()
    {
        string mpqPath = WmoTestPaths.Castle01AlphaMpqPath;
        if (!File.Exists(mpqPath))
            return;

        byte[]? bytes = AlphaArchiveReader.ReadWithMpqFallback(mpqPath);
        Assert.NotNull(bytes);

        using MemoryStream chunkStream = new(bytes);
        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(chunkStream);
        Assert.NotEmpty(chunks);

        WowFileDetection detection = WowFileDetector.Detect(chunkStream, mpqPath);
        Assert.True(
            detection.Kind == WowFileKind.Wmo,
            $"Detected {detection.Kind}; firstBytes={Convert.ToHexString(bytes.AsSpan(0, Math.Min(16, bytes.Length)))}");

        using MemoryStream summaryStream = new(bytes);
        WmoSummary summary = WmoSummaryReader.Read(summaryStream, mpqPath);

        Assert.NotNull(summary.Version);
        Assert.True(summary.MaterialEntryCount > 0);
        Assert.True(summary.GroupInfoCount > 0);
        Assert.True(summary.TextureNameCount > 0);
        using MemoryStream materialStream = new(bytes);
        WmoMaterialSummary materialSummary = WmoMaterialSummaryReader.Read(materialStream, mpqPath);
        Assert.True(materialSummary.EntryCount > 0);

        using MemoryStream groupInfoStream = new(bytes);
        WmoGroupInfoSummary groupInfoSummary = WmoGroupInfoSummaryReader.Read(groupInfoStream, mpqPath);
        Assert.True(groupInfoSummary.EntryCount > 0);

        using MemoryStream embeddedGroupStream = new(bytes);
        WmoEmbeddedGroupSummary embeddedGroupSummary = WmoEmbeddedGroupSummaryReader.Read(embeddedGroupStream, mpqPath);
        Assert.Equal(summary.GroupInfoCount, embeddedGroupSummary.GroupCount);
        Assert.True(embeddedGroupSummary.TotalVertexCount > 0);
        Assert.True(embeddedGroupSummary.TotalIndexCount > 0);

        using MemoryStream embeddedGroupLinkageStream = new(bytes);
        WmoEmbeddedGroupLinkageSummary embeddedGroupLinkageSummary = WmoEmbeddedGroupLinkageSummaryReader.Read(embeddedGroupLinkageStream, mpqPath);
        Assert.Equal(summary.GroupInfoCount, embeddedGroupLinkageSummary.GroupInfoCount);
        Assert.Equal(embeddedGroupSummary.GroupCount, embeddedGroupLinkageSummary.EmbeddedGroupCount);
        Assert.Equal(summary.GroupInfoCount, embeddedGroupLinkageSummary.CoveredPairCount);

        if (summary.ReportedPortalCount > 0)
        {
            using MemoryStream portalVertexStream = new(bytes);
            WmoPortalVertexSummary portalVertexSummary = WmoPortalVertexSummaryReader.Read(portalVertexStream, mpqPath);
            Assert.True(portalVertexSummary.VertexCount > 0);

            using MemoryStream portalInfoStream = new(bytes);
            WmoPortalInfoSummary portalInfoSummary = WmoPortalInfoSummaryReader.Read(portalInfoStream, mpqPath);
            Assert.True(portalInfoSummary.EntryCount > 0);

            using MemoryStream portalRefStream = new(bytes);
            WmoPortalRefSummary portalRefSummary = WmoPortalRefSummaryReader.Read(portalRefStream, mpqPath);
            Assert.True(portalRefSummary.EntryCount > 0);

            using MemoryStream portalVertexRangeStream = new(bytes);
            WmoPortalVertexRangeSummary portalVertexRangeSummary = WmoPortalVertexRangeSummaryReader.Read(portalVertexRangeStream, mpqPath);
            Assert.True(portalVertexRangeSummary.EntryCount > 0);

            using MemoryStream portalRefRangeStream = new(bytes);
            WmoPortalRefRangeSummary portalRefRangeSummary = WmoPortalRefRangeSummaryReader.Read(portalRefRangeStream, mpqPath);
            Assert.True(portalRefRangeSummary.RefCount > 0);

            using MemoryStream portalGroupRangeStream = new(bytes);
            WmoPortalGroupRangeSummary portalGroupRangeSummary = WmoPortalGroupRangeSummaryReader.Read(portalGroupRangeStream, mpqPath);
            Assert.True(portalGroupRangeSummary.RefCount > 0);
        }
    }
}

internal static class WmoTestPaths
{
    public static string Castle01AlphaMpqPath => Path.Combine(GetWowViewerRoot(), "testdata", "0.5.3", "tree", "World", "wmo", "Azeroth", "Buildings", "Castle", "castle01.wmo.MPQ");

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
