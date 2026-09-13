using WowViewer.Core.Files;
using System.Buffers.Binary;
using System.Numerics;
using SixLabors.ImageSharp;
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
        Assert.Equal(0, embeddedGroupSummary.TotalLightRefCount);
        Assert.Equal(583, embeddedGroupSummary.TotalBspNodeCount);
        Assert.Equal(6716, embeddedGroupSummary.TotalBspFaceRefCount);

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

    [Fact]
    public void Read_Castle01AlphaPerAssetMpq_EmbeddedGroupsProduceExpectedOptionalChunkSignals()
    {
        string mpqPath = WmoTestPaths.Castle01AlphaMpqPath;
        if (!File.Exists(mpqPath))
            return;

        byte[]? bytes = AlphaArchiveReader.ReadWithMpqFallback(mpqPath);
        Assert.NotNull(bytes);

        using MemoryStream rootStream = new(bytes);
        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(rootStream);
        List<ChunkSpan> groupChunks = chunks.Where(static chunk => chunk.Header.Id == WmoChunkIds.Mogp).ToList();
        Assert.Equal(2, groupChunks.Count);

        int totalLightRefs = 0;
        int totalBspNodes = 0;
        int totalBspFaceRefs = 0;

        foreach (ChunkSpan groupChunk in groupChunks)
        {
            byte[] groupPayload = ReadPayload(rootStream, groupChunk);
            byte[] groupFileBytes = CreateGroupFile(14, groupPayload);

            using MemoryStream groupSummaryStream = new(groupFileBytes);
            WmoGroupSummary groupSummary = WmoGroupSummaryReader.Read(groupSummaryStream, $"{mpqPath}#MOGP@{groupChunk.HeaderOffset}");

            totalLightRefs += groupSummary.LightRefCount;
            totalBspNodes += groupSummary.BspNodeCount;
            totalBspFaceRefs += groupSummary.BspFaceRefCount;

            if (groupSummary.LightRefCount > 0)
            {
                using MemoryStream groupLightStream = new(groupFileBytes);
                WmoGroupLightRefSummary lightSummary = WmoGroupLightRefSummaryReader.Read(groupLightStream, $"{mpqPath}#MOGP@{groupChunk.HeaderOffset}");
                Assert.Equal(groupSummary.LightRefCount, lightSummary.RefCount);
            }

            if (groupSummary.BspNodeCount > 0)
            {
                using MemoryStream groupBspNodeStream = new(groupFileBytes);
                WmoGroupBspNodeSummary bspNodeSummary = WmoGroupBspNodeSummaryReader.Read(groupBspNodeStream, $"{mpqPath}#MOGP@{groupChunk.HeaderOffset}");
                Assert.Equal(groupSummary.BspNodeCount, bspNodeSummary.NodeCount);

                using MemoryStream groupBspFaceRangeStream = new(groupFileBytes);
                WmoGroupBspFaceRangeSummary bspFaceRangeSummary = WmoGroupBspFaceRangeSummaryReader.Read(groupBspFaceRangeStream, $"{mpqPath}#MOGP@{groupChunk.HeaderOffset}");
                Assert.Equal(groupSummary.BspNodeCount, bspFaceRangeSummary.NodeCount);
            }

            if (groupSummary.BspFaceRefCount > 0)
            {
                using MemoryStream groupBspFaceStream = new(groupFileBytes);
                WmoGroupBspFaceSummary bspFaceSummary = WmoGroupBspFaceSummaryReader.Read(groupBspFaceStream, $"{mpqPath}#MOGP@{groupChunk.HeaderOffset}");
                Assert.Equal(groupSummary.BspFaceRefCount, bspFaceSummary.RefCount);
            }
        }

        Assert.Equal(0, totalLightRefs);
        Assert.Equal(583, totalBspNodes);
        Assert.Equal(6716, totalBspFaceRefs);
    }

    [Fact]
    public void Read_Castle01AlphaPerAssetMpq_EmbeddedGroupDetailsExposePerGroupSummaries()
    {
        string mpqPath = WmoTestPaths.Castle01AlphaMpqPath;
        if (!File.Exists(mpqPath))
            return;

        byte[] bytes = AlphaArchiveReader.ReadWithMpqFallback(mpqPath)!;

        using MemoryStream aggregateStream = new(bytes);
        WmoEmbeddedGroupSummary embeddedGroupSummary = WmoEmbeddedGroupSummaryReader.Read(aggregateStream, mpqPath);

        using MemoryStream detailStream = new(bytes);
        IReadOnlyList<WmoEmbeddedGroupDetail> details = WmoEmbeddedGroupDetailReader.Read(detailStream, mpqPath);

        Assert.Equal(2, details.Count);
        Assert.All(details, static detail => Assert.NotNull(detail.FaceMaterialSummary));
        Assert.All(details, static detail => Assert.NotNull(detail.IndexSummary));
        Assert.All(details, static detail => Assert.NotNull(detail.VertexSummary));
        Assert.All(details, static detail => Assert.NotNull(detail.NormalSummary));
        Assert.All(details, static detail => Assert.NotNull(detail.BatchSummary));
        Assert.All(details, static detail => Assert.NotNull(detail.BspNodeSummary));
        Assert.All(details, static detail => Assert.NotNull(detail.BspFaceSummary));
        Assert.All(details, static detail => Assert.NotNull(detail.BspFaceRangeSummary));
        Assert.All(details, static detail => Assert.Null(detail.LightRefSummary));
        Assert.All(details, static detail => Assert.Equal(detail.GroupSummary.DoodadRefCount > 0, detail.DoodadRefSummary is not null));
        Assert.All(details, static detail => Assert.Equal(detail.GroupSummary.PrimaryUvCount > 0, detail.UvSummary is not null));
        Assert.All(details, static detail => Assert.Equal(detail.GroupSummary.VertexColorCount > 0, detail.VertexColorSummary is not null));
        Assert.All(details, static detail => Assert.Equal(detail.GroupSummary.HasLiquid, detail.LiquidSummary is not null));
        Assert.Equal(embeddedGroupSummary.TotalFaceMaterialCount, details.Sum(static detail => detail.FaceMaterialSummary!.FaceCount));
        Assert.Equal(embeddedGroupSummary.TotalVertexCount, details.Sum(static detail => detail.VertexSummary!.VertexCount));
        Assert.Equal(embeddedGroupSummary.TotalIndexCount, details.Sum(static detail => detail.IndexSummary!.IndexCount));
        Assert.Equal(embeddedGroupSummary.TotalNormalCount, details.Sum(static detail => detail.NormalSummary!.NormalCount));
        Assert.Equal(embeddedGroupSummary.TotalBatchCount, details.Sum(static detail => detail.BatchSummary!.EntryCount));
        Assert.Equal(embeddedGroupSummary.TotalDoodadRefCount, details.Sum(static detail => detail.DoodadRefSummary?.RefCount ?? 0));
        Assert.Equal(583, details.Sum(static detail => detail.BspNodeSummary!.NodeCount));
        Assert.Equal(6716, details.Sum(static detail => detail.BspFaceSummary!.RefCount));
    }

    [Fact]
    public void Read_IronforgeAlphaPerAssetMpq_EmbeddedGroupDetailsExposePositiveLightAndLiquidSignals()
    {
        string mpqPath = WmoTestPaths.IronforgeAlphaMpqPath;
        if (!File.Exists(mpqPath))
            return;

        byte[] bytes = AlphaArchiveReader.ReadWithMpqFallback(mpqPath)!;

        using MemoryStream detailStream = new(bytes);
        IReadOnlyList<WmoEmbeddedGroupDetail> details = WmoEmbeddedGroupDetailReader.Read(detailStream, mpqPath);

        Assert.NotEmpty(details);
        Assert.Contains(details, static detail => detail.LightRefSummary is not null && detail.LightRefSummary.RefCount > 0);
        Assert.Contains(details, static detail => detail.LiquidSummary is not null && detail.LiquidSummary.HeightCount > 0);
        Assert.True(details.Sum(static detail => detail.LightRefSummary?.RefCount ?? 0) > 0);
        Assert.True(details.Count(static detail => detail.LiquidSummary is not null) > 0);
    }

    [Fact]
    public void Read_IronforgeAlphaPerAssetMpq_ProducesExpectedRootLightSummary()
    {
        string mpqPath = WmoTestPaths.IronforgeAlphaMpqPath;
        if (!File.Exists(mpqPath))
            return;

        byte[] bytes = AlphaArchiveReader.ReadWithMpqFallback(mpqPath)!;

        using MemoryStream summaryStream = new(bytes);
        WmoSummary summary = WmoSummaryReader.Read(summaryStream, mpqPath);

        using MemoryStream lightStream = new(bytes);
        WmoLightSummary lightSummary = WmoLightSummaryReader.Read(lightStream, mpqPath);

        Assert.Equal(summary.ReportedLightCount, lightSummary.EntryCount);
        Assert.Equal(6976, lightSummary.PayloadSizeBytes);
        Assert.Equal(218, lightSummary.EntryCount);
        Assert.True(lightSummary.DistinctTypeCount > 0);
        Assert.True(lightSummary.MaxAttenStart > 0f);
        Assert.True(lightSummary.MaxAttenEnd > 0f);
        Assert.True(lightSummary.MaxAttenEnd >= lightSummary.MaxAttenStart);
    }

    [Fact]
    public void Read_IronforgeAlphaPerAssetMpq_RootLightDetails_UseLegacyLayout()
    {
        string mpqPath = WmoTestPaths.IronforgeAlphaMpqPath;
        if (!File.Exists(mpqPath))
            return;

        byte[] bytes = AlphaArchiveReader.ReadWithMpqFallback(mpqPath)!;

        using MemoryStream lightStream = new(bytes);
        IReadOnlyList<WmoLightDetail> details = WmoLightDetailReader.Read(lightStream, mpqPath);

        Assert.Equal(218, details.Count);
        Assert.All(details, static detail => Assert.Equal(32, detail.EntrySizeBytes));
        Assert.All(details, static detail => Assert.Null(detail.HeaderFlagsWord));
        Assert.All(details, static detail => Assert.Null(detail.Rotation));
        Assert.All(details, static detail => Assert.Null(detail.RotationLength));
        Assert.Contains(details, static detail => detail.AttenStart > 0f);
        Assert.Contains(details, static detail => detail.AttenEnd > detail.AttenStart);
    }

    [Fact]
    public void Read_StagedStormwindAlphaPerAssetMpq_ThroughCatalog_ProducesChunkedWmo()
    {
        string mpqPath = WmoTestPaths.StagedStormwindAlphaMpqPath;
        if (!File.Exists(mpqPath))
            return;

        using MpqArchiveCatalog catalog = new();
        byte[]? bytes = catalog.ReadFile0FromPath(
            mpqPath,
            "Stormwind.wmo",
            "World\\wmo\\Azeroth\\Buildings\\Stormwind\\Stormwind.wmo");

        Assert.NotNull(bytes);
        string head = Convert.ToHexString(bytes.AsSpan(0, Math.Min(16, bytes.Length)));

        using MemoryStream chunkStream = new(bytes!);
        IReadOnlyList<ChunkSpan> chunks;
        try
        {
            chunks = ChunkedFileReader.ReadTopLevelChunks(chunkStream, padOddChunkSizes: false);
        }
        catch (InvalidDataException exception)
        {
            throw new Xunit.Sdk.XunitException($"{exception.Message}; len={bytes.Length}; head={head}");
        }

        Assert.NotEmpty(chunks);

        WowFileDetection detection = WowFileDetector.Detect(chunkStream, mpqPath);
        Assert.True(
            detection.Kind == WowFileKind.Wmo,
            $"Detected {detection.Kind}; firstBytes={head}");
    }

    [Fact]
    public void Read_StagedStormwindAlphaMpqVirtualPath_ThroughArchiveVirtualFileReader_ProducesChunkedWmo()
    {
        if (!Directory.Exists(WmoTestPaths.StagedStormwindAlphaClientRoot))
            return;

        byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(
            "World\\wmo\\Azeroth\\Buildings\\Stormwind\\Stormwind.wmo.MPQ",
            [WmoTestPaths.StagedStormwindAlphaClientRoot],
            (ArchiveCatalogBootstrapOptions?)null);

        using MemoryStream chunkStream = new(bytes);
        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(chunkStream, padOddChunkSizes: false);
        Assert.NotEmpty(chunks);

        WowFileDetection detection = WowFileDetector.Detect(chunkStream, "World\\wmo\\Azeroth\\Buildings\\Stormwind\\Stormwind.wmo.MPQ");
        Assert.Equal(WowFileKind.Wmo, detection.Kind);
    }

    [Fact]
    public void Read_IronforgeStandard060_RootLightSummary_UsesStandardTailAttenuationOffsets()
    {
        if (!Directory.Exists(WmoTestPaths.Standard060DataPath) || !File.Exists(WmoTestPaths.ListfilePath))
            return;

        byte[]? bytes = ReadStandardArchiveWmo(WmoTestPaths.Standard060IronforgeRootPath);
        Assert.NotNull(bytes);

        using MemoryStream chunkStream = new(bytes);
        IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(chunkStream);
        ChunkSpan mver = Assert.Single(chunks, static chunk => chunk.Header.Id == WmoChunkIds.Mver);
        ChunkSpan molt = Assert.Single(chunks, static chunk => chunk.Header.Id == WmoChunkIds.Molt);
        uint? version = ChunkedFileReader.TryReadUInt32(chunkStream, mver);
        Assert.DoesNotContain(chunks, static chunk => chunk.Header.Id == WmoChunkIds.Momo);
        byte[] payload = ReadPayload(chunkStream, molt);

        Assert.NotNull(version);
        Assert.True(version >= 16);
        Assert.NotEmpty(payload);
        Assert.Equal(0, payload.Length % 48);

        Assert.Equal("[-0.000, 0.000] nonZero=0", DescribeFloatRange(payload, 48, 24));
        Assert.Equal("[-0.000, 0.000] nonZero=0", DescribeFloatRange(payload, 48, 28));
        Assert.Equal("[-1.000, -1.000] nonZero=218", DescribeFloatRange(payload, 48, 32));
        Assert.Equal("[-0.500, -0.500] nonZero=218", DescribeFloatRange(payload, 48, 36));
        Assert.Equal("[1.306, 8.333] nonZero=218", DescribeFloatRange(payload, 48, 40));
        Assert.Equal("[9.167, 29.611] nonZero=218", DescribeFloatRange(payload, 48, 44));
        Assert.Equal("[0x0101, 0x0101] nonZero=218 distinct=1", DescribeUInt16Range(payload, 48, 2));

        using MemoryStream lightStream = new(bytes);
        WmoLightSummary summary = WmoLightSummaryReader.Read(lightStream, WmoTestPaths.Standard060IronforgeRootPath);

        Assert.Equal(payload.Length, summary.PayloadSizeBytes);
        Assert.Equal(payload.Length / 48, summary.EntryCount);
        Assert.Equal(1.306f, summary.MinAttenStart, 3);
        Assert.Equal(8.333f, summary.MaxAttenStart, 3);
        Assert.Equal(29.611f, summary.MaxAttenEnd, 3);
        Assert.Equal(payload.Length / 48, summary.NonZeroHeaderFlagsWordCount);
        Assert.Equal(1, summary.DistinctHeaderFlagsWordCount);
        Assert.Equal((ushort)0x0101, summary.MinHeaderFlagsWord);
        Assert.Equal((ushort)0x0101, summary.MaxHeaderFlagsWord);
        Assert.Equal(payload.Length / 48, summary.RotationEntryCount);
        Assert.Equal(payload.Length / 48, summary.NonIdentityRotationCount);
        Assert.Equal(1.118f, summary.MinRotationLength, 3);
        Assert.Equal(1.118f, summary.MaxRotationLength, 3);
    }

    [Fact]
    public void Read_IronforgeStandard060_RootLightDetails_ExposeRawStandardLayoutFields()
    {
        if (!Directory.Exists(WmoTestPaths.Standard060DataPath) || !File.Exists(WmoTestPaths.ListfilePath))
            return;

        byte[]? bytes = ReadStandardArchiveWmo(WmoTestPaths.Standard060IronforgeRootPath);
        Assert.NotNull(bytes);

        using MemoryStream lightStream = new(bytes);
        IReadOnlyList<WmoLightDetail> details = WmoLightDetailReader.Read(lightStream, WmoTestPaths.Standard060IronforgeRootPath);

        Assert.NotEmpty(details);
        Assert.All(details, static detail => Assert.Equal(48, detail.EntrySizeBytes));
        Assert.All(details, static detail => Assert.Equal((ushort)0x0101, detail.HeaderFlagsWord));
        Assert.All(details, static detail => Assert.True(detail.Rotation.HasValue));
        Assert.All(details, static detail => Assert.True(detail.RotationLength.HasValue));
        Assert.All(details, static detail => Assert.Equal(1.118f, detail.RotationLength!.Value, 3));
        Assert.Contains(details, static detail => detail.AttenStart >= 1.306f && detail.AttenEnd <= 29.611f);
    }

    private static byte[]? ReadStandardArchiveWmo(string virtualPath)
    {
        using MpqArchiveCatalog catalog = new();
        catalog.LoadArchives([WmoTestPaths.Standard060DataPath]);
        catalog.LoadListfile(WmoTestPaths.ListfilePath);
        return catalog.ReadFile(virtualPath);
    }

    private static string DescribeFloatRange(byte[] payload, int entrySize, int floatOffset)
    {
        float min = float.MaxValue;
        float max = float.MinValue;
        int nonZeroCount = 0;

        for (int offset = floatOffset; offset < payload.Length; offset += entrySize)
        {
            float value = BitConverter.ToSingle(payload, offset);
            min = Math.Min(min, value);
            max = Math.Max(max, value);
            if (value != 0f)
                nonZeroCount++;
        }

        return $"[{min:F3}, {max:F3}] nonZero={nonZeroCount}";
    }

    private static string DescribeUInt16Range(byte[] payload, int entrySize, int wordOffset)
    {
        ushort min = ushort.MaxValue;
        ushort max = ushort.MinValue;
        int nonZeroCount = 0;
        HashSet<ushort> distinct = [];

        for (int offset = wordOffset; offset < payload.Length; offset += entrySize)
        {
            ushort value = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset, sizeof(ushort)));
            min = Math.Min(min, value);
            max = Math.Max(max, value);
            if (value != 0)
                nonZeroCount++;

            distinct.Add(value);
        }

        return $"[0x{min:X4}, 0x{max:X4}] nonZero={nonZeroCount} distinct={distinct.Count}";
    }

    private static byte[] ReadPayload(Stream stream, ChunkSpan chunk)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = chunk.DataOffset;
            byte[] payload = new byte[chunk.Header.Size];
            stream.ReadExactly(payload);
            return payload;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static byte[] CreateGroupFile(uint version, byte[] mogpPayload)
    {
        return
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(version)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", mogpPayload),
        ];
    }

    [Fact]
    public void Read_NagrandFloatingRock_TBC_Inspect()
    {
        string clientRoot = @"H:\CLIENTS\TBC\2.X_Retail_Windows_enUS_2.4.3.8606\World of Warcraft";
        if (!Directory.Exists(clientRoot))
            return;

        using MpqArchiveCatalog catalog = new();
        var bootstrap = ArchiveCatalogBootstrapper.Bootstrap(catalog, [clientRoot], WmoTestPaths.ListfilePath);

        var matches = bootstrap.InternalFiles.Where(f => f.Contains("NAGRAND", StringComparison.OrdinalIgnoreCase) && f.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase)).Take(20).ToList();
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"InternalFiles total={bootstrap.InternalFiles.Count}, KnownFiles={bootstrap.KnownFiles.Count}");
        sb.AppendLine($"Matches with NAGRAND & .wmo: {matches.Count}");
        foreach (var m in matches)
            sb.AppendLine("  " + m);

        string rootPath = @"World\wmo\Outland\FloatingRocks\Nagrand_RockFloating_01.wmo";
        byte[]? rootBytes = catalog.ReadFile(rootPath);
        sb.AppendLine($"Root '{rootPath}': bytes={(rootBytes != null ? rootBytes.Length : -1)}");

        string[] texNames = [
            @"WORLD\ENVIRONMENT\DOODAD\NAGRAND\ROCKSFLOATING\NAGRAND_ROCKFLOATING_BOTTOM.BLP",
            @"TILESET\EXPANSION01\NAGRAND\NAGRANDROCKSOLID.BLP",
            @"WORLD\ENVIRONMENT\DOODAD\NAGRAND\ROCKSFLOATING\NAGRAND_ROCKFLOATING_EDGE.BLP",
            @"WORLD\ENVIRONMENT\DOODAD\NAGRAND\ROCKSFLOATING\NAGRAND_ROCKFLOATING_GRASSCENTER.BLP"
        ];
        string[] outNames = ["bottom.png", "rocksolid.png", "edge.png", "grasscenter.png"];
        for (int ti = 0; ti < texNames.Length; ti++)
        {
            byte[]? b = catalog.ReadFile(texNames[ti]);
            if (b != null)
            {
                using var ms = new MemoryStream(b);
                using var blp = new SereniaBLPLib.BlpFile(ms);
                using var img = blp.GetImage(0);
                img.SaveAsPng($@"C:\Users\akspa\.gemini\antigravity-ide\brain\d464bb6e-1bdb-482e-b49e-5441dbb8d94d\scratch\{outNames[ti]}");
                
                // Inspect row 0 (y=0) and row H-1 (y=H-1)
                float r0R = 0, r0G = 0, r0B = 0;
                float r1R = 0, r1G = 0, r1B = 0;
                for (int x = 0; x < img.Width; x++)
                {
                    var c0 = img[x, 0];
                    r0R += c0.R; r0G += c0.G; r0B += c0.B;
                    var c1 = img[x, img.Height - 1];
                    r1R += c1.R; r1G += c1.G; r1B += c1.B;
                }
                sb.AppendLine($"Saved {outNames[ti]}: {img.Width}x{img.Height}");
                sb.AppendLine($"  y=0: avg RGB=({r0R/img.Width:F1}, {r0G/img.Width:F1}, {r0B/img.Width:F1})");
                sb.AppendLine($"  y={img.Height-1}: avg RGB=({r1R/img.Width:F1}, {r1G/img.Width:F1}, {r1B/img.Width:F1})");
            }
        }

        if (rootBytes != null)
        {
            var groups = new List<byte[]>();
            for (int gi = 0; ; gi++)
            {
                string gPath = $@"World\wmo\Outland\FloatingRocks\Nagrand_RockFloating_01_{gi:D3}.wmo";
                byte[]? gBytes = catalog.ReadFile(gPath);
                if (gBytes == null) break;
                groups.Add(gBytes);
            }

            var v17Parser = new WowViewer.Core.IO.Converters.WmoV17ToV14Converter();
            var wmo = v17Parser.ParseV17ToModel(rootBytes, groups);

            sb.AppendLine($"WMO: Groups={wmo.Groups.Count}, Materials={wmo.Materials.Count}");
            for (int i = 0; i < wmo.Materials.Count; i++)
            {
                var m = wmo.Materials[i];
                sb.AppendLine($"Mat {i}: Shader={m.Shader}, Blend={m.BlendMode}, Flags=0x{m.Flags:X}, Tex1='{m.Texture1Name}', Tex2='{m.Texture2Name}', Tex3='{m.Texture3Name}'");
            }
            for (int gi = 0; gi < wmo.Groups.Count; gi++)
            {
                var g = wmo.Groups[gi];
                sb.AppendLine($"Group {gi} '{g.Name}': Flags=0x{g.Flags:X}, Verts={g.Vertices.Count}, UVs={g.UVs.Count}, Colors={g.VertexColors.Count}, Batches={g.Batches.Count}");
                for (int bi = 0; bi < g.Batches.Count; bi++)
                {
                    var b = g.Batches[bi];
                    sb.AppendLine($"  Batch {bi}: FirstIdx={b.FirstIndex}, Count={b.IndexCount}, MatId={b.MaterialId}");
                    float minU = float.MaxValue, maxU = float.MinValue, minV = float.MaxValue, maxV = float.MinValue;
                    for (int ii = (int)b.FirstIndex; ii < (int)b.FirstIndex + (int)b.IndexCount && ii < g.Indices.Count; ii++)
                    {
                        int vi = g.Indices[ii];
                        if (vi < g.UVs.Count)
                        {
                            var uv = g.UVs[vi];
                            minU = Math.Min(minU, uv.X); maxU = Math.Max(maxU, uv.X);
                            minV = Math.Min(minV, uv.Y); maxV = Math.Max(maxV, uv.Y);
                        }
                    }
                    sb.AppendLine($"    Batch {bi} UV range: U=[{minU:F4}, {maxU:F4}], V=[{minV:F4}, {maxV:F4}]");

                    var batchVerts = new HashSet<int>();
                    for (int ii = (int)b.FirstIndex; ii < (int)b.FirstIndex + (int)b.IndexCount; ii++)
                        batchVerts.Add(g.Indices[ii]);

                    var lowZ = batchVerts.OrderBy(vi => g.Vertices[vi].Z).Take(2);
                    var highZ = batchVerts.OrderByDescending(vi => g.Vertices[vi].Z).Take(2);
                    foreach (int vi in lowZ)
                        sb.AppendLine($"      [min Z] v{vi}: Z={g.Vertices[vi].Z:F3}, uv=({g.UVs[vi].X:F4}, {g.UVs[vi].Y:F4})");
                    foreach (int vi in highZ)
                        sb.AppendLine($"      [max Z] v{vi}: Z={g.Vertices[vi].Z:F3}, uv=({g.UVs[vi].X:F4}, {g.UVs[vi].Y:F4})");

                    if (bi == 2)
                    {
                        var b1Verts = new HashSet<int>();
                        for (int ii = (int)g.Batches[1].FirstIndex; ii < (int)g.Batches[1].FirstIndex + (int)g.Batches[1].IndexCount; ii++)
                            b1Verts.Add(g.Indices[ii]);

                        var b2Verts = new HashSet<int>();
                        for (int ii = (int)b.FirstIndex; ii < (int)b.FirstIndex + (int)b.IndexCount; ii++)
                            b2Verts.Add(g.Indices[ii]);

                        var b3Verts = new HashSet<int>();
                        for (int ii = (int)g.Batches[3].FirstIndex; ii < (int)g.Batches[3].FirstIndex + (int)g.Batches[3].IndexCount; ii++)
                            b3Verts.Add(g.Indices[ii]);

                        var v132Pos = g.Vertices[132];
                        float bestDist = float.MaxValue;
                        int bestV = -1;
                        foreach (int vi in b1Verts)
                        {
                            float d = Vector3.Distance(g.Vertices[vi], v132Pos);
                            if (d < bestDist)
                            {
                                bestDist = d;
                                bestV = vi;
                            }
                        }
                        sb.AppendLine($"    v132 pos={v132Pos}, uv=({g.UVs[132].X:F4}, {g.UVs[132].Y:F4})");
                        if (bestV >= 0)
                        {
                            var bestPos = g.Vertices[bestV];
                            var bestUv = g.UVs[bestV];
                            sb.AppendLine($"    Closest Batch 1 vertex to v132 is v{bestV}: dist={bestDist:F5}, pos={bestPos}, uv=({bestUv.X:F4}, {bestUv.Y:F4})");
                        }

                        sb.AppendLine("    Batch 2 samples (Z vs UV):");
                        foreach (int vi in b2Verts.OrderBy(vi => g.Vertices[vi].Z).Take(5))
                        {
                            var pos = g.Vertices[vi];
                            var uv = g.UVs[vi];
                            sb.AppendLine($"      bottom v{vi}: Z={pos.Z:F3}, U={uv.X:F4}, V={uv.Y:F4}");
                        }
                        foreach (int vi in b2Verts.OrderByDescending(vi => g.Vertices[vi].Z).Take(5))
                        {
                            var pos = g.Vertices[vi];
                            var uv = g.UVs[vi];
                            sb.AppendLine($"      top v{vi}: Z={pos.Z:F3}, U={uv.X:F4}, V={uv.Y:F4}");
                        }
                    }

                    if (bi == 3)
                    {
                        sb.AppendLine("    Batch 3 samples (X, Y, Z vs UV):");
                        var b3Verts = new HashSet<int>();
                        for (int ii = (int)b.FirstIndex; ii < (int)b.FirstIndex + (int)b.IndexCount; ii++)
                            b3Verts.Add(g.Indices[ii]);

                        foreach (int vi in b3Verts.Take(10))
                        {
                            var pos = g.Vertices[vi];
                            var uv = g.UVs[vi];
                            sb.AppendLine($"      v{vi}: pos=({pos.X:F2}, {pos.Y:F2}, {pos.Z:F2}), uv=({uv.X:F4}, {uv.Y:F4})");
                        }
                    }
                }
            }
        }

        File.WriteAllText(@"C:\Users\akspa\.gemini\antigravity-ide\brain\d464bb6e-1bdb-482e-b49e-5441dbb8d94d\scratch\wmo_inspect.txt", sb.ToString());
    }
}

internal static class WmoTestPaths
{
    public static string Castle01AlphaMpqPath => Path.Combine(GetWowViewerRoot(), "testdata", "0.5.3", "tree", "World", "wmo", "Azeroth", "Buildings", "Castle", "castle01.wmo.MPQ");

    public static string IronforgeAlphaMpqPath => Path.Combine(GetWowViewerRoot(), "testdata", "0.5.3", "tree", "World", "wmo", "KhazModan", "Cities", "Ironforge", "ironforge.wmo.MPQ");

    public static string StagedStormwindAlphaMpqPath => Path.Combine(GetWowViewerRoot(), "..", "output", "tmp", "wowarchive-clients", "0_5_3_3368", "World of Warcraft", "Data", "World", "wmo", "Azeroth", "Buildings", "Stormwind", "Stormwind.wmo.MPQ");

    public static string StagedStormwindAlphaClientRoot => Path.Combine(GetWowViewerRoot(), "..", "output", "tmp", "wowarchive-clients", "0_5_3_3368", "World of Warcraft");

    public static string Standard060DataPath => Path.Combine(GetWowViewerRoot(), "testdata", "0.6.0", "World of Warcraft", "Data");

    public static string ListfilePath => Path.Combine(GetWowViewerRoot(), "libs", "wowdev", "wow-listfile", "listfile.txt");

    public static string Standard060IronforgeRootPath => "world/wmo/khazmodan/cities/ironforge/ironforge.wmo";

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
