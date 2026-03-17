using MdxViewer.DataSources;
using MdxViewer.Export;
using MdxViewer.Terrain;
using WoWMapConverter.Core.Formats.LichKing;

var repoRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

var alphaWdtPath = Path.Combine(repoRoot, "test_data", "0.5.3", "alphawdt", "World", "Maps", "Azeroth", "Azeroth.wdt");
var lkRootPath = Path.Combine(repoRoot, "test_data", "WoWMuseum", "335-dev");
var lkWdtVirtualPath = @"World\Maps\development\development.wdt";

Console.WriteLine("=== Terrain Alpha Edge Audit ===");
Console.WriteLine($"Repo root: {repoRoot}");
Console.WriteLine();

var reports = new List<TileReport>
{
    ScanAlpha(alphaWdtPath),
    ScanLk(lkRootPath, lkWdtVirtualPath, "development", "3.3.5.12340")
};

foreach (var report in reports)
{
    PrintReport(report);
}

var chunkTrace = TraceLkChunk(lkRootPath, lkWdtVirtualPath, "development", "3.3.5.12340", tileX: 0, tileY: 0, chunkX: 6, chunkY: 2);
PrintChunkTrace(chunkTrace);

return;

static ChunkTrace? TraceLkChunk(
    string rootPath,
    string wdtVirtualPath,
    string mapName,
    string buildVersion,
    int tileX,
    int tileY,
    int chunkX,
    int chunkY)
{
    if (!Directory.Exists(rootPath))
    {
        return null;
    }

    using var dataSource = new LooseFileDataSource(rootPath);
    var wdtBytes = dataSource.ReadFile(wdtVirtualPath);
    if (wdtBytes == null || wdtBytes.Length == 0)
    {
        return null;
    }

    var adapter = new StandardTerrainAdapter(wdtBytes, mapName, dataSource, buildVersion);
    var tileResult = adapter.LoadTileWithPlacements(tileX, tileY);
    if (tileResult.Chunks.Count == 0)
    {
        return null;
    }

    var requestedChunk = tileResult.Chunks.FirstOrDefault(c => c.ChunkX == chunkX && c.ChunkY == chunkY);
    var chunk = requestedChunk;

    bool requestedHasUsefulAlpha = requestedChunk != null
        && requestedChunk.Layers.Length > 1
        && requestedChunk.AlphaMaps.Count > 0;

    if (!requestedHasUsefulAlpha)
    {
        chunk = tileResult.Chunks
            .OrderByDescending(c => c.Layers.Length)
            .ThenByDescending(c => c.AlphaMaps.Count)
            .FirstOrDefault(c => c.Layers.Length > 1 && c.AlphaMaps.Count > 0)
            ?? tileResult.Chunks.First();
    }

    var tileTextures = adapter.TileTextures.TryGetValue((tileX, tileY), out var textures)
        ? textures
        : new List<string>();

    string adtPath = $@"World\Maps\{mapName}\{mapName}_{tileY}_{tileX}.adt";
    var adtBytes = dataSource.ReadFile(adtPath);
    Mcnk? mcnk = null;
    if (adtBytes != null && adtBytes.Length > 0)
    {
        mcnk = TryReadMcnkAtMcinIndex(adtBytes, chunk.McinIndex);
    }

    var layerTraces = BuildLayerTraces(chunk, tileTextures, dataSource, mcnk);

    var packed = BuildAlphaShadowArray(new[] { chunk });
    int slice = chunk.ChunkY * 16 + chunk.ChunkX;
    var packedParity = BuildPackedParity(chunk, packed, slice);

    int sampleX = 32;
    int sampleY = 32;
    int sampleOffset = (sampleY * 64 + sampleX) * 4;
    int sliceBase = slice * 64 * 64 * 4;

    var packedSample = new PackedSample(
        sampleX,
        sampleY,
        packed[sliceBase + sampleOffset + 0],
        packed[sliceBase + sampleOffset + 1],
        packed[sliceBase + sampleOffset + 2],
        packed[sliceBase + sampleOffset + 3]);

    byte[]? mcalRaw = mcnk?.McalRawData;
    int mcalLength = mcalRaw?.Length ?? 0;
    string mcalHeadHex = mcalRaw == null
        ? "<none>"
        : HexPreview(mcalRaw, 0, 16);

    return new ChunkTrace(
        TileX: tileX,
        TileY: tileY,
        ChunkX: chunk.ChunkX,
        ChunkY: chunk.ChunkY,
        McinIndex: chunk.McinIndex,
        BuildVersion: buildVersion,
        McalLength: mcalLength,
        McalHeadHex: mcalHeadHex,
        LayerTraces: layerTraces,
        PackedParity: packedParity,
        PackedSample: packedSample);
}

static Mcnk? TryReadMcnkAtMcinIndex(byte[] adtBytes, int mcinIndex)
{
    int mhdrOffset = FindChunk(adtBytes, "MHDR");
    if (mhdrOffset < 0 || mhdrOffset + 8 > adtBytes.Length)
    {
        return null;
    }

    var mhdr = new GillijimProject.WowFiles.Mhdr(adtBytes, mhdrOffset);
    int mhdrStart = mhdrOffset + 8;
    int mcinOffset = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.McinOffset);
    if (mcinOffset == 0)
    {
        return null;
    }

    int mcinAbs = mhdrStart + mcinOffset;
    if (mcinAbs + 8 > adtBytes.Length)
    {
        return null;
    }

    var mcin = new GillijimProject.WowFiles.Mcin(adtBytes, mcinAbs);
    var offsets = mcin.GetMcnkOffsets();
    if ((uint)mcinIndex >= (uint)offsets.Count)
    {
        return null;
    }

    int off = offsets[mcinIndex];
    if (off <= 0 || off + 8 > adtBytes.Length)
    {
        return null;
    }

    if (System.Text.Encoding.ASCII.GetString(adtBytes, off, 4) != "KNCM")
    {
        return null;
    }

    int size = BitConverter.ToInt32(adtBytes, off + 4);
    if (size <= 0 || off + 8 + size > adtBytes.Length)
    {
        return null;
    }

    var mcnkData = new byte[size];
    Array.Copy(adtBytes, off + 8, mcnkData, 0, size);
    return new Mcnk(mcnkData);
}

static int FindChunk(byte[] data, string fourCC)
{
    for (int i = 0; i + 8 <= data.Length;)
    {
        string sig = System.Text.Encoding.ASCII.GetString(data, i, 4);
        int size = BitConverter.ToInt32(data, i + 4);
        if (size < 0)
        {
            break;
        }

        if (sig == Reverse(fourCC))
        {
            return i;
        }

        int next = i + 8 + size + ((size & 1) != 0 ? 1 : 0);
        if (next <= i)
        {
            break;
        }

        i = next;
    }

    return -1;
}

static string Reverse(string value)
{
    var chars = value.ToCharArray();
    Array.Reverse(chars);
    return new string(chars);
}

static List<LayerTrace> BuildLayerTraces(
    TerrainChunkData chunk,
    List<string> tileTextures,
    LooseFileDataSource dataSource,
    Mcnk? mcnk)
{
    var results = new List<LayerTrace>();

    var textureLayers = mcnk?.TextureLayers;
    byte[]? mcalRaw = mcnk?.McalRawData;
    int mcalLength = mcalRaw?.Length ?? 0;

    for (int layer = 0; layer < Math.Min(4, chunk.Layers.Length); layer++)
    {
        var layerInfo = chunk.Layers[layer];

        string textureName = "<missing>";
        bool textureInBounds = layerInfo.TextureIndex >= 0 && layerInfo.TextureIndex < tileTextures.Count;
        bool textureFileExists = false;
        if (textureInBounds)
        {
            textureName = tileTextures[layerInfo.TextureIndex];
            textureFileExists = dataSource.FileExists(textureName) || dataSource.FileExists(textureName.Replace('/', '\\'));
        }

        uint? sourceFlags = null;
        uint? sourceOffset = null;
        int? rawSpan = null;
        string rawHead = "<none>";
        if (textureLayers != null && layer < textureLayers.Count)
        {
            sourceFlags = (uint)textureLayers[layer].Flags;
            sourceOffset = textureLayers[layer].AlphaMapOffset;

            if (layer > 0 && mcalRaw != null && sourceOffset.Value < mcalRaw.Length)
            {
                int start = unchecked((int)sourceOffset.Value);
                int end = mcalLength;
                for (int i = layer + 1; i < textureLayers.Count; i++)
                {
                    int candidate = unchecked((int)textureLayers[i].AlphaMapOffset);
                    if (candidate > start && candidate <= mcalLength)
                    {
                        end = candidate;
                        break;
                    }
                }

                rawSpan = Math.Max(0, end - start);
                rawHead = HexPreview(mcalRaw, start, 16);
            }
        }

        int decodedLength = 0;
        int nonZero = 0;
        int zero = 0;
        bool mostlyZero = false;
        if (layer > 0 && chunk.AlphaMaps.TryGetValue(layer, out var alpha) && alpha != null)
        {
            decodedLength = alpha.Length;
            foreach (byte b in alpha)
            {
                if (b == 0) zero++; else nonZero++;
            }

            mostlyZero = decodedLength > 0 && (nonZero / (double)decodedLength) < 0.05;
        }

        results.Add(new LayerTrace(
            Layer: layer,
            TextureIndex: layerInfo.TextureIndex,
            TextureName: textureName,
            TextureInBounds: textureInBounds,
            TextureFileExists: textureFileExists,
            LayerFlags: layerInfo.Flags,
            LayerAlphaOffset: layerInfo.AlphaOffset,
            SourceFlags: sourceFlags,
            SourceAlphaOffset: sourceOffset,
            RawSpan: rawSpan,
            RawHeadHex: rawHead,
            DecodedLength: decodedLength,
            NonZeroCount: nonZero,
            ZeroCount: zero,
            MostlyZero: mostlyZero));
    }

    return results;
}

static PackedParity BuildPackedParity(TerrainChunkData chunk, byte[] packed, int slice)
{
    int sliceBase = slice * 64 * 64 * 4;
    chunk.AlphaMaps.TryGetValue(1, out var alpha1);
    chunk.AlphaMaps.TryGetValue(2, out var alpha2);
    chunk.AlphaMaps.TryGetValue(3, out var alpha3);

    int alpha1Diff = CountChannelDiff(alpha1, packed, sliceBase, 0);
    int alpha2Diff = CountChannelDiff(alpha2, packed, sliceBase, 1);
    int alpha3Diff = CountChannelDiff(alpha3, packed, sliceBase, 2);
    int shadowDiff = CountChannelDiff(chunk.ShadowMap, packed, sliceBase, 3);

    return new PackedParity(alpha1Diff, alpha2Diff, alpha3Diff, shadowDiff);
}

static byte[] BuildAlphaShadowArray(IReadOnlyList<TerrainChunkData> chunks)
{
    const int size = 64;
    const int slices = 256;
    var data = new byte[size * size * 4 * slices];

    for (int chunkIndex = 0; chunkIndex < chunks.Count; chunkIndex++)
    {
        var chunk = chunks[chunkIndex];
        int slice = chunk.ChunkY * 16 + chunk.ChunkX;
        if ((uint)slice >= 256u)
        {
            slice = chunkIndex & 255;
        }

        int sliceBase = slice * size * size * 4;
        for (int layer = 1; layer <= 3; layer++)
        {
            int channel = layer - 1;
            bool hasLayer = layer < chunk.Layers.Length;
            bool usesAlphaMap = hasLayer && (chunk.Layers[layer].Flags & 0x100u) != 0;

            if (chunk.AlphaMaps.TryGetValue(layer, out var alpha) && alpha != null && alpha.Length >= size * size)
            {
                for (int i = 0; i < size * size; i++)
                {
                    data[sliceBase + i * 4 + channel] = alpha[i];
                }
                continue;
            }

            if (hasLayer && !usesAlphaMap)
            {
                for (int i = 0; i < size * size; i++)
                {
                    data[sliceBase + i * 4 + channel] = 255;
                }
            }
        }

        if (chunk.ShadowMap != null && chunk.ShadowMap.Length >= size * size)
        {
            for (int i = 0; i < size * size; i++)
            {
                data[sliceBase + i * 4 + 3] = chunk.ShadowMap[i];
            }
        }
    }

    return data;
}

static int CountChannelDiff(byte[]? source, byte[] packed, int sliceBase, int channel)
{
    if (source == null || source.Length < 64 * 64)
    {
        return 0;
    }

    int diff = 0;
    for (int i = 0; i < 64 * 64; i++)
    {
        byte packedValue = packed[sliceBase + i * 4 + channel];
        if (packedValue != source[i])
        {
            diff++;
        }
    }

    return diff;
}

static string HexPreview(byte[] data, int offset, int count)
{
    if (offset < 0 || offset >= data.Length)
    {
        return "<out-of-range>";
    }

    int length = Math.Min(count, data.Length - offset);
    var bytes = new byte[length];
    Array.Copy(data, offset, bytes, 0, length);
    return Convert.ToHexString(bytes);
}

static void PrintChunkTrace(ChunkTrace? trace)
{
    Console.WriteLine("=== Chunk Trace (MCAL -> Decode -> Pack -> Renderer Inputs) ===");
    if (trace == null)
    {
        Console.WriteLine("No chunk trace available.");
        Console.WriteLine();
        return;
    }

    Console.WriteLine($"Build: {trace.BuildVersion}");
    Console.WriteLine($"Tile ({trace.TileX},{trace.TileY}) Chunk ({trace.ChunkX},{trace.ChunkY}) MCIN={trace.McinIndex}");
    Console.WriteLine($"Raw MCAL length: {trace.McalLength}");
    Console.WriteLine($"Raw MCAL head: {trace.McalHeadHex}");

    Console.WriteLine("Layers:");
    foreach (var layer in trace.LayerTraces)
    {
        Console.WriteLine(
            $"  L{layer.Layer}: texIdx={layer.TextureIndex} inBounds={layer.TextureInBounds} exists={layer.TextureFileExists} flags=0x{layer.LayerFlags:X} alphaOff={layer.LayerAlphaOffset} srcFlags={(layer.SourceFlags.HasValue ? $"0x{layer.SourceFlags.Value:X}" : "<n/a>")} srcOff={(layer.SourceAlphaOffset?.ToString() ?? "<n/a>")} span={(layer.RawSpan?.ToString() ?? "<n/a>")} decodedLen={layer.DecodedLength} nonZero={layer.NonZeroCount} zero={layer.ZeroCount} mostlyZero={layer.MostlyZero}");
        if (layer.Layer > 0)
        {
            Console.WriteLine($"    rawHead={layer.RawHeadHex}");
            Console.WriteLine($"    texture={layer.TextureName}");
        }
    }

    Console.WriteLine("Packed parity diffs:");
    Console.WriteLine($"  alpha1={trace.PackedParity.Alpha1Diff}, alpha2={trace.PackedParity.Alpha2Diff}, alpha3={trace.PackedParity.Alpha3Diff}, shadow={trace.PackedParity.ShadowDiff}");
    Console.WriteLine($"Packed sample texel ({trace.PackedSample.X},{trace.PackedSample.Y}): a1={trace.PackedSample.Alpha1} a2={trace.PackedSample.Alpha2} a3={trace.PackedSample.Alpha3} shadow={trace.PackedSample.Shadow}");
    Console.WriteLine();
}

static TileReport ScanAlpha(string alphaWdtPath)
{
    if (!File.Exists(alphaWdtPath))
    {
        return TileReport.Missing("alpha-0.5.3", $"Missing WDT: {alphaWdtPath}");
    }

    var adapter = new AlphaTerrainAdapter(alphaWdtPath);
    return ScanTiles(
        "alpha-0.5.3",
        adapter.ExistingTiles,
        tileIndex =>
        {
            int tileX = tileIndex / 64;
            int tileY = tileIndex % 64;
            return (tileX, tileY, adapter.LoadTileWithPlacements(tileX, tileY).Chunks);
        });
}

static TileReport ScanLk(string rootPath, string wdtVirtualPath, string mapName, string buildVersion)
{
    if (!Directory.Exists(rootPath))
    {
        return TileReport.Missing("lk-3.3.5", $"Missing root: {rootPath}");
    }

    using var dataSource = new LooseFileDataSource(rootPath);
    var wdtBytes = dataSource.ReadFile(wdtVirtualPath);
    if (wdtBytes == null || wdtBytes.Length == 0)
    {
        return TileReport.Missing("lk-3.3.5", $"Missing WDT: {wdtVirtualPath}");
    }

    var adapter = new StandardTerrainAdapter(wdtBytes, mapName, dataSource, buildVersion);
    return ScanTiles(
        "lk-3.3.5",
        adapter.ExistingTiles,
        tileIndex =>
        {
            int tileX = tileIndex / 64;
            int tileY = tileIndex % 64;
            return (tileX, tileY, adapter.LoadTileWithPlacements(tileX, tileY).Chunks);
        });
}

static TileReport ScanTiles(
    string dataset,
    IReadOnlyList<int> existingTiles,
    Func<int, (int tileX, int tileY, IReadOnlyList<TerrainChunkData> chunks)> loadTile)
{
    int scannedTiles = 0;
    foreach (var tileIndex in existingTiles.OrderBy(index => index))
    {
        var (tileX, tileY, chunks) = loadTile(tileIndex);
        scannedTiles++;

        if (chunks.Count == 0)
        {
            continue;
        }

        var report = AnalyzeTile(dataset, tileX, tileY, chunks, scannedTiles);
        if (report.HasImpact)
        {
            return report;
        }
    }

    return TileReport.NoImpact(dataset, scannedTiles, existingTiles.Count);
}

static TileReport AnalyzeTile(string dataset, int tileX, int tileY, IReadOnlyList<TerrainChunkData> chunks, int scannedTiles)
{
    using var atlas = TerrainImageIo.BuildAlphaAtlasFromChunks(chunks);
    var atlasRoundtrip = TerrainImageIo.DecodeAlphaShadowArrayFromAtlas(atlas);

    var legacyDiffs = new ChannelTotals();
    var atlasDiffs = new ChannelTotals();
    int impactedChunks = 0;
    DifferenceSample? sample = null;

    foreach (var chunk in chunks)
    {
        bool chunkImpacted = false;
        int slice = (chunk.ChunkY * 16) + chunk.ChunkX;

        for (int layer = 1; layer <= 3; layer++)
        {
            if (!chunk.AlphaMaps.TryGetValue(layer, out var alphaMap) || alphaMap == null || alphaMap.Length < 64 * 64)
            {
                continue;
            }

            AnalyzeChannel(
                alphaMap,
                slice,
                channelOffset: layer - 1,
                channelName: $"alpha{layer}",
                chunk.ChunkX,
                chunk.ChunkY,
                legacyDiffs,
                atlasDiffs,
                ref chunkImpacted,
                ref sample,
                atlasRoundtrip);
        }

        if (chunk.ShadowMap != null && chunk.ShadowMap.Length >= 64 * 64)
        {
            AnalyzeChannel(
                chunk.ShadowMap,
                slice,
                channelOffset: 3,
                channelName: "shadow",
                chunk.ChunkX,
                chunk.ChunkY,
                legacyDiffs,
                atlasDiffs,
                ref chunkImpacted,
                ref sample,
                atlasRoundtrip);
        }

        if (chunkImpacted)
        {
            impactedChunks++;
        }
    }

    return new TileReport(
        dataset,
        tileX,
        tileY,
        chunks.Count,
        impactedChunks,
        scannedTiles,
        legacyDiffs,
        atlasDiffs,
        sample,
        MissingReason: null);
}

static void AnalyzeChannel(
    byte[] source,
    int slice,
    int channelOffset,
    string channelName,
    int chunkX,
    int chunkY,
    ChannelTotals legacyDiffs,
    ChannelTotals atlasDiffs,
    ref bool chunkImpacted,
    ref DifferenceSample? sample,
    byte[] atlasRoundtrip)
{
    int sliceBase = slice * 64 * 64 * 4;

    for (int y = 0; y < 64; y++)
    {
        for (int x = 0; x < 64; x++)
        {
            int directIndex = y * 64 + x;
            int legacyIndex = EdgeFixedIndex64(x, y);
            byte directValue = source[directIndex];
            byte legacyValue = source[legacyIndex];
            byte atlasValue = atlasRoundtrip[sliceBase + directIndex * 4 + channelOffset];

            if (legacyValue != directValue)
            {
                legacyDiffs.Add(channelName);
                chunkImpacted = true;
                sample ??= new DifferenceSample(chunkX, chunkY, channelName, x, y, directValue, legacyValue, atlasValue);
            }

            if (atlasValue != directValue)
            {
                atlasDiffs.Add(channelName);
                chunkImpacted = true;
                sample ??= new DifferenceSample(chunkX, chunkY, channelName, x, y, directValue, legacyValue, atlasValue);
            }
        }
    }
}

static int EdgeFixedIndex64(int x, int y)
{
    if (x >= 63) x = 62;
    if (y >= 63) y = 62;
    return y * 64 + x;
}

static void PrintReport(TileReport report)
{
    Console.WriteLine($"Dataset: {report.Dataset}");

    if (report.MissingReason != null)
    {
        Console.WriteLine($"  Status: missing data ({report.MissingReason})");
        Console.WriteLine();
        return;
    }

    if (!report.HasImpact)
    {
        Console.WriteLine($"  Status: scanned {report.ScannedTiles} tiles, found no explicit alpha/shadow edge differences");
        Console.WriteLine();
        return;
    }

    Console.WriteLine($"  Tile: ({report.TileX},{report.TileY}) after scanning {report.ScannedTiles} tile(s)");
    Console.WriteLine($"  Chunks: {report.ChunkCount}, impacted chunks: {report.ImpactedChunks}");
    Console.WriteLine($"  Legacy edge-remap diff bytes: {report.LegacyDiffs}");
    Console.WriteLine($"  Atlas roundtrip diff bytes: {report.AtlasDiffs}");
    if (report.Sample != null)
    {
        Console.WriteLine(
            $"  Sample: chunk ({report.Sample.ChunkX},{report.Sample.ChunkY}) {report.Sample.Channel} texel ({report.Sample.X},{report.Sample.Y}) direct={report.Sample.DirectValue} legacy={report.Sample.LegacyValue} atlas={report.Sample.AtlasValue}");
    }
    Console.WriteLine();
}

internal sealed class ChannelTotals
{
    public int Alpha1 { get; private set; }
    public int Alpha2 { get; private set; }
    public int Alpha3 { get; private set; }
    public int Shadow { get; private set; }

    public int Total => Alpha1 + Alpha2 + Alpha3 + Shadow;

    public void Add(string channelName)
    {
        switch (channelName)
        {
            case "alpha1":
                Alpha1++;
                break;
            case "alpha2":
                Alpha2++;
                break;
            case "alpha3":
                Alpha3++;
                break;
            case "shadow":
                Shadow++;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(channelName), channelName, null);
        }
    }

    public override string ToString()
    {
        return $"alpha1={Alpha1}, alpha2={Alpha2}, alpha3={Alpha3}, shadow={Shadow}, total={Total}";
    }
}

internal sealed record DifferenceSample(
    int ChunkX,
    int ChunkY,
    string Channel,
    int X,
    int Y,
    byte DirectValue,
    byte LegacyValue,
    byte AtlasValue);

internal sealed record TileReport(
    string Dataset,
    int TileX,
    int TileY,
    int ChunkCount,
    int ImpactedChunks,
    int ScannedTiles,
    ChannelTotals LegacyDiffs,
    ChannelTotals AtlasDiffs,
    DifferenceSample? Sample,
    string? MissingReason)
{
    public bool HasImpact => LegacyDiffs.Total > 0 || AtlasDiffs.Total > 0;

    public static TileReport Missing(string dataset, string reason) =>
        new(dataset, -1, -1, 0, 0, 0, new ChannelTotals(), new ChannelTotals(), null, reason);

    public static TileReport NoImpact(string dataset, int scannedTiles, int totalTiles) =>
        new(dataset, -1, -1, totalTiles, 0, scannedTiles, new ChannelTotals(), new ChannelTotals(), null, null);
}

internal sealed record LayerTrace(
    int Layer,
    int TextureIndex,
    string TextureName,
    bool TextureInBounds,
    bool TextureFileExists,
    uint LayerFlags,
    uint LayerAlphaOffset,
    uint? SourceFlags,
    uint? SourceAlphaOffset,
    int? RawSpan,
    string RawHeadHex,
    int DecodedLength,
    int NonZeroCount,
    int ZeroCount,
    bool MostlyZero);

internal sealed record PackedParity(
    int Alpha1Diff,
    int Alpha2Diff,
    int Alpha3Diff,
    int ShadowDiff);

internal sealed record PackedSample(
    int X,
    int Y,
    byte Alpha1,
    byte Alpha2,
    byte Alpha3,
    byte Shadow);

internal sealed record ChunkTrace(
    int TileX,
    int TileY,
    int ChunkX,
    int ChunkY,
    int McinIndex,
    string BuildVersion,
    int McalLength,
    string McalHeadHex,
    IReadOnlyList<LayerTrace> LayerTraces,
    PackedParity PackedParity,
    PackedSample PackedSample);