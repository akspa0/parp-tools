using System.Buffers.Binary;
using WowViewer.Core.IO.Files;

/// <summary>
/// Histograms the second uint16 of every MH2O SMLiquidInstance across a client's ADTs.
/// </summary>
/// <remarks>
/// In Cataclysm and later that field is <c>liquid_object_or_lvf</c>: values below 42 are a liquid
/// vertex format (0-3), values 42 and above are a <c>LiquidObject.dbc</c> id and the real vertex
/// format has to be resolved through it. <see cref="WowViewer.Core.IO.Maps.AdtLiquidReader"/> casts
/// the field straight to a vertex-format enum and switches on it with no default, so an id lands on
/// no case, the height array stays null, and the mesh falls back to a flat plane at the layer's
/// minimum height.
/// <para>
/// This command exists to establish whether that actually happens in the operator's data before
/// anything is changed on the strength of it. The reported histogram is the evidence; a run that
/// shows only values 0-3 refutes the hypothesis outright.
/// </para>
/// </remarks>
internal static class AdtLiquidFormatSupport
{
    private const int ChunkHeaderSize = 8;
    private const int Mh2oChunkHeaderCount = 256;
    private const int Mh2oChunkHeaderSize = 12;
    private const int LayerSize = 24;

    /// <summary>Values at or above this are a LiquidObject.dbc id, not a vertex format.</summary>
    private const int LiquidObjectThreshold = 42;

    public static void Run(string[] args)
    {
        string? clientRoot = GetOption(args, "--client");
        if (string.IsNullOrWhiteSpace(clientRoot))
        {
            Console.Error.WriteLine("Usage: adt liquid-formats --client <client-dir> [--map <name>] [--limit <n>]");
            Environment.ExitCode = 1;
            return;
        }

        string? mapFilter = GetOption(args, "--map");
        int limit = int.TryParse(GetOption(args, "--limit"), out int parsedLimit) ? parsedLimit : 200;

        using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
        ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(
            archiveCatalog, [clientRoot], new ArchiveCatalogBootstrapOptions());

        IEnumerable<string> candidates = bootstrap.AllFiles
            .Where(static path => path.EndsWith(".adt", StringComparison.OrdinalIgnoreCase));

        if (!string.IsNullOrWhiteSpace(mapFilter))
            candidates = candidates.Where(path => path.Contains(mapFilter, StringComparison.OrdinalIgnoreCase));

        // Split ADTs carry no MH2O; only the root tile does. Filtering here keeps the scanned count
        // meaningful rather than counting three-quarters misses.
        candidates = candidates.Where(static path =>
            !path.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase)
            && !path.EndsWith("_obj1.adt", StringComparison.OrdinalIgnoreCase)
            && !path.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase)
            && !path.EndsWith("_tex1.adt", StringComparison.OrdinalIgnoreCase));

        var formatCounts = new SortedDictionary<int, int>();
        var typeByFormat = new SortedDictionary<int, SortedSet<int>>();
        var detail = new SortedDictionary<int, FormatDetail>();
        int scanned = 0;
        int withWater = 0;
        int layers = 0;

        foreach (string path in candidates.OrderBy(static p => p, StringComparer.OrdinalIgnoreCase))
        {
            if (scanned >= limit)
                break;

            byte[]? bytes = archiveCatalog.ReadFile(path);
            if (bytes is null)
                continue;

            scanned++;
            if (!TryFindChunk(bytes, "MH2O", out int payloadOffset, out int payloadLength))
                continue;

            withWater++;
            layers += ScanMh2o(bytes, payloadOffset, payloadLength, formatCounts, typeByFormat, detail);
        }

        Console.WriteLine("WowViewer.Tool.Inspect MH2O liquid vertex-format histogram");
        Console.WriteLine($"client={clientRoot} map={mapFilter ?? "(all)"}");
        Console.WriteLine($"root ADTs scanned={scanned} withMh2o={withWater} liquidLayers={layers}");
        Console.WriteLine();
        Console.WriteLine("liquid_object_or_lvf  count      meaning");

        int objectIdLayers = 0;
        foreach ((int value, int count) in formatCounts)
        {
            string meaning = value >= LiquidObjectThreshold
                ? $"LiquidObject.dbc id -> NOT a vertex format (liquidTypes: {string.Join(',', typeByFormat[value])})"
                : value switch
                {
                    0 => "LVF 0 height+depth",
                    1 => "LVF 1 height+uv",
                    2 => "LVF 2 depth only",
                    3 => "LVF 3 height+uv+depth",
                    _ => "unknown, below the LiquidObject threshold",
                };

            if (value >= LiquidObjectThreshold)
                objectIdLayers += count;

            Console.WriteLine($"{value,20}  {count,9}  {meaning}");
        }

        Console.WriteLine();
        Console.WriteLine("value   layers  withVertexData  sloped  maxSpread  sizes(w x h)");
        foreach ((int value, FormatDetail d) in detail)
        {
            string sizes = string.Join(" ", d.Sizes.OrderBy(static s => s.Width).ThenBy(static s => s.Height)
                .Select(static s => $"{s.Width}x{s.Height}").Distinct().Take(6));
            Console.WriteLine($"{value,5}  {d.Layers,6}  {d.WithVertexData,14}  {d.Sloped,6}  {d.MaxSpread,9:0.00}  {sizes}");
            Console.WriteLine($"        vertexBlocks: plausibleHeights={d.HeightsPlausible} implausible={d.HeightsImplausible}"
                + $" varying={d.HeightsVary} maxSpread={d.MaxVertexSpread:0.00} disagreeWithHeader={d.HeightsDisagreeWithHeader}");
        }

        Console.WriteLine();
        if (layers == 0)
        {
            Console.WriteLine("VERDICT: no liquid layers found. Nothing measured; widen --map or --limit.");
            return;
        }

        double share = 100.0 * objectIdLayers / layers;
        Console.WriteLine(objectIdLayers > 0
            ? $"VERDICT: {objectIdLayers} of {layers} layers ({share:0.0}%) carry a LiquidObject id. "
              + "AdtLiquidReader's switch has no case for these, so their heights parse as null and "
              + "the mesh falls back to a flat plane. Hypothesis CONFIRMED."
            : "VERDICT: every layer uses a vertex format in 0-3. The LiquidObject hypothesis is "
              + "REFUTED for this data; the flat-plane cause is elsewhere.");
    }

    private static int ScanMh2o(
        byte[] bytes,
        int payloadOffset,
        int payloadLength,
        SortedDictionary<int, int> formatCounts,
        SortedDictionary<int, SortedSet<int>> typeByFormat,
        SortedDictionary<int, FormatDetail> detail)
    {
        int layers = 0;
        ReadOnlySpan<byte> payload = bytes.AsSpan(payloadOffset, payloadLength);

        for (int chunkIndex = 0; chunkIndex < Mh2oChunkHeaderCount; chunkIndex++)
        {
            int headerOffset = chunkIndex * Mh2oChunkHeaderSize;
            if (headerOffset + Mh2oChunkHeaderSize > payload.Length)
                break;

            uint offsetInstances = BinaryPrimitives.ReadUInt32LittleEndian(payload[headerOffset..]);
            uint layerCount = BinaryPrimitives.ReadUInt32LittleEndian(payload[(headerOffset + 4)..]);
            if (layerCount == 0 || offsetInstances == 0)
                continue;

            for (int layer = 0; layer < layerCount; layer++)
            {
                int layerOffset = (int)offsetInstances + (layer * LayerSize);
                if (layerOffset < 0 || layerOffset + LayerSize > payload.Length)
                    break;

                int liquidType = BinaryPrimitives.ReadUInt16LittleEndian(payload[layerOffset..]);
                int format = BinaryPrimitives.ReadUInt16LittleEndian(payload[(layerOffset + 2)..]);

                // Geometry, so the fix can be written from evidence rather than from the wiki.
                // A layer whose min and max height differ is a SLOPED surface: flattening it to
                // minHeight is visibly wrong and is what produces the chunk-boundary steps.
                float minHeight = BitConverter.ToSingle(payload[(layerOffset + 4)..]);
                float maxHeight = BitConverter.ToSingle(payload[(layerOffset + 8)..]);
                int width = payload[layerOffset + 14];
                int height = payload[layerOffset + 15];
                uint vertexDataOffset = BinaryPrimitives.ReadUInt32LittleEndian(payload[(layerOffset + 20)..]);

                if (!detail.TryGetValue(format, out FormatDetail? d))
                {
                    d = new FormatDetail();
                    detail[format] = d;
                }

                d.Layers++;
                if (vertexDataOffset != 0)
                    d.WithVertexData++;
                if (maxHeight - minHeight > 0.001f)
                {
                    d.Sloped++;
                    d.MaxSpread = Math.Max(d.MaxSpread, maxHeight - minHeight);
                }

                d.Sizes.Add((width, height));

                // Read the vertex block as if it were an LVF-0/1/3 heightmap. If those floats vary
                // and sit in a plausible world-Z range, they are heights the current reader is
                // discarding — which is the whole question. If they are garbage, the layer is
                // genuinely depth-only and flat is correct.
                if (vertexDataOffset != 0)
                {
                    int vertexCount = (width + 1) * (height + 1);
                    int vo = (int)vertexDataOffset;
                    if (vo >= 0 && vo + (vertexCount * 4) <= payload.Length)
                    {
                        float lo = float.MaxValue;
                        float hi = float.MinValue;
                        bool plausible = true;
                        for (int v = 0; v < vertexCount; v++)
                        {
                            float value = BitConverter.ToSingle(payload[(vo + (v * 4))..]);
                            if (float.IsNaN(value) || float.IsInfinity(value) || Math.Abs(value) > 20000f)
                            {
                                plausible = false;
                                break;
                            }

                            lo = Math.Min(lo, value);
                            hi = Math.Max(hi, value);
                        }

                        if (plausible)
                        {
                            d.HeightsPlausible++;
                            float spread = hi - lo;
                            if (spread > 0.01f)
                            {
                                d.HeightsVary++;
                                d.MaxVertexSpread = Math.Max(d.MaxVertexSpread, spread);
                            }

                            // Does the vertex block agree with the header's declared level?
                            if (Math.Abs(lo - minHeight) > 0.5f)
                                d.HeightsDisagreeWithHeader++;
                        }
                        else
                        {
                            d.HeightsImplausible++;
                        }
                    }
                }

                formatCounts[format] = formatCounts.TryGetValue(format, out int existing) ? existing + 1 : 1;
                if (!typeByFormat.TryGetValue(format, out SortedSet<int>? types))
                {
                    types = [];
                    typeByFormat[format] = types;
                }

                types.Add(liquidType);
                layers++;
            }
        }

        return layers;
    }

    /// <summary>Walks the top-level chunk list. ADT chunk sizes are exact; no padding is applied.</summary>
    private static bool TryFindChunk(byte[] bytes, string magic, out int payloadOffset, out int payloadLength)
    {
        payloadOffset = 0;
        payloadLength = 0;

        // Chunk magics are stored reversed on disk.
        Span<char> reversed = stackalloc char[4];
        for (int i = 0; i < 4; i++)
            reversed[i] = magic[3 - i];

        int offset = 0;
        while (offset + ChunkHeaderSize <= bytes.Length)
        {
            bool match = true;
            for (int i = 0; i < 4; i++)
            {
                if (bytes[offset + i] != (byte)reversed[i])
                {
                    match = false;
                    break;
                }
            }

            uint size = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset + 4, 4));
            if (size > int.MaxValue || offset + ChunkHeaderSize + (long)size > bytes.Length)
                return false;

            if (match)
            {
                payloadOffset = offset + ChunkHeaderSize;
                payloadLength = (int)size;
                return payloadLength > 0;
            }

            offset += ChunkHeaderSize + (int)size;
        }

        return false;
    }

    private sealed class FormatDetail
    {
        public int Layers;
        public int WithVertexData;

        /// <summary>Layers whose max height exceeds their min height — a surface with actual slope.</summary>
        public int Sloped;
        public float MaxSpread;

        /// <summary>Vertex blocks that read as finite, in-range floats — i.e. look like heights.</summary>
        public int HeightsPlausible;
        public int HeightsImplausible;

        /// <summary>Vertex blocks whose values actually differ across the 9x9 grid: a sloped surface.</summary>
        public int HeightsVary;
        public float MaxVertexSpread;

        /// <summary>Blocks whose lowest vertex differs from the header's minHeight by more than 0.5.</summary>
        public int HeightsDisagreeWithHeader;
        public HashSet<(int Width, int Height)> Sizes { get; } = [];
    }

    private static string? GetOption(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
                return args[i + 1];
        }

        return null;
    }
}
