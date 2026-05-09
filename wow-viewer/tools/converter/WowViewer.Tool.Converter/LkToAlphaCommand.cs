using System.Diagnostics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Tool.Converter;

internal static class LkToAlphaCommand
{
    public static void Run(string[] args)
    {
        try
        {
            LkToAlphaOptions options = ParseOptions(args);

            bool useMpq = !string.IsNullOrEmpty(options.ClientRoot) && !string.IsNullOrEmpty(options.MapName);

            if (!useMpq && string.IsNullOrEmpty(options.InputDir))
            {
                Console.Error.WriteLine("Error: Provide either --input <dir> (loose ADT files) or --client-root <dir> --map <name> (MPQ archives).");
                Environment.ExitCode = 1;
                return;
            }

            if (string.IsNullOrEmpty(options.OutputPath))
            {
                Console.Error.WriteLine("Error: --output <path> is required (output .wdt file path).");
                Environment.ExitCode = 1;
                return;
            }

            string outputPath = Path.GetFullPath(options.OutputPath);
            string mapName = useMpq ? options.MapName! : Path.GetFileNameWithoutExtension(outputPath);

            Console.WriteLine("WowViewer.Tool.Converter convert-lk-to-alpha report");

            var sw = Stopwatch.StartNew();
            var tiles = new Dictionary<(int, int), AlphaTileData>();
            int converted = 0;
            int failed = 0;
            int totalTiles = 0;
            var warnings = new List<string>();

            if (useMpq)
            {
                string clientRoot = Path.GetFullPath(options.ClientRoot!);
                if (!Directory.Exists(clientRoot))
                {
                    Console.Error.WriteLine($"Error: Client root not found: {clientRoot}");
                    Environment.ExitCode = 1;
                    return;
                }

                Console.WriteLine($"  Client:   {clientRoot}");
                Console.WriteLine($"  Map:      {mapName}");
                Console.WriteLine($"  Output:   {outputPath}");
                Console.WriteLine($"  Verbose:  {options.Verbose}");

                using var catalog = new NativeMpqService();
                catalog.LoadArchives([clientRoot]);

                string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
                byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
                if (wdtBytes is null)
                {
                    Console.Error.WriteLine($"Error: Could not read WDT '{wdtVirtual}' from client.");
                    Environment.ExitCode = 1;
                    return;
                }

                using var wdtStream = new MemoryStream(wdtBytes, writable: false);

                int? limit = GetIntOption(args, "--limit", "-n");

                for (int ty = 0; ty < 64; ty++)
                {
                    for (int tx = 0; tx < 64; tx++)
                    {

                        string adtVirtual = $"World\\Maps\\{mapName}\\{mapName}_{tx}_{ty}.adt";
                        byte[]? adtBytes = catalog.ReadFile(adtVirtual);
                        if (adtBytes is null)
                            continue;

                        totalTiles++;

                        try
                        {
                            LkAdtData adtData = ReadLkAdt(adtBytes, tx, ty);
                            AlphaTileData tileData = LkToAlphaConverter.ConvertTile(adtData, tx, ty);
                            tiles[(tx, ty)] = tileData;
                            converted++;

                            if (options.Verbose)
                                Console.WriteLine($"  Converted: {mapName}_{tx}_{ty} ({adtBytes.Length:N0} bytes)");

                            if (limit.HasValue && converted >= limit.Value)
                                break;
                        }
                        catch (Exception ex)
                        {
                            failed++;
                            warnings.Add($"{mapName}_{tx}_{ty}: {ex.Message}");
                            if (options.Verbose)
                                Console.Error.WriteLine($"  Error converting {mapName}_{tx}_{ty}: {ex}");
                        }
                    }

                    if (limit.HasValue && converted >= limit.Value)
                        break;
                }
            }
            else
            {
                string inputDir = Path.GetFullPath(options.InputDir!);

                if (!Directory.Exists(inputDir))
                {
                    Console.Error.WriteLine($"Error: Directory not found: {inputDir}");
                    Environment.ExitCode = 1;
                    return;
                }

                Console.WriteLine($"  Input:    {inputDir}");
                Console.WriteLine($"  Output:   {outputPath}");
                Console.WriteLine($"  Verbose:  {options.Verbose}");

                var adtFiles = Directory.GetFiles(inputDir, "*_*.adt", SearchOption.TopDirectoryOnly)
                    .Where(f => !f.Contains("_obj", StringComparison.OrdinalIgnoreCase) && !f.Contains("_tex", StringComparison.OrdinalIgnoreCase) && !f.Contains("_lod", StringComparison.OrdinalIgnoreCase))
                    .ToList();

                if (adtFiles.Count == 0)
                {
                    Console.Error.WriteLine("Error: No ADT files found in directory.");
                    Environment.ExitCode = 1;
                    return;
                }

                Console.WriteLine($"  Found {adtFiles.Count} ADT files.");
                totalTiles = adtFiles.Count;

                foreach (var adtFile in adtFiles)
                {
                    string fileName = Path.GetFileNameWithoutExtension(adtFile);
                    string[] parts = fileName.Split('_');
                    if (parts.Length < 3 || !int.TryParse(parts[^2], out int tileX) || !int.TryParse(parts[^1], out int tileY))
                    {
                        warnings.Add($"Cannot parse tile coords from: {fileName}");
                        continue;
                    }

                    try
                    {
                        byte[] adtBytes = File.ReadAllBytes(adtFile);
                        LkAdtData adtData = ReadLkAdt(adtBytes, tileX, tileY);
                        AlphaTileData tileData = LkToAlphaConverter.ConvertTile(adtData, tileX, tileY);
                        tiles[(tileX, tileY)] = tileData;
                        converted++;

                        if (options.Verbose)
                            Console.WriteLine($"  Converted: {fileName} ({adtBytes.Length:N0} bytes)");
                    }
                    catch (Exception ex)
                    {
                        failed++;
                        warnings.Add($"{fileName}: {ex.Message}");
                        if (options.Verbose)
                            Console.Error.WriteLine($"  Error converting {fileName}: {ex}");
                    }
                }
            }

            if (tiles.Count == 0)
            {
                Console.Error.WriteLine("Error: No tiles were successfully converted.");
                Environment.ExitCode = 1;
                return;
            }

            string outputDir = Path.GetDirectoryName(outputPath) ?? ".";
            Directory.CreateDirectory(outputDir);

            byte[] wdtData = AlphaWdtWriter.Build(mapName, tiles);
            File.WriteAllBytes(outputPath, wdtData);

            sw.Stop();
            Console.WriteLine($"  Converted: {converted}/{totalTiles} tiles");
            Console.WriteLine($"  Failed:    {failed} tiles");
            Console.WriteLine($"  Output:    {outputPath} ({wdtData.Length:N0} bytes)");
            Console.WriteLine($"  Elapsed:   {sw.ElapsedMilliseconds}ms");

            if (warnings.Count > 0)
            {
                Console.WriteLine($"  Warnings:  {warnings.Count}");
                foreach (var w in warnings.Take(10))
                    Console.WriteLine($"    {w}");
                if (warnings.Count > 10)
                    Console.WriteLine($"    ... and {warnings.Count - 10} more");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            if (args.Contains("--verbose") || args.Contains("-v"))
                Console.Error.WriteLine(ex.StackTrace);
            Environment.ExitCode = 1;
        }
    }

    public static LkAdtData ReadLkAdtPublic(byte[] adtBytes, int tileX, int tileY)
        => ReadLkAdt(adtBytes, tileX, tileY);

private static LkAdtData ReadLkAdt(byte[] adtBytes, int tileX, int tileY)
    {
        using var ms = new MemoryStream(adtBytes, writable: false);
        using var br = new BinaryReader(ms, System.Text.Encoding.ASCII, leaveOpen: true);

        var textureNames = new List<string>();
        var modelNames = new List<string>();
        var worldModelNames = new List<string>();
        var modelPlacements = new List<LkMddfEntry>();
        var worldModelPlacements = new List<LkModfEntry>();
        var chunks = new List<LkMcnkData>(256);
        uint mhdrFlags = 0;
        int[,,]? mfboFlightBounds = null;

        while (ms.Position + 8 <= ms.Length)
        {
            byte[] tagBytes = br.ReadBytes(4);
            uint size = br.ReadUInt32();
            long chunkEnd = Math.Min(ms.Position + size, ms.Length);
            string tag = System.Text.Encoding.ASCII.GetString(tagBytes);

            if (tag == "MVER")
            {
                br.ReadInt32();
            }
            else if (tag == "MHDR")
            {
                mhdrFlags = br.ReadUInt32();
                ms.Position = chunkEnd;
            }
            else if (tag == "MTEX")
            {
                textureNames = ReadStringBlock(br, (int)size);
            }
            else if (tag == "MMDX")
            {
                modelNames = ReadStringBlock(br, (int)size);
            }
            else if (tag == "MWMO")
            {
                worldModelNames = ReadStringBlock(br, (int)size);
            }
            else if (tag == "MDDF")
            {
                modelPlacements = ReadMddfEntries(br, (int)size);
            }
            else if (tag == "MODF")
            {
                worldModelPlacements = ReadModfEntries(br, (int)size);
            }
            else if (tag == "MCNK")
            {
                chunks.Add(ReadMcnkChunk(br, adtBytes, (int)size));
            }
            else if (tag == "MFBO")
            {
                if (size >= 36)
                {
                    mfboFlightBounds = new int[2, 3, 3];
                    for (int plane = 0; plane < 2; plane++)
                    {
                        for (int row = 0; row < 3; row++)
                        {
                            for (int col = 0; col < 3; col++)
                            {
                                short val = br.ReadInt16();
                                mfboFlightBounds[plane, row, col] = val;
                            }
                        }
                    }
                }
                else
                {
                    ms.Position = chunkEnd;
                }
            }
            else
            {
                ms.Position = chunkEnd;
            }

            if ((size & 1) != 0 && chunkEnd < ms.Length)
                ms.Position = chunkEnd + 1;
            else if (chunkEnd <= ms.Length)
                ms.Position = chunkEnd;
            else
                break;
        }

        return new LkAdtData
        {
            MapName = "",
            TileX = tileX,
            TileY = tileY,
            TextureNames = textureNames,
            ModelNames = modelNames,
            WorldModelNames = worldModelNames,
            ModelPlacements = modelPlacements,
            WorldModelPlacements = worldModelPlacements,
            Chunks = AttachLiquidData(adtBytes, chunks, tileX, tileY),
            MhdrFlags = mhdrFlags,
            MfboFlightBounds = mfboFlightBounds
        };
    }

    private static IReadOnlyList<LkMcnkData> AttachLiquidData(byte[] adtBytes, List<LkMcnkData> chunks, int tileX, int tileY)
    {
        try
        {
            using var stream = new MemoryStream(adtBytes, writable: false);
            MapFileSummary summary = MapFileSummaryReader.Read(stream, $"tile_{tileX}_{tileY}.adt");
            AdtLiquidFile liquidFile = AdtLiquidReader.Read(stream, summary);
            if (liquidFile.Chunks.Count == 0)
                return chunks;

            var liquidByChunkIndex = liquidFile.Chunks
                .Where(static chunk => chunk.Layers.Count > 0)
                .ToDictionary(static chunk => chunk.ChunkIndex);

            var result = new List<LkMcnkData>(chunks.Count);
            for (int index = 0; index < chunks.Count; index++)
            {
                LkMcnkData chunk = chunks[index];
                liquidByChunkIndex.TryGetValue(index, out AdtLiquidChunk? liquidData);
                result.Add(new LkMcnkData
                {
                    IndexX = chunk.IndexX,
                    IndexY = chunk.IndexY,
                    Flags = chunk.Flags,
                    AreaId = chunk.AreaId,
                    NLayers = chunk.NLayers,
                    HoleMask = chunk.HoleMask,
                    BaseHeight = chunk.BaseHeight,
                    Heights = chunk.Heights,
                    Normals = chunk.Normals,
                    ShadowMap = chunk.ShadowMap,
                    AlphaMapData = chunk.AlphaMapData,
                    AlphaMapSize = chunk.AlphaMapSize,
                    Layers = chunk.Layers,
                    DoodadRefs = chunk.DoodadRefs,
                    WorldModelRefs = chunk.WorldModelRefs,
                    LiquidData = liquidData,
                    MccvColors = chunk.MccvColors,
                    MclvLighting = chunk.MclvLighting,
                    PosX = chunk.PosX,
                    PosY = chunk.PosY,
                    PosZ = chunk.PosZ
                });
            }

            return result;
        }
        catch
        {
            return chunks;
        }
    }

    private static List<string> ReadStringBlock(BinaryReader br, int size)
    {
        var names = new List<string>();
        byte[] data = br.ReadBytes(size);
        int start = 0;
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] == 0)
            {
                if (i > start)
                    names.Add(System.Text.Encoding.UTF8.GetString(data, start, i - start));
                start = i + 1;
            }
        }
        if (start < data.Length)
            names.Add(System.Text.Encoding.UTF8.GetString(data, start, data.Length - start));
        return names;
    }

    private const int LkMddfEntrySize = 36;
    private const int LkModfEntrySize = 64;
    private const float MapOrigin = 17066.666f;

    private static List<LkMddfEntry> ReadMddfEntries(BinaryReader br, int size)
    {
        var entries = new List<LkMddfEntry>();
        int count = size / LkMddfEntrySize;
        for (int i = 0; i < count; i++)
        {
            int nameId = br.ReadInt32();
            int uniqueId = br.ReadInt32();
            float rawX = br.ReadSingle();
            float rawZ = br.ReadSingle();
            float rawY = br.ReadSingle();
            float rotX = br.ReadSingle();
            float rotZ = br.ReadSingle();
            float rotY = br.ReadSingle();
            ushort scale = br.ReadUInt16();
            br.ReadUInt16();
            entries.Add(new LkMddfEntry(nameId, uniqueId,
                new System.Numerics.Vector3(MapOrigin - rawY, MapOrigin - rawX, rawZ),
                new System.Numerics.Vector3(rotX, rotY, rotZ),
                scale / 1024f));
        }
        return entries;
    }

    private static List<LkModfEntry> ReadModfEntries(BinaryReader br, int size)
    {
        var entries = new List<LkModfEntry>();
        int count = size / LkModfEntrySize;
        for (int i = 0; i < count; i++)
        {
            int nameId = br.ReadInt32();
            int uniqueId = br.ReadInt32();
            float rawX = br.ReadSingle();
            float rawZ = br.ReadSingle();
            float rawY = br.ReadSingle();
            float rotX = br.ReadSingle();
            float rotZ = br.ReadSingle();
            float rotY = br.ReadSingle();
            float bbMinX = br.ReadSingle();
            float bbMinZ = br.ReadSingle();
            float bbMinY = br.ReadSingle();
            float bbMaxX = br.ReadSingle();
            float bbMaxZ = br.ReadSingle();
            float bbMaxY = br.ReadSingle();
            ushort modfFlags = br.ReadUInt16();
            ushort doodadSet = br.ReadUInt16();
            ushort nameSet = br.ReadUInt16();
            ushort modfScale = br.ReadUInt16();
            entries.Add(new LkModfEntry(nameId, uniqueId,
                new System.Numerics.Vector3(MapOrigin - rawY, MapOrigin - rawX, rawZ),
                new System.Numerics.Vector3(rotX, rotY, rotZ),
                new System.Numerics.Vector3(MapOrigin - bbMaxY, MapOrigin - bbMaxX, bbMinZ),
                new System.Numerics.Vector3(MapOrigin - bbMinY, MapOrigin - bbMinX, bbMaxZ),
                modfFlags, doodadSet, nameSet, modfScale / 1024f));
        }
        return entries;
    }

    private static LkMcnkData ReadMcnkChunk(BinaryReader br, byte[] adtBytes, int declaredSize)
    {
        long mcnkStart = br.BaseStream.Position - 8;
        int headerSize = 128;
        if (declaredSize < headerSize)
        {
            br.BaseStream.Position = mcnkStart + 8 + declaredSize;
            return new LkMcnkData { IndexX = 0, IndexY = 0, Heights = [], Normals = [], Layers = [] };
        }

        byte[] header = br.ReadBytes(headerSize);

        int mcnkFlags = BitConverter.ToInt32(header, 0x00);
        int indexX = BitConverter.ToInt32(header, 0x04);
        int indexY = BitConverter.ToInt32(header, 0x08);
        int nLayers = BitConverter.ToInt32(header, 0x0C);
        int ofsMcvt = BitConverter.ToInt32(header, 0x14);
        int ofsMcnr = BitConverter.ToInt32(header, 0x18);
        int ofsMcly = BitConverter.ToInt32(header, 0x1C);
        int ofsMcrf = BitConverter.ToInt32(header, 0x20);
        int ofsMcal = BitConverter.ToInt32(header, 0x24);
        int sizeMcal = BitConverter.ToInt32(header, 0x28);
        int ofsMcsh = BitConverter.ToInt32(header, 0x2C);
        int sizeMcsh = BitConverter.ToInt32(header, 0x30);
        int areaId = BitConverter.ToInt32(header, 0x34);
        int nDoodadRefs = BitConverter.ToInt32(header, 0x38);
        int holeMask = BitConverter.ToInt32(header, 0x3C);
        int nMapObjRefs = BitConverter.ToInt32(header, 0x3C + 0x14);
        int sizeMclq = BitConverter.ToInt32(header, 0x64);
        int ofsLiquid = BitConverter.ToInt32(header, 0x68);
        float baseHeight = BitConverter.ToSingle(header, 0x70);
        float posX = BitConverter.ToSingle(header, 0x68);
        float posY = BitConverter.ToSingle(header, 0x6C);

        int mcnkPayloadOffset = (int)mcnkStart + 8;
        int mcnkPayloadEnd = mcnkPayloadOffset + declaredSize;
        if (mcnkPayloadEnd > adtBytes.Length)
            mcnkPayloadEnd = adtBytes.Length;

        byte[] heightData = [];
        byte[] normalData = [];
        byte[]? shadowData = null;
        var layers = new List<LkMclyEntry>();
        byte[]? alphaData = null;
        int alphaTotalSize = 0;
        var doodadRefs = new List<int>();
        var worldModelRefs = new List<int>();

        // MCVT — use header offset
        if (ofsMcvt >= headerSize && ofsMcvt + 145 * 4 <= declaredSize)
        {
            int srcOffset = mcnkPayloadOffset + ofsMcvt;
            if (srcOffset + 145 * 4 <= adtBytes.Length)
            {
                heightData = new byte[145 * 4];
                Buffer.BlockCopy(adtBytes, srcOffset, heightData, 0, 145 * 4);
            }
        }

        // MCNR — use header offset, always 448 bytes
        if (ofsMcnr >= headerSize && ofsMcnr + 448 <= declaredSize)
        {
            int srcOffset = mcnkPayloadOffset + ofsMcnr;
            if (srcOffset + 448 <= adtBytes.Length)
            {
                normalData = new byte[448];
                Buffer.BlockCopy(adtBytes, srcOffset, normalData, 0, 448);
            }
        }

        // MCLY — use header offset
        if (ofsMcly >= headerSize)
        {
            int srcOffset = mcnkPayloadOffset + ofsMcly;
            if (srcOffset + 8 <= adtBytes.Length)
            {
                int mclyReadEnd = Math.Min(srcOffset + declaredSize - ofsMcly, adtBytes.Length) - srcOffset;
                int mclyAvail = Math.Max(0, mclyReadEnd);
                int layerCount = mclyAvail / 16;
                for (int i = 0; i < layerCount; i++)
                {
                    int off = srcOffset + i * 16;
                    if (off + 16 > adtBytes.Length) break;
                    uint texId = BitConverter.ToUInt32(adtBytes, off);
                    uint layerMclyFlags = BitConverter.ToUInt32(adtBytes, off + 4);
                    uint alphaOff = BitConverter.ToUInt32(adtBytes, off + 8);
                    uint effectId = BitConverter.ToUInt32(adtBytes, off + 12);
                    layers.Add(new LkMclyEntry(texId, layerMclyFlags, alphaOff, effectId));
                }
            }
        }

        // MCRF — use header offset
        if (ofsMcrf >= headerSize)
        {
            int srcOffset = mcnkPayloadOffset + ofsMcrf;
            int available = Math.Min(declaredSize - ofsMcrf, adtBytes.Length - srcOffset);
            if (available > 8)
            {
                using var mcrfMs = new MemoryStream(adtBytes, srcOffset, available, writable: false);
                using var mcrfBr = new BinaryReader(mcrfMs, System.Text.Encoding.ASCII, leaveOpen: true);
                int nRefs = mcrfBr.ReadInt32();
                for (int i = 0; i < nRefs && mcrfMs.Position + 4 <= mcrfMs.Length; i++)
                    doodadRefs.Add(mcrfBr.ReadInt32());
                if (mcrfMs.Position + 4 <= mcrfMs.Length)
                {
                    int nWmoRefs = mcrfBr.ReadInt32();
                    for (int i = 0; i < nWmoRefs && mcrfMs.Position + 4 <= mcrfMs.Length; i++)
                        worldModelRefs.Add(mcrfBr.ReadInt32());
                }
            }
        }

        // MCAL — use header offset and size
        if (ofsMcal >= headerSize && sizeMcal > 0)
        {
            int srcOffset = mcnkPayloadOffset + ofsMcal;
            if (srcOffset + sizeMcal <= adtBytes.Length)
            {
                alphaData = new byte[sizeMcal];
                Buffer.BlockCopy(adtBytes, srcOffset, alphaData, 0, sizeMcal);
                alphaTotalSize = sizeMcal;
            }
        }

        // MCSH — use header offset and size
        if (ofsMcsh >= headerSize && sizeMcsh > 0)
        {
            int srcOffset = mcnkPayloadOffset + ofsMcsh;
            if (srcOffset + sizeMcsh <= adtBytes.Length)
            {
                shadowData = new byte[sizeMcsh];
                Buffer.BlockCopy(adtBytes, srcOffset, shadowData, 0, sizeMcsh);
            }
        }

        // Scan for MCCV and MCLV sub-chunks in MCNK payload
        byte[]? mccvData = null;
        byte[]? mclvData = null;
        int scanStart = mcnkPayloadOffset + headerSize;
        int scanEnd = mcnkPayloadEnd;
        int pos = scanStart;
        while (pos + 8 <= scanEnd)
        {
            if (pos + 4 <= adtBytes.Length)
            {
                string subTag = System.Text.Encoding.ASCII.GetString(adtBytes, pos, 4);
                if (pos + 8 <= adtBytes.Length)
                {
                    int subSize = BitConverter.ToInt32(adtBytes, pos + 4);
                    if (subSize < 0 || pos + 8 + subSize > scanEnd)
                        break;

                    if (subTag == "MCCV" && subSize >= 580)
                    {
                        int dataOffset = pos + 8;
                        if (dataOffset + 580 <= adtBytes.Length)
                        {
                            mccvData = new byte[580];
                            Buffer.BlockCopy(adtBytes, dataOffset, mccvData, 0, 580);
                        }
                    }
                    else if (subTag == "MCLV" && subSize >= 580)
                    {
                        int dataOffset = pos + 8;
                        if (dataOffset + 580 <= adtBytes.Length)
                        {
                            mclvData = new byte[580];
                            Buffer.BlockCopy(adtBytes, dataOffset, mclvData, 0, 580);
                        }
                    }

                    int skip = 8 + subSize;
                    pos += (skip + 3) & ~3;
                }
                else break;
            }
            else break;
        }

        // Advance stream past MCNK
        br.BaseStream.Position = mcnkStart + 8 + declaredSize;

        float[] heights = new float[145];
        for (int i = 0; i < 145 && i * 4 + 4 <= heightData.Length; i++)
            heights[i] = BitConverter.ToSingle(heightData, i * 4);

        return new LkMcnkData
        {
            IndexX = indexX,
            IndexY = indexY,
            Flags = mcnkFlags,
            AreaId = areaId,
            NLayers = nLayers,
            HoleMask = holeMask,
            BaseHeight = baseHeight,
            Heights = heights,
            Normals = normalData,
            ShadowMap = shadowData,
            AlphaMapData = alphaData,
            AlphaMapSize = alphaTotalSize,
            Layers = layers,
            DoodadRefs = doodadRefs,
            WorldModelRefs = worldModelRefs,
            MccvColors = mccvData,
            MclvLighting = mclvData,
            PosX = posX,
            PosY = posY,
            PosZ = baseHeight
        };
    }

    private static LkToAlphaOptions ParseOptions(string[] args)
    {
        return new LkToAlphaOptions(
            InputDir: GetOption(args, "--input", "-i"),
            OutputPath: GetOption(args, "--output", "-o"),
            ClientRoot: GetOption(args, "--client-root", "-c"),
            MapName: GetOption(args, "--map", "-m"),
            Verbose: HasFlag(args, "--verbose") || HasFlag(args, "-v"));
    }

    private static string? GetOption(string[] args, string longName, string shortName)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], longName, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(args[i], shortName, StringComparison.OrdinalIgnoreCase))
            {
                return args[i + 1];
            }
        }
        return null;
    }

    private static int? GetIntOption(string[] args, string longName, string shortName)
    {
        string? value = GetOption(args, longName, shortName);
        return int.TryParse(value, out int result) ? result : null;
    }

    private static bool HasFlag(string[] args, string name)
    {
        foreach (var arg in args)
        {
            if (string.Equals(arg, name, StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    private readonly record struct LkToAlphaOptions(
        string? InputDir,
        string? OutputPath,
        string? ClientRoot,
        string? MapName,
        bool Verbose);
}