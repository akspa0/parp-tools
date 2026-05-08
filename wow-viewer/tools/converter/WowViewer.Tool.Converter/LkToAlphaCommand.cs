using System.Diagnostics;
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

            if (string.IsNullOrEmpty(options.InputDir))
            {
                Console.Error.WriteLine("Error: --input <dir> is required (directory containing LK ADT files).");
                Environment.ExitCode = 1;
                return;
            }

            if (string.IsNullOrEmpty(options.OutputPath))
            {
                Console.Error.WriteLine("Error: --output <path> is required (output .wdt file path).");
                Environment.ExitCode = 1;
                return;
            }

            string inputDir = Path.GetFullPath(options.InputDir);
            string outputPath = Path.GetFullPath(options.OutputPath);

            if (!Directory.Exists(inputDir))
            {
                Console.Error.WriteLine($"Error: Directory not found: {inputDir}");
                Environment.ExitCode = 1;
                return;
            }

            Console.WriteLine("WowViewer.Tool.Converter convert-lk-to-alpha report");
            Console.WriteLine($"  Input:    {inputDir}");
            Console.WriteLine($"  Output:   {outputPath}");
            Console.WriteLine($"  Verbose:  {options.Verbose}");

            var sw = Stopwatch.StartNew();

            string mapName = Path.GetFileNameWithoutExtension(outputPath);

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

            var tiles = new Dictionary<(int, int), AlphaTileData>();
            int converted = 0;
            int failed = 0;
            var warnings = new List<string>();

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
            Console.WriteLine($"  Converted: {converted}/{adtFiles.Count} tiles");
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

    private static LkAdtData ReadLkAdt(byte[] adtBytes, int tileX, int tileY)
    {
        using var ms = new MemoryStream(adtBytes, writable: false);
        using var br = new BinaryReader(ms);

        var textureNames = new List<string>();
        var modelNames = new List<string>();
        var worldModelNames = new List<string>();
        var modelPlacements = new List<LkMddfEntry>();
        var worldModelPlacements = new List<LkModfEntry>();
        var chunks = new List<LkMcnkData>(256);
        uint mhdrFlags = 0;

        while (ms.Position + 8 <= ms.Length)
        {
            byte[] tagBytes = br.ReadBytes(4);
            uint size = br.ReadUInt32();
            long chunkEnd = ms.Position + size;
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
                chunks.Add(ReadMcnkChunk(br, (int)size));
            }
            else
            {
                ms.Position = chunkEnd;
            }

            if ((size & 1) != 0 && chunkEnd < ms.Length)
                ms.Position = chunkEnd + 1;
            else
                ms.Position = chunkEnd;
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
            MhdrFlags = mhdrFlags
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

    private static List<LkMddfEntry> ReadMddfEntries(BinaryReader br, int size)
    {
        var entries = new List<LkMddfEntry>();
        int count = size / LkMddfEntrySize;
        for (int i = 0; i < count; i++)
        {
            int nameId = br.ReadInt32();
            int uniqueId = br.ReadInt32();
            float posX = br.ReadSingle();
            float posY = br.ReadSingle();
            float posZ = br.ReadSingle();
            float rotX = br.ReadSingle();
            float rotY = br.ReadSingle();
            float rotZ = br.ReadSingle();
            ushort scale = br.ReadUInt16();
            br.ReadUInt16();
            entries.Add(new LkMddfEntry(nameId, uniqueId,
                new System.Numerics.Vector3(posX, posY, posZ),
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
            float posX = br.ReadSingle();
            float posY = br.ReadSingle();
            float posZ = br.ReadSingle();
            float rotX = br.ReadSingle();
            float rotY = br.ReadSingle();
            float rotZ = br.ReadSingle();
            float bbMinX = br.ReadSingle();
            float bbMinY = br.ReadSingle();
            float bbMinZ = br.ReadSingle();
            float bbMaxX = br.ReadSingle();
            float bbMaxY = br.ReadSingle();
            float bbMaxZ = br.ReadSingle();
            ushort modfFlags = br.ReadUInt16();
            ushort doodadSet = br.ReadUInt16();
            ushort nameSet = br.ReadUInt16();
            ushort modfScale = br.ReadUInt16();
            entries.Add(new LkModfEntry(nameId, uniqueId,
                new System.Numerics.Vector3(posX, posY, posZ),
                new System.Numerics.Vector3(rotX, rotY, rotZ),
                new System.Numerics.Vector3(bbMinX, bbMinY, bbMinZ),
                new System.Numerics.Vector3(bbMaxX, bbMaxY, bbMaxZ),
                modfFlags, doodadSet, nameSet, modfScale / 1024f));
        }
        return entries;
    }

    private static LkMcnkData ReadMcnkChunk(BinaryReader br, int size)
    {
        long chunkStart = br.BaseStream.Position - 8;
        byte[] header = br.ReadBytes(128);

        int mcnkFlags = BitConverter.ToInt32(header, 0x00);
        int indexX = BitConverter.ToInt32(header, 0x04);
        int indexY = BitConverter.ToInt32(header, 0x08);
        int nLayers = BitConverter.ToInt32(header, 0x0C);
        float baseHeight = BitConverter.ToSingle(header, 0x70);
        float posX = BitConverter.ToSingle(header, 0x68);
        float posY = BitConverter.ToSingle(header, 0x6C);
        int areaId = BitConverter.ToInt32(header, 0x34);
        int holeMask = BitConverter.ToInt32(header, 0x3C);

        byte[] heightData = [];
        byte[] normalData = [];
        byte[]? shadowData = null;
        var layers = new List<LkMclyEntry>();
        byte[]? alphaData = null;
        int alphaTotalSize = 0;
        var doodadRefs = new List<int>();
        var worldModelRefs = new List<int>();

        while (br.BaseStream.Position + 8 <= chunkStart + 8 + size)
        {
            if (br.BaseStream.Position + 8 > br.BaseStream.Length) break;
            byte[] subTag = br.ReadBytes(4);
            int subSize = br.ReadInt32();
            long subEnd = br.BaseStream.Position + subSize;
            string subTagStr = System.Text.Encoding.ASCII.GetString(subTag);

            if (subTagStr == "MCVT")
            {
                heightData = br.ReadBytes(Math.Min(subSize, 145 * 4));
                if (heightData.Length < 145 * 4)
                {
                    var padded = new byte[145 * 4];
                    Buffer.BlockCopy(heightData, 0, padded, 0, heightData.Length);
                    heightData = padded;
                }
            }
            else if (subTagStr == "MCNR")
            {
                normalData = br.ReadBytes(Math.Min(subSize, 448));
                if (normalData.Length < 448)
                {
                    var padded = new byte[448];
                    Buffer.BlockCopy(normalData, 0, padded, 0, normalData.Length);
                    normalData = padded;
                }
            }
            else if (subTagStr == "MCLY")
            {
                int layerCount = subSize / 16;
                for (int i = 0; i < layerCount; i++)
                {
                    uint texId = br.ReadUInt32();
                    uint layerMclyFlags = br.ReadUInt32();
                    uint alphaOff = br.ReadUInt32();
                    uint effectId = br.ReadUInt32();
                    layers.Add(new LkMclyEntry(texId, layerMclyFlags, alphaOff, effectId));
                }
            }
            else if (subTagStr == "MCAL")
            {
                alphaData = br.ReadBytes(subSize);
                alphaTotalSize = subSize;
            }
            else if (subTagStr == "MCSH")
            {
                shadowData = br.ReadBytes(subSize);
            }
            else if (subTagStr == "MCRF")
            {
                if (subSize >= 8)
                {
                    int nRefs = br.ReadInt32();
                    for (int i = 0; i < nRefs && br.BaseStream.Position < subEnd; i++)
                        doodadRefs.Add(br.ReadInt32());
                    int nWmoRefs = br.ReadInt32();
                    for (int i = 0; i < nWmoRefs && br.BaseStream.Position < subEnd; i++)
                        worldModelRefs.Add(br.ReadInt32());
                }
            }
            else
            {
                br.BaseStream.Position = subEnd;
            }

            if ((subSize & 1) != 0 && subEnd < br.BaseStream.Length)
                br.BaseStream.Position = subEnd + 1;
            else
                br.BaseStream.Position = subEnd;
        }

        // Skip to end of MCNK if we haven't reached it yet
        long mcnkEnd = chunkStart + 8 + size;
        if (br.BaseStream.Position < mcnkEnd)
            br.BaseStream.Position = mcnkEnd;

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
        bool Verbose);
}