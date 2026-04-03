using System.Text;
using WoWMapConverter.Core.Formats.Liquids;
using WoWMapConverter.Core.Builders;

namespace WoWMapConverter.Core.Converters;

/// <summary>
/// Converts LK 3.3.5 ADT files to Alpha 0.5.3 monolithic WDT format.
/// Handles coordinate systems, MCLQ liquid conversion, and object positioning.
/// </summary>
public class LkToAlphaConverter
{
    private readonly LkToAlphaOptions _options;

    public LkToAlphaConverter(LkToAlphaOptions? options = null)
    {
        _options = options ?? new LkToAlphaOptions();
    }

    /// <summary>
    /// Convert LK ADTs to Alpha monolithic WDT.
    /// </summary>
    /// <param name="lkWdtPath">Path to LK WDT file</param>
    /// <param name="lkMapDir">Directory containing LK ADT files</param>
    /// <param name="outputWdtPath">Output path for Alpha WDT</param>
    /// <param name="ct">Cancellation token</param>
    public async Task<LkToAlphaResult> ConvertAsync(
        string lkWdtPath, 
        string lkMapDir, 
        string outputWdtPath,
        CancellationToken ct = default)
    {
        var result = new LkToAlphaResult { SourceWdtPath = lkWdtPath };
        var sw = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            if (!File.Exists(lkWdtPath))
                throw new FileNotFoundException($"LK WDT not found: {lkWdtPath}");

            Directory.CreateDirectory(Path.GetDirectoryName(outputWdtPath) ?? ".");
            var mapName = Path.GetFileNameWithoutExtension(lkWdtPath);
            result.MapName = mapName;
            result.OutputPath = outputWdtPath;

            // Discover LK ADT files
            var rootAdts = DiscoverRootAdts(lkMapDir, mapName);
            if (_options.Verbose)
                Console.WriteLine($"[lk2alpha] Found {rootAdts.Count} root ADTs");

            result.TotalTiles = rootAdts.Count;

            // Collect all M2/WMO names from ADTs
            var (m2Names, wmoNames) = await CollectAssetNamesAsync(rootAdts, ct);
            if (_options.Verbose)
            {
                Console.WriteLine($"[lk2alpha] M2 names: {m2Names.Count}");
                Console.WriteLine($"[lk2alpha] WMO names: {wmoNames.Count}");
            }

            // Build global name indices
            var mdnmIndex = BuildNameIndex(m2Names);
            var monmIndex = BuildNameIndex(wmoNames);

            // Write Alpha WDT
            using var ms = new MemoryStream();
            
            // MVER (version 18 for Alpha)
            WriteChunk(ms, "MVER", BitConverter.GetBytes(18));

            // MPHD placeholder (128 bytes) - will patch offsets later
            long mphdStart = ms.Position;
            var mphdData = new byte[128];
            WriteChunk(ms, "MPHD", mphdData);
            long mphdDataStart = mphdStart + 8;

            // MAIN placeholder (4096 * 16 bytes)
            var mainData = new byte[4096 * 16];
            long mainStart = ms.Position;
            WriteChunk(ms, "MAIN", mainData);

            // MDNM (M2 names)
            long mdnmStart = ms.Position;
            var mdnmData = BuildNameTableData(m2Names);
            WriteChunkNoPadding(ms, "MDNM", mdnmData);

            // MONM (WMO names)
            long monmStart = ms.Position;
            var monmData = BuildNameTableData(wmoNames);
            WriteChunkNoPadding(ms, "MONM", monmData);

            // Patch MPHD with offsets
            PatchMphd(ms, mphdDataStart, m2Names.Count, (int)mdnmStart, wmoNames.Count, (int)monmStart);

            // Process each tile
            var mhdrOffsets = new int[4096];
            var mhdrToFirstMcnk = new int[4096];
            int tilesConverted = 0;

            foreach (var rootAdt in rootAdts)
            {
                ct.ThrowIfCancellationRequested();

                var (tileX, tileY) = ParseTileCoords(rootAdt, mapName);
                if (tileX < 0 || tileY < 0) continue;

                // Alpha MAIN uses X-major ordering: index = tileX * 64 + tileY
                int tileIndex = tileX * 64 + tileY;

                try
                {
                    long tileStart = ms.Position;
                    
                    // Convert and write tile
                    var tileResult = await ConvertTileAsync(
                        ms, rootAdt, lkMapDir, mapName,
                        tileX, tileY, mdnmIndex, monmIndex, ct);

                    if (tileResult.Success)
                    {
                        mhdrOffsets[tileIndex] = (int)tileStart;
                        mhdrToFirstMcnk[tileIndex] = tileResult.MhdrToFirstMcnkSize;
                        tilesConverted++;

                        if (_options.Verbose)
                            Console.WriteLine($"[lk2alpha] Converted tile {tileY}_{tileX}");
                    }
                }
                catch (Exception ex)
                {
                    if (_options.Verbose)
                        Console.WriteLine($"[lk2alpha] Tile {tileY}_{tileX} failed: {ex.Message}");
                    result.Warnings.Add($"Tile {tileY}_{tileX}: {ex.Message}");
                }
            }

            // Patch MAIN with tile offsets
            ms.Position = mainStart + 8;
            var patchedMain = AlphaMainBuilder.BuildMain(mhdrOffsets, mhdrToFirstMcnk);
            ms.Write(patchedMain);

            // Write output file
            using (var fs = File.Create(outputWdtPath))
            {
                ms.Position = 0;
                await ms.CopyToAsync(fs, ct);
            }

            result.TilesConverted = tilesConverted;
            result.Success = tilesConverted > 0;

            if (!result.Success)
            {
                result.Error = rootAdts.Count == 0
                    ? "No root ADTs were discovered for the input map"
                    : "No LK tiles converted successfully; see warnings for the failing tile diagnostics";
            }

            if (_options.Verbose)
                Console.WriteLine($"[lk2alpha] Complete: {tilesConverted}/{rootAdts.Count} tiles");
        }
        catch (OperationCanceledException)
        {
            result.Success = false;
            result.Error = "Conversion cancelled";
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.Error = ex.Message;
            if (_options.Verbose)
                Console.Error.WriteLine($"[lk2alpha] Error: {ex}");
        }

        sw.Stop();
        result.ElapsedMs = sw.ElapsedMilliseconds;
        return result;
    }

    private List<string> DiscoverRootAdts(string mapDir, string mapName)
    {
        if (!Directory.Exists(mapDir))
            return new List<string>();

        return Directory.EnumerateFiles(mapDir, $"{mapName}_*.adt", SearchOption.TopDirectoryOnly)
            .Where(p => !p.Contains("_obj", StringComparison.OrdinalIgnoreCase) 
                     && !p.Contains("_tex", StringComparison.OrdinalIgnoreCase)
                     && !p.Contains("_lod", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private async Task<(List<string> m2Names, List<string> wmoNames)> CollectAssetNamesAsync(
        List<string> rootAdts, CancellationToken ct)
    {
        var m2Set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var wmoSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        await Task.Run(() =>
        {
            Parallel.ForEach(rootAdts, new ParallelOptions { CancellationToken = ct }, rootAdt =>
            {
                try
                {
                    var bytes = File.ReadAllBytes(rootAdt);
                    foreach (var name in ExtractMmdxNames(bytes))
                        lock (m2Set) m2Set.Add(NormalizePath(name));
                    foreach (var name in ExtractMwmoNames(bytes))
                        lock (wmoSet) wmoSet.Add(NormalizePath(name));

                    // Also check _obj0/_obj1 files
                    var baseName = Path.GetFileNameWithoutExtension(rootAdt);
                    var dir = Path.GetDirectoryName(rootAdt) ?? ".";
                    
                    foreach (var suffix in new[] { "_obj.adt", "_obj0.adt", "_obj1.adt" })
                    {
                        var objPath = Path.Combine(dir, baseName + suffix);
                        if (File.Exists(objPath))
                        {
                            var objBytes = File.ReadAllBytes(objPath);
                            foreach (var name in ExtractMmdxNames(objBytes))
                                lock (m2Set) m2Set.Add(NormalizePath(name));
                            foreach (var name in ExtractMwmoNames(objBytes))
                                lock (wmoSet) wmoSet.Add(NormalizePath(name));
                        }
                    }
                }
                catch { /* best effort */ }
            });
        }, ct);

        return (m2Set.Where(s => !string.IsNullOrWhiteSpace(s)).ToList(),
                wmoSet.Where(s => !string.IsNullOrWhiteSpace(s)).ToList());
    }

    private Dictionary<string, int> BuildNameIndex(List<string> names)
    {
        var index = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < names.Count; i++)
            index[NormalizePath(names[i])] = i;
        return index;
    }

    private async Task<TileConversionResult> ConvertTileAsync(
        MemoryStream ms,
        string rootAdtPath,
        string mapDir,
        string mapName,
        int tileX,
        int tileY,
        Dictionary<string, int> mdnmIndex,
        Dictionary<string, int> monmIndex,
        CancellationToken ct)
    {
        var result = new TileConversionResult();
        var bytes = await File.ReadAllBytesAsync(rootAdtPath, ct);

        // Find LK MHDR and MCIN. Later split roots can omit MCIN and still expose top-level MCNK order.
        int mhdrOffset = FindChunk(bytes, "MHDR");
        if (mhdrOffset < 0)
            throw new InvalidDataException("MHDR not found");

        var mcnkOffsets = ResolveMcnkOffsets(bytes, minimumChunkSize: 128);
        if (!mcnkOffsets.Any(offset => offset > 0))
            throw new InvalidDataException("No root MCNK chunks found");

        // Load optional split texture data. Some 4.x _tex0 files do not contain LK-style MCNK payloads,
        // so only adopt per-chunk offsets when the file exposes valid MCNK-sized chunks.
        byte[]? texBytes = null;
        List<int>? texMcnkOffsets = null;
        var baseName = Path.GetFileNameWithoutExtension(rootAdtPath);
        foreach (var texSuffix in new[] { "_tex0.adt", "_tex.adt" })
        {
            var texPath = Path.Combine(mapDir, baseName + texSuffix);
            if (!File.Exists(texPath))
                continue;

            var candidateBytes = await File.ReadAllBytesAsync(texPath, ct);
            texBytes ??= candidateBytes;

            var candidateOffsets = ResolveMcnkOffsets(candidateBytes, minimumChunkSize: 128);
            if (candidateOffsets.Any(offset => offset > 0))
            {
                texBytes = candidateBytes;
                texMcnkOffsets = candidateOffsets;
                break;
            }
        }

        // Load _obj files for placements
        var objBytes = await LoadObjFilesAsync(mapDir, baseName, ct);

        // Write Alpha tile structure: MHDR -> MCIN -> MTEX -> MDDF -> MODF -> MCNKs
        long mhdrAbsolute = ms.Position;
        var alphaMhdr = AlphaMhdrBuilder.BuildMhdr();
        ms.Write(alphaMhdr);

        // MCIN placeholder
        long mcinPosition = ms.Position;
        var mcinPlaceholder = new byte[256 * 16 + 8]; // chunk header + 256 entries
        WriteChunk(ms, "MCIN", new byte[256 * 16]);

        // MTEX
        var mtexData = ExtractMtexData(texBytes ?? bytes);
        if (mtexData.Length == 0)
            mtexData = Encoding.ASCII.GetBytes(_options.BaseTexture + "\0");
        WriteChunk(ms, "MTEX", mtexData);

        // Build MDDF/MODF from obj files
        var (mddfData, doodadRefs) = BuildMddfFromLk(objBytes, bytes, mdnmIndex, tileX, tileY);
        var (modfData, wmoRefs) = BuildModfFromLk(objBytes, bytes, monmIndex, tileX, tileY);

        WriteChunk(ms, "MDDF", mddfData);
        WriteChunk(ms, "MODF", modfData);

        // Convert and write MCNKs
        var mcnkAbsOffsets = new int[256];
        var mcnkSizes = new int[256];

        for (int i = 0; i < 256; i++)
        {
            int lkOffset = mcnkOffsets[i];
            if (lkOffset == 0)
            {
                mcnkAbsOffsets[i] = 0;
                mcnkSizes[i] = 0;
                continue;
            }

            int texOffset = (texMcnkOffsets != null && i < texMcnkOffsets.Count) ? texMcnkOffsets[i] : 0;
            var dRefs = doodadRefs.TryGetValue(i, out var dr) ? dr : new List<int>();
            var wRefs = wmoRefs.TryGetValue(i, out var wr) ? wr : new List<int>();

            mcnkAbsOffsets[i] = (int)ms.Position;
            var mcnkData = AlphaMcnkBuilder.BuildFromLk(bytes, lkOffset, texBytes, texOffset, dRefs, wRefs, _options);
            ms.Write(mcnkData);
            mcnkSizes[i] = mcnkData.Length;
        }

        // Patch MCIN with actual offsets
        long savePos = ms.Position;
        ms.Position = mcinPosition + 8; // skip chunk header
        for (int i = 0; i < 256; i++)
        {
            ms.Write(BitConverter.GetBytes(mcnkAbsOffsets[i]));
            ms.Write(BitConverter.GetBytes(mcnkSizes[i]));
            ms.Write(new byte[8]); // flags + pad
        }
        ms.Position = savePos;

        result.Success = true;
        result.MhdrToFirstMcnkSize = mcnkAbsOffsets.FirstOrDefault(o => o > 0) > 0 
            ? mcnkAbsOffsets.First(o => o > 0) - (int)mhdrAbsolute 
            : 0;

        return result;
    }

    private async Task<Dictionary<string, byte[]>> LoadObjFilesAsync(string mapDir, string baseName, CancellationToken ct)
    {
        var result = new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase);
        
        foreach (var suffix in new[] { "_obj.adt", "_obj0.adt", "_obj1.adt" })
        {
            var path = Path.Combine(mapDir, baseName + suffix);
            if (File.Exists(path))
                result[suffix] = await File.ReadAllBytesAsync(path, ct);
        }

        return result;
    }

    private (byte[] data, Dictionary<int, List<int>> perChunkRefs) BuildMddfFromLk(
        Dictionary<string, byte[]> objFiles,
        byte[] rootBytes,
        Dictionary<string, int> mdnmIndex,
        int tileX,
        int tileY)
    {
        using var ms = new MemoryStream();
        var perChunkRefs = new Dictionary<int, List<int>>();
        for (int i = 0; i < 256; i++) perChunkRefs[i] = new List<int>();

        int baseIndex = 0;

        // Process obj0, obj, then root (priority order)
        foreach (var key in new[] { "_obj0.adt", "_obj.adt" })
        {
            if (objFiles.TryGetValue(key, out var objBytes))
            {
                var mmdxNames = ExtractMmdxNames(objBytes).ToList();
                baseIndex = WriteMddfEntries(ms, objBytes, mmdxNames, mdnmIndex, tileX, tileY, perChunkRefs, baseIndex);
            }
        }

        // Root ADT
        var rootMmdx = ExtractMmdxNames(rootBytes).ToList();
        WriteMddfEntries(ms, rootBytes, rootMmdx, mdnmIndex, tileX, tileY, perChunkRefs, baseIndex);

        return (ms.ToArray(), perChunkRefs);
    }

    private int WriteMddfEntries(
        MemoryStream ms,
        byte[] bytes,
        List<string> mmdxNames,
        Dictionary<string, int> mdnmIndex,
        int tileX,
        int tileY,
        Dictionary<int, List<int>> perChunkRefs,
        int baseIndex)
    {
        int mddfOffset = FindChunk(bytes, "MDDF");
        if (mddfOffset < 0) return baseIndex;

        int size = BitConverter.ToInt32(bytes, mddfOffset + 4);
        int dataStart = mddfOffset + 8;
        const int entrySize = 36;
        int count = size / entrySize;
        int written = 0;

        for (int i = 0; i < count; i++)
        {
            int p = dataStart + i * entrySize;
            if (p + entrySize > bytes.Length) break;

            int localIdx = BitConverter.ToInt32(bytes, p);
            if (localIdx < 0 || localIdx >= mmdxNames.Count) continue;

            string name = NormalizePath(mmdxNames[localIdx]);
            if (!mdnmIndex.TryGetValue(name, out int globalIdx)) continue;

            // Read LK MDDF entry — positions are (X, Y, Z) per spec
            int uniqueId = BitConverter.ToInt32(bytes, p + 4);
            float posX = BitConverter.ToSingle(bytes, p + 8);   // position[0] = X (North)
            float posY = BitConverter.ToSingle(bytes, p + 12);  // position[1] = Y (West)
            float posZ = BitConverter.ToSingle(bytes, p + 16);  // position[2] = Z (Up/height)
            float rotX = BitConverter.ToSingle(bytes, p + 20);
            float rotY = BitConverter.ToSingle(bytes, p + 24);
            float rotZ = BitConverter.ToSingle(bytes, p + 28);
            ushort scale = BitConverter.ToUInt16(bytes, p + 32);
            ushort flags = BitConverter.ToUInt16(bytes, p + 34);

            // Write Alpha MDDF entry (same format, positions unchanged)
            ms.Write(BitConverter.GetBytes(globalIdx));
            ms.Write(BitConverter.GetBytes(uniqueId));
            ms.Write(BitConverter.GetBytes(posX));
            ms.Write(BitConverter.GetBytes(posY));
            ms.Write(BitConverter.GetBytes(posZ));
            ms.Write(BitConverter.GetBytes(rotX));
            ms.Write(BitConverter.GetBytes(rotY));
            ms.Write(BitConverter.GetBytes(rotZ));
            ms.Write(BitConverter.GetBytes(scale));
            ms.Write(BitConverter.GetBytes(flags));

            // Calculate chunk index for MCRF using horizontal axes (X, Y)
            int chunkIdx = CalculateChunkIndex(posX, posY, tileX, tileY);
            if (chunkIdx >= 0 && chunkIdx < 256)
                perChunkRefs[chunkIdx].Add(baseIndex + written);

            written++;
        }

        return baseIndex + written;
    }

    private (byte[] data, Dictionary<int, List<int>> perChunkRefs) BuildModfFromLk(
        Dictionary<string, byte[]> objFiles,
        byte[] rootBytes,
        Dictionary<string, int> monmIndex,
        int tileX,
        int tileY)
    {
        using var ms = new MemoryStream();
        var perChunkRefs = new Dictionary<int, List<int>>();
        for (int i = 0; i < 256; i++) perChunkRefs[i] = new List<int>();

        int baseIndex = 0;

        // Process obj1, obj, then root (priority order for WMOs)
        foreach (var key in new[] { "_obj1.adt", "_obj.adt" })
        {
            if (objFiles.TryGetValue(key, out var objBytes))
            {
                var mwmoNames = ExtractMwmoNames(objBytes).ToList();
                baseIndex = WriteModfEntries(ms, objBytes, mwmoNames, monmIndex, tileX, tileY, perChunkRefs, baseIndex);
            }
        }

        // Root ADT
        var rootMwmo = ExtractMwmoNames(rootBytes).ToList();
        WriteModfEntries(ms, rootBytes, rootMwmo, monmIndex, tileX, tileY, perChunkRefs, baseIndex);

        return (ms.ToArray(), perChunkRefs);
    }

    private int WriteModfEntries(
        MemoryStream ms,
        byte[] bytes,
        List<string> mwmoNames,
        Dictionary<string, int> monmIndex,
        int tileX,
        int tileY,
        Dictionary<int, List<int>> perChunkRefs,
        int baseIndex)
    {
        int modfOffset = FindChunk(bytes, "MODF");
        if (modfOffset < 0) return baseIndex;

        int size = BitConverter.ToInt32(bytes, modfOffset + 4);
        int dataStart = modfOffset + 8;
        const int entrySize = 64;
        int count = size / entrySize;
        int written = 0;

        for (int i = 0; i < count; i++)
        {
            int p = dataStart + i * entrySize;
            if (p + entrySize > bytes.Length) break;

            int localIdx = BitConverter.ToInt32(bytes, p);
            if (localIdx < 0 || localIdx >= mwmoNames.Count) continue;

            string name = NormalizePath(mwmoNames[localIdx]);
            if (!monmIndex.TryGetValue(name, out int globalIdx)) continue;

            // Read LK MODF entry — positions are (X, Y, Z) per spec
            int uniqueId = BitConverter.ToInt32(bytes, p + 4);
            float posX = BitConverter.ToSingle(bytes, p + 8);   // position[0] = X (North)
            float posY = BitConverter.ToSingle(bytes, p + 12);  // position[1] = Y (West)
            float posZ = BitConverter.ToSingle(bytes, p + 16);  // position[2] = Z (Up/height)
            float rotX = BitConverter.ToSingle(bytes, p + 20);
            float rotY = BitConverter.ToSingle(bytes, p + 24);
            float rotZ = BitConverter.ToSingle(bytes, p + 28);
            
            // Extents — (X, Y, Z) order per spec
            float minX = BitConverter.ToSingle(bytes, p + 32);  // extentsLower[0]
            float minY = BitConverter.ToSingle(bytes, p + 36);  // extentsLower[1]
            float minZ = BitConverter.ToSingle(bytes, p + 40);  // extentsLower[2]
            float maxX = BitConverter.ToSingle(bytes, p + 44);  // extentsUpper[0]
            float maxY = BitConverter.ToSingle(bytes, p + 48);  // extentsUpper[1]
            float maxZ = BitConverter.ToSingle(bytes, p + 52);  // extentsUpper[2]
            
            ushort flags = BitConverter.ToUInt16(bytes, p + 56);
            ushort doodadSet = BitConverter.ToUInt16(bytes, p + 58);
            ushort nameSet = BitConverter.ToUInt16(bytes, p + 60);
            ushort scale = BitConverter.ToUInt16(bytes, p + 62);

            // Write Alpha MODF entry (same format, byte-for-byte copy)
            ms.Write(BitConverter.GetBytes(globalIdx));
            ms.Write(BitConverter.GetBytes(uniqueId));
            ms.Write(BitConverter.GetBytes(posX));
            ms.Write(BitConverter.GetBytes(posY));
            ms.Write(BitConverter.GetBytes(posZ));
            ms.Write(BitConverter.GetBytes(rotX));
            ms.Write(BitConverter.GetBytes(rotY));
            ms.Write(BitConverter.GetBytes(rotZ));
            ms.Write(BitConverter.GetBytes(minX));
            ms.Write(BitConverter.GetBytes(minY));
            ms.Write(BitConverter.GetBytes(minZ));
            ms.Write(BitConverter.GetBytes(maxX));
            ms.Write(BitConverter.GetBytes(maxY));
            ms.Write(BitConverter.GetBytes(maxZ));
            ms.Write(BitConverter.GetBytes(flags));
            ms.Write(BitConverter.GetBytes(doodadSet));
            ms.Write(BitConverter.GetBytes(nameSet));
            ms.Write(BitConverter.GetBytes(scale));

            // Calculate chunk index for MCRF using horizontal axes (X, Y)
            int chunkIdx = CalculateChunkIndex(posX, posY, tileX, tileY);
            if (chunkIdx >= 0 && chunkIdx < 256)
                perChunkRefs[chunkIdx].Add(baseIndex + written);

            written++;
        }

        return baseIndex + written;
    }

    private int CalculateChunkIndex(float posX, float posY, int tileX, int tileY)
    {
        const float TileSize = 533.33333f;
        const float ChunkSize = TileSize / 16f;

        float tileMinX = 32f * TileSize - (tileX + 1) * TileSize;
        float tileMinY = 32f * TileSize - (tileY + 1) * TileSize;

        float localX = posX - tileMinX;
        float localY = posY - tileMinY;

        int cx = (int)Math.Floor(localX / ChunkSize);
        int cy = (int)Math.Floor(localY / ChunkSize);

        if (cx >= 0 && cx < 16 && cy >= 0 && cy < 16)
            return cy * 16 + cx;

        return -1;
    }

    #region Chunk Helpers

    private static int FindChunk(byte[] bytes, string fourCC)
    {
        // LK uses reversed FourCC on disk
        string reversed = new string(fourCC.Reverse().ToArray());
        
        for (int i = 0; i + 8 <= bytes.Length;)
        {
            string fcc = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            if (size < 0)
                break;
            
            if (fcc == reversed)
                return i;

            int next = AdvanceChunkPosition(bytes, i, size);
            if (next <= i) break;
            i = next;
        }

        return -1;
    }

    private static List<int> ReadMcinOffsets(byte[] bytes, int mcinOffset)
    {
        var offsets = new List<int>(256);
        int dataStart = mcinOffset + 8; // skip chunk header

        for (int i = 0; i < 256; i++)
        {
            int entryOffset = dataStart + i * 16;
            if (entryOffset + 4 > bytes.Length) break;
            offsets.Add(BitConverter.ToInt32(bytes, entryOffset));
        }

        while (offsets.Count < 256)
            offsets.Add(0);

        return offsets;
    }

    private static List<int> ResolveMcnkOffsets(byte[] bytes, int minimumChunkSize)
    {
        int mhdrOffset = FindChunk(bytes, "MHDR");
        if (mhdrOffset >= 0)
        {
            int mhdrDataStart = mhdrOffset + 8;
            int mcinRelOffset = BitConverter.ToInt32(bytes, mhdrDataStart);
            if (mcinRelOffset > 0)
            {
                int mcinOffset = mhdrDataStart + mcinRelOffset;
                var mcinOffsets = NormalizeMcnkOffsets(bytes, ReadMcinOffsets(bytes, mcinOffset), minimumChunkSize);
                if (mcinOffsets.Any(offset => offset > 0))
                    return mcinOffsets;
            }
        }

        return PadMcnkOffsets(ReadTopLevelMcnkOffsets(bytes, minimumChunkSize));
    }

    private static List<int> ReadTopLevelMcnkOffsets(byte[] bytes, int minimumChunkSize)
    {
        var offsets = new List<int>(256);
        int position = 0;

        while (position + 8 <= bytes.Length)
        {
            string signature = Encoding.ASCII.GetString(bytes, position, 4);
            int size = BitConverter.ToInt32(bytes, position + 4);
            if (size < 0)
                break;

            if ((signature == "KNCM" || signature == "MCNK") && size >= minimumChunkSize && position + 8 + size <= bytes.Length)
                offsets.Add(position);

            int next = AdvanceChunkPosition(bytes, position, size);
            if (next <= position)
                break;

            position = next;
        }

        return offsets;
    }

    private static List<int> NormalizeMcnkOffsets(byte[] bytes, List<int> offsets, int minimumChunkSize)
    {
        var normalized = new List<int>(256);
        for (int i = 0; i < 256; i++)
        {
            int offset = i < offsets.Count ? offsets[i] : 0;
            normalized.Add(IsValidMcnkOffset(bytes, offset, minimumChunkSize) ? offset : 0);
        }

        return normalized;
    }

    private static bool IsValidMcnkOffset(byte[] bytes, int offset, int minimumChunkSize)
    {
        if (offset <= 0 || offset + 8 > bytes.Length)
            return false;

        string signature = Encoding.ASCII.GetString(bytes, offset, 4);
        if (signature != "KNCM" && signature != "MCNK")
            return false;

        int size = BitConverter.ToInt32(bytes, offset + 4);
        return size >= minimumChunkSize && offset + 8 + size <= bytes.Length;
    }

    private static int AdvanceChunkPosition(byte[] bytes, int position, int size)
    {
        int unpadded = position + 8 + size;
        if (unpadded <= position)
            return -1;

        if ((size & 1) == 0)
            return unpadded;

        int padded = unpadded + 1;
        bool unpaddedLooksValid = LooksLikeChunkHeader(bytes, unpadded);
        bool paddedLooksValid = LooksLikeChunkHeader(bytes, padded);

        if (unpaddedLooksValid && !paddedLooksValid)
            return unpadded;

        if (paddedLooksValid && !unpaddedLooksValid)
            return padded;

        if (unpaddedLooksValid)
            return unpadded;

        if (padded <= bytes.Length)
            return padded;

        return unpadded <= bytes.Length ? unpadded : -1;
    }

    private static bool LooksLikeChunkHeader(byte[] bytes, int position)
    {
        if (position < 0 || position + 8 > bytes.Length)
            return false;

        for (int i = 0; i < 4; i++)
        {
            byte value = bytes[position + i];
            bool isUppercase = value >= (byte)'A' && value <= (byte)'Z';
            bool isDigit = value >= (byte)'0' && value <= (byte)'9';
            if (!isUppercase && !isDigit && value != (byte)'_')
                return false;
        }

        int size = BitConverter.ToInt32(bytes, position + 4);
        return size >= 0;
    }

    private static List<int> PadMcnkOffsets(List<int> offsets)
    {
        if (offsets.Count >= 256)
            return offsets;

        var padded = new List<int>(offsets);
        while (padded.Count < 256)
            padded.Add(0);

        return padded;
    }

    private static void WriteChunk(MemoryStream ms, string fourCC, byte[] data)
    {
        // Alpha uses reversed FourCC
        var fccBytes = Encoding.ASCII.GetBytes(new string(fourCC.Reverse().ToArray()));
        ms.Write(fccBytes, 0, 4);
        ms.Write(BitConverter.GetBytes(data.Length));
        ms.Write(data);
        
        // Pad to even boundary
        if ((data.Length & 1) == 1)
            ms.WriteByte(0);
    }

    private static void WriteChunkNoPadding(MemoryStream ms, string fourCC, byte[] data)
    {
        var fccBytes = Encoding.ASCII.GetBytes(new string(fourCC.Reverse().ToArray()));
        ms.Write(fccBytes, 0, 4);
        ms.Write(BitConverter.GetBytes(data.Length));
        ms.Write(data);
    }

    private static void PatchMphd(MemoryStream ms, long mphdDataStart, int m2Count, int mdnmOffset, int wmoCount, int monmOffset)
    {
        long savePos = ms.Position;
        ms.Position = mphdDataStart;

        var data = new byte[128];
        // MPHD layout: [0..3]=nTextures (M2 count+1), [4..7]=MDNM abs, [8..11]=nMapObjNames (WMO count+1), [12..15]=MONM abs
        BitConverter.GetBytes(m2Count > 0 ? m2Count + 1 : 0).CopyTo(data, 0);
        BitConverter.GetBytes(mdnmOffset).CopyTo(data, 4);
        BitConverter.GetBytes(wmoCount > 0 ? wmoCount + 1 : 0).CopyTo(data, 8);
        BitConverter.GetBytes(monmOffset).CopyTo(data, 12);

        ms.Write(data);
        ms.Position = savePos;
    }

    #endregion

    #region Name Extraction

    private static IEnumerable<string> ExtractMmdxNames(byte[] bytes)
    {
        int offset = FindChunk(bytes, "MMDX");
        if (offset < 0) yield break;

        int size = BitConverter.ToInt32(bytes, offset + 4);
        if (size <= 0) yield break;

        int dataStart = offset + 8;
        if (dataStart > bytes.Length) yield break;

        int end = Math.Min(dataStart + size, bytes.Length);
        if (end <= dataStart) yield break;

        int pos = dataStart;
        while (pos < end)
        {
            int remaining = end - pos;
            if (remaining <= 0)
                yield break;

            int nul = Array.IndexOf(bytes, (byte)0, pos, remaining);
            if (nul == -1) nul = end;
            int len = nul - pos;
            if (len > 0)
                yield return Encoding.UTF8.GetString(bytes, pos, len);
            pos = nul + 1;
        }
    }

    private static IEnumerable<string> ExtractMwmoNames(byte[] bytes)
    {
        int offset = FindChunk(bytes, "MWMO");
        if (offset < 0) yield break;

        int size = BitConverter.ToInt32(bytes, offset + 4);
        if (size <= 0) yield break;

        int dataStart = offset + 8;
        if (dataStart > bytes.Length) yield break;

        int end = Math.Min(dataStart + size, bytes.Length);
        if (end <= dataStart) yield break;

        int pos = dataStart;
        while (pos < end)
        {
            int remaining = end - pos;
            if (remaining <= 0)
                yield break;

            int nul = Array.IndexOf(bytes, (byte)0, pos, remaining);
            if (nul == -1) nul = end;
            int len = nul - pos;
            if (len > 0)
                yield return Encoding.UTF8.GetString(bytes, pos, len);
            pos = nul + 1;
        }
    }

    private static byte[] ExtractMtexData(byte[] bytes)
    {
        int offset = FindChunk(bytes, "MTEX");
        if (offset < 0) return Array.Empty<byte>();

        int size = BitConverter.ToInt32(bytes, offset + 4);
        if (size <= 0) return Array.Empty<byte>();

        var data = new byte[size];
        Buffer.BlockCopy(bytes, offset + 8, data, 0, size);
        return data;
    }

    private static byte[] BuildNameTableData(List<string> names)
    {
        if (names == null || names.Count == 0)
            return new byte[] { 0 }; // Empty string list terminator

        using var ms = new MemoryStream();
        foreach (var name in names)
        {
            var nameBytes = Encoding.ASCII.GetBytes(NormalizePath(name));
            ms.Write(nameBytes);
            ms.WriteByte(0);
        }
        ms.WriteByte(0); // Extra terminator
        return ms.ToArray();
    }

    private static string NormalizePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) return string.Empty;
        return path.Replace('/', '\\').TrimStart('\\');
    }

    private static (int x, int y) ParseTileCoords(string adtPath, string mapName)
    {
        var fileName = Path.GetFileNameWithoutExtension(adtPath);
        var parts = fileName.Split('_');
        
        if (parts.Length >= 3 &&
            int.TryParse(parts[^2], out int y) &&
            int.TryParse(parts[^1], out int x))
        {
            return (x, y);
        }

        return (-1, -1);
    }

    #endregion
}

/// <summary>
/// Options for LK to Alpha conversion.
/// </summary>
public class LkToAlphaOptions
{
    /// <summary>Verbose logging output.</summary>
    public bool Verbose { get; set; } = false;

    /// <summary>Base texture path for chunks without textures.</summary>
    public string BaseTexture { get; set; } = "Tileset\\Generic\\Checkers.blp";

    /// <summary>Skip M2 doodad placements.</summary>
    public bool SkipM2 { get; set; } = false;

    /// <summary>Skip WMO placements.</summary>
    public bool SkipWmo { get; set; } = false;

    /// <summary>Convert MH2O to MCLQ for liquid support.</summary>
    public bool ConvertLiquids { get; set; } = true;
}

/// <summary>
/// Result of LK to Alpha conversion.
/// </summary>
public class LkToAlphaResult
{
    public bool Success { get; set; }
    public string? SourceWdtPath { get; set; }
    public string? MapName { get; set; }
    public string? OutputPath { get; set; }
    public string? Error { get; set; }
    public long ElapsedMs { get; set; }
    public int TilesConverted { get; set; }
    public int TotalTiles { get; set; }
    public List<string> Warnings { get; set; } = new();
}

internal class TileConversionResult
{
    public bool Success { get; set; }
    public int MhdrToFirstMcnkSize { get; set; }
}
