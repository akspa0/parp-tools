using WoWMapConverter.Core.Services;
using GillijimProject.WowFiles.Alpha;
using static WoWMapConverter.Core.Services.AdtAreaIdPatcher;

namespace WoWMapConverter.Core.Converters;

/// <summary>
/// Converts Alpha WDT/ADT files to LK 3.3.5 format.
/// Supports AreaID crosswalk and asset path fixups.
/// </summary>
public class AlphaToLkConverter
{
    private readonly AreaIdCrosswalk? _areaCrosswalk;
    private readonly ListfileService? _listfileService;
    private readonly ConversionOptions _options;

    public AlphaToLkConverter(ConversionOptions? options = null)
    {
        _options = options ?? new ConversionOptions();
        
        // Always instantiate crosswalk to load embedded defaults
        _areaCrosswalk = new AreaIdCrosswalk();

        if (!string.IsNullOrEmpty(_options.CrosswalkDirectory))
        {
            _areaCrosswalk.LoadFromDirectory(_options.CrosswalkDirectory);
        }

        if (!string.IsNullOrEmpty(_options.CommunityListfile))
        {
            _listfileService = new ListfileService();
            _listfileService.LoadCommunityListfile(_options.CommunityListfile);
            
            if (!string.IsNullOrEmpty(_options.LkListfile))
            {
                _listfileService.LoadLkListfile(_options.LkListfile);
            }
        }
    }

    /// <summary>
    /// Convert an Alpha WDT to LK format.
    /// </summary>
    public async Task<ConversionResult> ConvertWdtAsync(string wdtPath, string outputDir, CancellationToken ct = default)
    {
        var result = new ConversionResult { SourcePath = wdtPath };
        var sw = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            if (!File.Exists(wdtPath))
                throw new FileNotFoundException($"WDT not found: {wdtPath}");

            Directory.CreateDirectory(outputDir);
            var mapName = Path.GetFileNameWithoutExtension(wdtPath);
            result.MapName = mapName;
            result.OutputDirectory = outputDir;

            // Parse Alpha WDT
            if (_options.Verbose)
                Console.WriteLine($"Parsing Alpha WDT: {wdtPath}");

            var wdtAlpha = new WdtAlpha(wdtPath);
            var existingTiles = wdtAlpha.GetExistingAdtsNumbers();
            var adtOffsets = wdtAlpha.GetAdtOffsetsInMain();
            var mdnmNames = wdtAlpha.GetMdnmFileNames();
            var monmNames = wdtAlpha.GetMonmFileNames();

            result.TotalTiles = existingTiles.Count;

            if (_options.Verbose)
            {
                Console.WriteLine($"  Found {existingTiles.Count} tiles");
                Console.WriteLine($"  MDNM entries: {mdnmNames.Count}");
                Console.WriteLine($"  MONM entries: {monmNames.Count}");
            }

            // Convert WDT to LK format and write
            // FIX: Write WDT to same directory as ADTs, typically named {mapName}.wdt
            // User requested .wdt_new to coexist? Or just standard WDT alongside?
            // "the .wdt_new file needs to coexist in the map data folder, not in a separate folder called <map>.wdt"
            // Interpreting as: Output should be directly in outputDir, named {mapName}.wdt
            var wdtLk = wdtAlpha.ToWdt();
            var wdtOutPath = Path.Combine(outputDir, $"{mapName}.wdt"); 
            wdtLk.ToExactFile(wdtOutPath);
            
            // Also generate WDL (Low resolution heightmap/markers)
            // Extract heights from Alpha ADT chunks for proper terrain horizon
            var adtOffsetsDict = new Dictionary<int, int>();
            foreach (var tileNum in existingTiles)
            {
                adtOffsetsDict[tileNum] = adtOffsets[tileNum];
            }
            GenerateWdlWithHeights(wdtPath, existingTiles, adtOffsetsDict, Path.Combine(outputDir, $"{mapName}.wdl"));


            if (_options.Verbose)
                Console.WriteLine($"  Wrote WDT: {wdtOutPath}");

            // Convert each ADT tile
            int converted = 0;
            foreach (var tileNum in existingTiles)
            {
                ct.ThrowIfCancellationRequested();

                int offset = adtOffsets[tileNum];
                if (offset == 0) continue;

                int x = tileNum % 64;
                int y = tileNum / 64;

                if (_options.Verbose)
                    Console.WriteLine($"  Converting tile {x}_{y} (#{tileNum})...");

                // Parse Alpha ADT from WDT
                var adtAlpha = new AdtAlpha(wdtPath, offset, tileNum);

                // Convert to LK ADT
                var adtLk = adtAlpha.ToAdtLk(mdnmNames, monmNames);

                // Write LK ADT
                var adtOutPath = Path.Combine(outputDir, $"{mapName}_{x}_{y}.adt");
                adtLk.ToFile(adtOutPath);

                converted++;
            }

            result.TilesConverted = converted;

            // Apply AreaID crosswalk if available
            if (_areaCrosswalk != null && _areaCrosswalk.HasMapData(mapName))
            {
                if (_options.Verbose)
                    Console.WriteLine($"Applying AreaID crosswalk for {mapName}...");

                var (filesPatched, chunksPatched) = AdtAreaIdPatcher.PatchDirectory(
                    outputDir, 
                    mapName, 
                    MapAreaId,
                    _options.Verbose);

                if (_options.Verbose)
                    Console.WriteLine($"  Patched {chunksPatched} chunks in {filesPatched} files");
            }

            // Convert WMO v14 files if enabled
            if (_options.ConvertWmos && !string.IsNullOrEmpty(_options.AlphaWmoDirectory))
            {
                if (_options.Verbose)
                    Console.WriteLine($"Converting WMO files from: {_options.AlphaWmoDirectory}");

                var wmoPathMapping = ConvertWmoFiles(monmNames, outputDir);
                
                if (_options.Verbose)
                    Console.WriteLine($"  Converted {wmoPathMapping.Count} WMO files");
                
                // TODO: Patch ADT MWMO chunks with remapped paths
                // For now, log the mappings for manual verification
                foreach (var kvp in wmoPathMapping)
                {
                    result.Warnings.Add($"WMO: {kvp.Key} → {kvp.Value}");
                }
            }

            result.Success = true;

            if (_options.Verbose)
                Console.WriteLine($"Conversion complete: {converted}/{existingTiles.Count} tiles");
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
                Console.Error.WriteLine($"Error: {ex}");
        }

        sw.Stop();
        result.ElapsedMs = sw.ElapsedMilliseconds;
        return result;
    }

    /// <summary>
    /// Generate a WDL file for the map by extracting heights from Alpha ADT chunks.
    /// Based on noggit-red map_horizon.cpp save_wdl implementation.
    /// WDL stores 17x17 outer + 16x16 inner heights per tile as int16.
    /// </summary>
    private void GenerateWdl(List<int> existingTiles, string wdlPath)
    {
        var tileFlags = new bool[64, 64];
        var tileHeights = new Dictionary<int, short[]>(); // tileNum -> 545 heights
        foreach (var t in existingTiles)
        {
            tileFlags[t / 64, t % 64] = true;
        }

        using var fs = new FileStream(wdlPath, FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs);

        // MVER -> REVM
        bw.Write(System.Text.Encoding.ASCII.GetBytes("REVM"));
        bw.Write(4);
        bw.Write(18);

        // MWMO -> OWMM
        bw.Write(System.Text.Encoding.ASCII.GetBytes("OWMM"));
        bw.Write(0);

        // MWID -> DIWM
        bw.Write(System.Text.Encoding.ASCII.GetBytes("DIWM"));
        bw.Write(0);

        // MODF -> FDOM
        bw.Write(System.Text.Encoding.ASCII.GetBytes("FDOM"));
        bw.Write(0);

        // MAOF -> FOAM
        bw.Write(System.Text.Encoding.ASCII.GetBytes("FOAM"));
        bw.Write(64 * 64 * 4); // Size
        
        // Calculate offsets
        uint currentOffset = 12 + 8 + 8 + 8 + 8 + (64 * 64 * 4);
        
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                if (tileFlags[y, x])
                {
                    bw.Write(currentOffset);
                    currentOffset += 1098 + 40; // MARE + MAHO
                }
                else
                {
                    bw.Write(0u);
                }
            }
        }

        // Write MARE and MAHO chunks for existing tiles
        var zeroHoles = new byte[32];

        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                int tileNum = y * 64 + x;
                if (tileFlags[y, x])
                {
                    // Get heights for this tile (computed by caller in existingTiles context)
                    short[] heights;
                    if (!tileHeights.TryGetValue(tileNum, out heights!))
                    {
                        heights = new short[545]; // zeros if not computed
                    }

                    // MARE -> ERAM (17x17 outer + 16x16 inner = 545 int16 = 1090 bytes)
                    bw.Write(System.Text.Encoding.ASCII.GetBytes("ERAM"));
                    bw.Write(1090);
                    for (int i = 0; i < 545; i++)
                    {
                        bw.Write(heights[i]);
                    }

                    // MAHO -> OHAM
                    bw.Write(System.Text.Encoding.ASCII.GetBytes("OHAM"));
                    bw.Write(32);
                    bw.Write(zeroHoles);
                }
            }
        }
    }

    /// <summary>
    /// Generate a WDL file for the map by extracting heights from Alpha ADT chunks.
    /// This overload takes the WDT path to read chunk heights directly.
    /// Based on noggit-red map_horizon.cpp save_wdl implementation.
    /// </summary>
    private void GenerateWdlWithHeights(string wdtPath, List<int> existingTiles, Dictionary<int, int> adtOffsets, string wdlPath)
    {
        var tileFlags = new bool[64, 64];
        var tileHeights = new Dictionary<int, short[]>();
        
        foreach (var t in existingTiles)
        {
            tileFlags[t / 64, t % 64] = true;
        }

        // Extract heights from Alpha ADT chunks
        using (var wdtStream = File.OpenRead(wdtPath))
        {
            foreach (var tileNum in existingTiles)
            {
                if (!adtOffsets.TryGetValue(tileNum, out int offset) || offset == 0)
                    continue;

                try
                {
                    var heights = ExtractTileHeights(wdtStream, offset);
                    if (heights != null)
                    {
                        tileHeights[tileNum] = heights;
                    }
                }
                catch
                {
                    // Skip tiles that fail to parse
                }
            }
        }

        using var fs = new FileStream(wdlPath, FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs);

        // MVER -> REVM
        bw.Write(System.Text.Encoding.ASCII.GetBytes("REVM"));
        bw.Write(4);
        bw.Write(18);

        // MWMO -> OWMM
        bw.Write(System.Text.Encoding.ASCII.GetBytes("OWMM"));
        bw.Write(0);

        // MWID -> DIWM
        bw.Write(System.Text.Encoding.ASCII.GetBytes("DIWM"));
        bw.Write(0);

        // MODF -> FDOM
        bw.Write(System.Text.Encoding.ASCII.GetBytes("FDOM"));
        bw.Write(0);

        // MAOF -> FOAM
        bw.Write(System.Text.Encoding.ASCII.GetBytes("FOAM"));
        bw.Write(64 * 64 * 4);
        
        uint currentOffset = 12 + 8 + 8 + 8 + 8 + (64 * 64 * 4);
        
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                if (tileFlags[y, x])
                {
                    bw.Write(currentOffset);
                    currentOffset += 1098 + 40;
                }
                else
                {
                    bw.Write(0u);
                }
            }
        }

        var zeroHoles = new byte[32];

        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                int tileNum = y * 64 + x;
                if (tileFlags[y, x])
                {
                    short[] heights;
                    if (!tileHeights.TryGetValue(tileNum, out heights!))
                    {
                        heights = new short[545];
                    }

                    bw.Write(System.Text.Encoding.ASCII.GetBytes("ERAM"));
                    bw.Write(1090);
                    for (int i = 0; i < 545; i++)
                    {
                        bw.Write(heights[i]);
                    }

                    bw.Write(System.Text.Encoding.ASCII.GetBytes("OHAM"));
                    bw.Write(32);
                    bw.Write(zeroHoles);
                }
            }
        }

        if (_options.Verbose)
        {
            Console.WriteLine($"  Generated WDL with heights for {tileHeights.Count} tiles");
        }
    }

    /// <summary>
    /// Extract WDL-resolution heights (17x17 outer + 16x16 inner) from an Alpha ADT tile.
    /// Alpha MCVT has 145 floats per chunk (81 outer 9x9 + 64 inner 8x8), laid out as outer then inner.
    /// We downsample 16x16 chunks to 17x17 + 16x16 per tile by picking corner/edge values.
    /// </summary>
    private short[]? ExtractTileHeights(FileStream wdtStream, int adtOffset)
    {
        const int McnkHeaderSize = 128;
        const int ChunkLettersAndSize = 8;
        const int McvtFloatCount = 145; // 81 outer + 64 inner

        // Read ADT MHDR to find MCIN
        wdtStream.Seek(adtOffset, SeekOrigin.Begin);
        var mhdrBytes = new byte[ChunkLettersAndSize + 64];
        if (wdtStream.Read(mhdrBytes, 0, mhdrBytes.Length) < mhdrBytes.Length)
            return null;

        // Parse MCIN offset from MHDR (first 4 bytes of MHDR data)
        int mcinOffset = BitConverter.ToInt32(mhdrBytes, ChunkLettersAndSize);
        int mhdrDataStart = adtOffset + ChunkLettersAndSize;

        // Read MCIN to get MCNK offsets
        wdtStream.Seek(mhdrDataStart + mcinOffset, SeekOrigin.Begin);
        var mcinHeader = new byte[ChunkLettersAndSize];
        wdtStream.Read(mcinHeader, 0, mcinHeader.Length);
        int mcinSize = BitConverter.ToInt32(mcinHeader, 4);
        
        var mcinData = new byte[mcinSize];
        wdtStream.Read(mcinData, 0, mcinSize);

        // Each MCIN entry is 16 bytes: offset(4), size(4), flags(4), asyncId(4)
        var mcnkOffsets = new int[256];
        for (int i = 0; i < 256 && i * 16 + 4 <= mcinSize; i++)
        {
            mcnkOffsets[i] = BitConverter.ToInt32(mcinData, i * 16);
        }

        // Aggregate tile heights: 129x129 for outer grid, 128x128 for inner grid
        var outer129 = new float[129, 129];
        var inner128 = new float[128, 128];

        for (int chunkIdx = 0; chunkIdx < 256; chunkIdx++)
        {
            int mcnkOff = mcnkOffsets[chunkIdx];
            if (mcnkOff <= 0) continue;

            try
            {
                // Read Alpha MCNK header (128 bytes)
                wdtStream.Seek(mcnkOff + ChunkLettersAndSize, SeekOrigin.Begin);
                var hdrBytes = new byte[McnkHeaderSize];
                if (wdtStream.Read(hdrBytes, 0, McnkHeaderSize) < McnkHeaderSize)
                    continue;

                // Alpha MCNK header: IndexX at 0x04, IndexY at 0x08, MCVT offset at 0x18
                int idxX = BitConverter.ToInt32(hdrBytes, 0x04);
                int idxY = BitConverter.ToInt32(hdrBytes, 0x08);
                int mcvtOffsetInChunk = BitConverter.ToInt32(hdrBytes, 0x18);

                // Read MCVT data (skip chunk header at mcvtOffset)
                int mcvtDataOff = mcnkOff + ChunkLettersAndSize + McnkHeaderSize + mcvtOffsetInChunk + ChunkLettersAndSize;
                wdtStream.Seek(mcvtDataOff, SeekOrigin.Begin);
                
                var mcvtData = new byte[McvtFloatCount * 4]; // 145 floats
                if (wdtStream.Read(mcvtData, 0, mcvtData.Length) < mcvtData.Length)
                    continue;

                // Alpha MCVT layout: 81 outer floats (9x9), then 64 inner floats (8x8)
                // Note: Alpha stores outer then inner (not interleaved like LK)
                int rowBase = idxY * 8;
                int colBase = idxX * 8;

                // Outer 9x9 heights
                for (int oy = 0; oy < 9; oy++)
                {
                    for (int ox = 0; ox < 9; ox++)
                    {
                        int srcIdx = oy * 9 + ox;
                        float h = BitConverter.ToSingle(mcvtData, srcIdx * 4);
                        int rr = rowBase + oy;
                        int cc = colBase + ox;
                        if (rr >= 0 && rr < 129 && cc >= 0 && cc < 129)
                        {
                            outer129[rr, cc] = h;
                        }
                    }
                }

                // Inner 8x8 heights
                int innerOffset = 81 * 4; // After 81 outer floats
                for (int iy = 0; iy < 8; iy++)
                {
                    for (int ix = 0; ix < 8; ix++)
                    {
                        int srcIdx = iy * 8 + ix;
                        float h = BitConverter.ToSingle(mcvtData, innerOffset + srcIdx * 4);
                        int rr = rowBase + iy;
                        int cc = colBase + ix;
                        if (rr >= 0 && rr < 128 && cc >= 0 && cc < 128)
                        {
                            inner128[rr, cc] = h;
                        }
                    }
                }
            }
            catch
            {
                // Skip chunk on parse error
            }
        }

        // Downsample to WDL resolution: 17x17 outer + 16x16 inner
        var heights = new short[545];
        int pos = 0;

        // Outer 17x17: sample every 8th point from 129x129
        for (int y = 0; y < 17; y++)
        {
            int rr = Math.Min(128, y * 8);
            for (int x = 0; x < 17; x++)
            {
                int cc = Math.Min(128, x * 8);
                float v = outer129[rr, cc];
                heights[pos++] = (short)Math.Clamp(Math.Round(v), short.MinValue, short.MaxValue);
            }
        }

        // Inner 16x16: sample every 8th point from 128x128
        for (int y = 0; y < 16; y++)
        {
            int rr = y * 8;
            for (int x = 0; x < 16; x++)
            {
                int cc = x * 8;
                float v = inner128[rr, cc];
                heights[pos++] = (short)Math.Clamp(Math.Round(v), short.MinValue, short.MaxValue);
            }
        }

        return heights;
    }

    /// <summary>
    /// Map an Alpha AreaID to LK AreaID.
    /// </summary>
    public int MapAreaId(string mapName, int alphaAreaId)
    {
        return _areaCrosswalk?.MapAreaId(mapName, alphaAreaId) ?? 0;
    }

    /// <summary>
    /// Fix an asset path using listfile data.
    /// </summary>
    public string FixAssetPath(string path)
    {
        return _listfileService?.FixAssetPath(path, _options.FuzzyAssetMatching) ?? path;
    }

    /// <summary>
    /// Convert WMO v14 files to v17 format.
    /// Returns a dictionary mapping original paths to new _alpha suffixed paths.
    /// </summary>
    private Dictionary<string, string> ConvertWmoFiles(List<string> monmNames, string outputDir)
    {
        var pathMapping = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var converter = new WmoV14ToV17Converter();
        
        // Collect unique WMO paths from MONM
        var uniqueWmos = monmNames
            .Where(p => !string.IsNullOrEmpty(p) && p.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (_options.Verbose)
            Console.WriteLine($"  Found {uniqueWmos.Count} unique WMO references in MONM");

        foreach (var wmoPath in uniqueWmos)
        {
            try
            {
                // Normalize path separators
                var normalizedPath = wmoPath.Replace('/', '\\').TrimStart('\\');
                
                // Build source path
                var sourcePath = Path.Combine(_options.AlphaWmoDirectory!, normalizedPath);
                
                if (!File.Exists(sourcePath))
                {
                    // Try without "World\" prefix
                    if (normalizedPath.StartsWith("World\\", StringComparison.OrdinalIgnoreCase))
                    {
                        var withoutWorld = normalizedPath.Substring(6);
                        sourcePath = Path.Combine(_options.AlphaWmoDirectory!, withoutWorld);
                    }
                }

                if (!File.Exists(sourcePath))
                {
                    if (_options.Verbose)
                        Console.WriteLine($"    [SKIP] Source not found: {wmoPath}");
                    continue;
                }

                // Build output path with _alpha suffix
                var wmoBaseName = Path.GetFileNameWithoutExtension(normalizedPath);
                var wmoDir = Path.GetDirectoryName(normalizedPath) ?? "";
                var newWmoName = $"{wmoBaseName}_alpha.wmo";
                var newWmoPath = Path.Combine(wmoDir, newWmoName).Replace('\\', '/');
                
                // Create output directory structure
                var fullOutputDir = Path.Combine(outputDir, wmoDir);
                Directory.CreateDirectory(fullOutputDir);
                
                var fullOutputPath = Path.Combine(outputDir, wmoDir, newWmoName);
                
                if (_options.Verbose)
                    Console.WriteLine($"    Converting: {wmoPath} → {newWmoPath}");

                // Convert the WMO
                converter.Convert(sourcePath, fullOutputPath);
                
                // Record mapping
                pathMapping[wmoPath] = newWmoPath;
            }
            catch (Exception ex)
            {
                if (_options.Verbose)
                    Console.WriteLine($"    [ERROR] {wmoPath}: {ex.Message}");
            }
        }

        return pathMapping;
    }
}

/// <summary>
/// Options for Alpha to LK conversion.
/// </summary>
public class ConversionOptions
{
    /// <summary>
    /// Path to DBCTool.V2 crosswalk directory (compare/v2/).
    /// </summary>
    public string? CrosswalkDirectory { get; set; }

    /// <summary>
    /// Path to community listfile CSV.
    /// </summary>
    public string? CommunityListfile { get; set; }

    /// <summary>
    /// Path to LK 3.x listfile TXT.
    /// </summary>
    public string? LkListfile { get; set; }

    /// <summary>
    /// Enable fuzzy asset path matching by filename.
    /// </summary>
    public bool FuzzyAssetMatching { get; set; } = false;

    /// <summary>
    /// Verbose logging output.
    /// </summary>
    public bool Verbose { get; set; } = false;

    /// <summary>
    /// Path to alpha AreaTable.dbc for area name resolution.
    /// </summary>
    public string? AlphaAreaTablePath { get; set; }

    /// <summary>
    /// Path to WoWDBDefs definitions directory.
    /// </summary>
    public string? DbdDefinitionsPath { get; set; }

    /// <summary>
    /// Path to Alpha WMO root directory containing v14 WMO files.
    /// When set, references WMOs will be converted to v17 format.
    /// </summary>
    public string? AlphaWmoDirectory { get; set; }

    /// <summary>
    /// Enable automatic WMO v14→v17 conversion during ADT conversion.
    /// Converted WMOs will be renamed with _alpha suffix to avoid 3.3.5 collisions.
    /// </summary>
    public bool ConvertWmos { get; set; } = false;
}

/// <summary>
/// Result of a conversion operation.
/// </summary>
public class ConversionResult
{
    public bool Success { get; set; }
    public string? SourcePath { get; set; }
    public string? MapName { get; set; }
    public string? OutputDirectory { get; set; }
    public string? Error { get; set; }
    public long ElapsedMs { get; set; }
    public int TilesConverted { get; set; }
    public int TotalTiles { get; set; }
    public List<string> Warnings { get; set; } = new();
}
