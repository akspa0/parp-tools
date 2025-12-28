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
        
        if (!string.IsNullOrEmpty(_options.CrosswalkDirectory))
        {
            _areaCrosswalk = new AreaIdCrosswalk();
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
            var wdtLk = wdtAlpha.ToWdt();
            var wdtOutPath = Path.Combine(outputDir, $"{mapName}.wdt");
            wdtLk.ToFile(wdtOutPath);

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
